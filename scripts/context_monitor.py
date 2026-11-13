#!/usr/bin/env python3
"""Lightweight context window monitor and self-triggered compaction hook for Antigravity.

Reads exact token counts (`input_tokens`, `cache_read_tokens`, `output_tokens`) from
the tail of the conversation's `transcript.jsonl` in O(1) time without external
dependencies. Supports both lifecycle hook execution (`--hook`, `--hook --post`) and
on-demand CLI/MCP querying (`--status`, `--compact`).

Self-Triggered Compaction Architecture:
- Never uses `tmux send-keys` or emits `injectSteps: [{"checkpoint": ...}]` (which crashes
  external JSON hooks due to nil `trajectoryHelper` in `jsonhook.WrapMixinWithJSONHooks`).
- Does NOT arm passive continuity handoffs waiting for manual user `/compact`.
- When the agent calls `context_guard_compact`, it stages a one-shot trigger for the active
  invocation (`compact_trigger_<conversation_id>.json`). At the end of that turn, `PostInvocation`
  immediately consumes and deletes the trigger file, returning:
    `{"injectSteps": [{"userMessage": "/compact ..."}], "terminationBehavior": "force_continue"}`
  which starts the next invocation where Antigravity's built-in `ContextSummaryHook` sees the
  `/compact <prompt>` `USER_INPUT` step, compacts the trajectory into a `CHECKPOINT` step, and
  continues execution with the agent's continuation prompt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

DEFAULT_CONTEXT_LIMIT = 200_000
MAX_STATE_FILES = 30
TRIGGER_TTL_SECONDS = 300.0

MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "default": 200_000,
    "auto": 200_000,
    "argon": 200_000,
    "gemini": 200_000,
    "claude": 200_000,
}

# Progressive severity tiers:
# (tier_id, min_pct, max_pct, min_step_interval, min_pct_delta_for_repeat)
TIERS = [
    ("OK", 0.0, 30.0, 0, 100.0),
    ("ADVISORY", 30.0, 50.0, 6, 6.0),
    ("WARNING", 50.0, 70.0, 3, 4.0),
    ("HIGH", 70.0, 80.0, 2, 2.5),
    ("CRITICAL", 80.0, 1000.0, 1, 0.0),
]


def load_local_config() -> dict[str, Any]:
    """Load optional user-local config from ~/.gemini/config/context_guard.local.json or $PLUGIN_DATA/config.json."""
    candidates: list[Path] = []
    env_cfg = os.environ.get("AGY_CONTEXT_GUARD_LOCAL_CONFIG", "").strip()
    if env_cfg:
        candidates.append(Path(env_cfg))
    candidates.append(Path.home() / ".gemini" / "config" / "context_guard.local.json")
    plugin_data = os.environ.get("PLUGIN_DATA", "").strip()
    if plugin_data:
        candidates.append(Path(plugin_data) / "config.json")
    for cfg_path in candidates:
        if cfg_path.is_file():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return {}


def resolve_context_limit(model_name: str = "auto") -> int:
    """Return the configured context window limit in tokens (default 200,000 soft target)."""
    env_limit = os.environ.get("AGY_CONTEXT_LIMIT")
    if env_limit:
        try:
            val = int(env_limit)
            if val > 0:
                return val
        except ValueError:
            pass
    cfg = load_local_config()
    try:
        cfg_lim = int(cfg.get("context_limit", 0))
        if cfg_lim > 0 and cfg_lim != 1_000_000:
            return cfg_lim
    except Exception:
        pass
    low = model_name.lower()
    for key, lim in MODEL_CONTEXT_LIMITS.items():
        if key in low:
            return lim
    return DEFAULT_CONTEXT_LIMIT


def _read_tail_lines(path: Path, max_bytes: int = 262_144) -> list[str]:
    """Read the last `max_bytes` of a JSONL file and return complete lines."""
    if not path.is_file():
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                raw = fh.read()
                first_nl = raw.find(b"\n")
                if first_nl != -1:
                    raw = raw[first_nl + 1 :]
            else:
                raw = fh.read()
        return [ln for ln in raw.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    except OSError:
        return []


def _base_data_dir() -> Path:
    env_dir = os.environ.get("PLUGIN_DATA")
    if env_dir:
        base_dir = Path(env_dir)
    else:
        default_plugin_data = (
            Path.home() / ".gemini" / "antigravity-cli" / "plugin_data" / "agy-context-guard"
        )
        try:
            default_plugin_data.mkdir(parents=True, exist_ok=True)
            base_dir = default_plugin_data
        except OSError:
            base_dir = Path(tempfile.gettempdir()) / "agy-context-guard"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _prune_state_files(max_files: int = MAX_STATE_FILES) -> None:
    """Keep at most `max_files` state JSON files in `state/` (newest by mtime)."""
    try:
        state_dir = _base_data_dir() / "state"
        if not state_dir.is_dir():
            return
        files = [
            f
            for f in state_dir.glob("*.json")
            if f.is_file() and f.name != "active_conversation.json"
        ]
        if len(files) <= max_files:
            return
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[max_files:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass


def _safe_conv_id(conversation_id: str) -> str:
    return "".join(c for c in conversation_id if c.isalnum() or c in "-_") or "default"


def _state_file_for_conversation(conversation_id: str) -> Path:
    state_dir = _base_data_dir() / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{_safe_conv_id(conversation_id)}.json"


def _trigger_file_for_conversation(conversation_id: str) -> Path:
    state_dir = _base_data_dir() / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"compact_trigger_{_safe_conv_id(conversation_id)}.json"


def _active_conversation_file() -> Path:
    state_dir = _base_data_dir() / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "active_conversation.json"


def record_active_conversation(conversation_id: str, transcript_path: Path | None = None) -> None:
    """Persist the most recently active conversation ID from PreInvocation so MCP/CLI tools target the exact session."""
    if not conversation_id or conversation_id == "default":
        return
    tp = transcript_path or find_transcript_for_conversation(conversation_id)
    data = {
        "conversation_id": conversation_id,
        "transcript_path": str(tp) if tp else "",
        "updated_at": time.time(),
    }
    try:
        _active_conversation_file().write_text(json.dumps(data), encoding="utf-8")
        _prune_state_files()
    except OSError:
        pass


def get_active_conversation_id(explicit_id: str = "") -> str:
    """Resolve conversation ID from explicit argument, hook environment, or PreInvocation state."""
    if explicit_id and explicit_id != "default":
        return explicit_id
    env_id = os.environ.get("ANTIGRAVITY_CONVERSATION_ID", "").strip()
    if env_id:
        return env_id
    active_file = _active_conversation_file()
    if active_file.is_file():
        try:
            data = json.loads(active_file.read_text(encoding="utf-8"))
            cid = str(data.get("conversation_id", "")).strip()
            if cid:
                return cid
        except Exception:
            pass
    latest = find_latest_transcript()
    if latest is not None:
        try:
            return latest.parents[2].name
        except IndexError:
            pass
    return "default"


def find_transcript_for_conversation(conversation_id: str = "") -> Path | None:
    """Locate transcript.jsonl for a specific conversation ID in O(1), falling back to latest only if unknown."""
    cid = get_active_conversation_id(conversation_id)
    home = Path.home() / ".gemini"
    if cid and cid != "default" and home.is_dir():
        for app_dir in ("antigravity-cli", "antigravity", "antigravity-ide"):
            candidate = (
                home / app_dir / "brain" / cid / ".system_generated" / "logs" / "transcript.jsonl"
            )
            if candidate.is_file():
                return candidate
    active_file = _active_conversation_file()
    if active_file.is_file():
        try:
            data = json.loads(active_file.read_text(encoding="utf-8"))
            tp_str = str(data.get("transcript_path", "")).strip()
            if tp_str and Path(tp_str).is_file():
                return Path(tp_str)
        except Exception:
            pass
    return find_latest_transcript()


def find_latest_transcript() -> Path | None:
    """Fallback auto-discovery of the most recently modified transcript.jsonl under ~/.gemini."""
    home = Path.home() / ".gemini"
    if not home.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for app_dir in ("antigravity-cli", "antigravity", "antigravity-ide"):
        brain_dir = home / app_dir / "brain"
        if not brain_dir.is_dir():
            continue
        for transcript in brain_dir.glob("*/.system_generated/logs/transcript.jsonl"):
            try:
                candidates.append((transcript.stat().st_mtime, transcript))
            except OSError:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def parse_transcript_metrics(
    transcript_path: Path | None,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
) -> dict[str, Any]:
    """Extract current context token usage, cache hit ratio, burn rate, and latest checkpoint step from transcript.jsonl."""
    if transcript_path is None or not transcript_path.is_file():
        return {
            "available": False,
            "context_tokens": 0,
            "context_limit": context_limit,
            "usage_pct": 0.0,
            "tier": "OK",
            "input_tokens": 0,
            "cache_read_tokens": 0,
            "output_tokens": 0,
            "cache_hit_pct": 0.0,
            "avg_tokens_per_turn": 0,
            "est_turns_to_50pct": None,
            "est_turns_to_70pct": None,
            "step_index": 0,
            "last_checkpoint_step": -1,
            "planner_turns": 0,
        }

    lines = _read_tail_lines(transcript_path)
    planner_steps: list[dict[str, int]] = []
    last_step_idx = 0
    last_checkpoint_step = -1

    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        step_idx = int(obj.get("step_index", 0))
        if step_idx > last_step_idx:
            last_step_idx = step_idx
        step_type = str(obj.get("type", ""))
        if step_type in ("CHECKPOINT", "CORTEX_STEP_TYPE_CHECKPOINT", "COMPACTION"):
            if step_idx > last_checkpoint_step:
                last_checkpoint_step = step_idx
        if step_type == "PLANNER_RESPONSE" and (
            "input_tokens" in obj or "cache_read_tokens" in obj
        ):
            inp = int(obj.get("input_tokens", 0))
            cache = int(obj.get("cache_read_tokens", 0))
            out = int(obj.get("output_tokens", 0))
            planner_steps.append(
                {
                    "step_index": step_idx,
                    "input_tokens": inp,
                    "cache_read_tokens": cache,
                    "output_tokens": out,
                    "total_tokens": inp + cache + out,
                }
            )

    if not planner_steps:
        approx_tokens = int(transcript_path.stat().st_size // 4)
        pct = (approx_tokens / max(context_limit, 1)) * 100.0
        return {
            "available": True,
            "estimated": True,
            "context_tokens": approx_tokens,
            "context_limit": context_limit,
            "usage_pct": round(pct, 2),
            "tier": classify_tier(pct),
            "input_tokens": approx_tokens,
            "cache_read_tokens": 0,
            "output_tokens": 0,
            "cache_hit_pct": 0.0,
            "avg_tokens_per_turn": 0,
            "est_turns_to_50pct": None,
            "est_turns_to_70pct": None,
            "step_index": last_step_idx,
            "last_checkpoint_step": last_checkpoint_step,
            "planner_turns": 0,
        }

    latest = planner_steps[-1]
    total_tokens = latest["total_tokens"]
    prompt_tokens = latest["input_tokens"] + latest["cache_read_tokens"]
    cache_hit_pct = (
        (latest["cache_read_tokens"] / prompt_tokens) * 100.0 if prompt_tokens > 0 else 0.0
    )

    # Calculate recent growth rate per planner turn over up to last 5 turns
    window = planner_steps[-6:]
    deltas = [
        window[i]["total_tokens"] - window[i - 1]["total_tokens"]
        for i in range(1, len(window))
        if window[i]["total_tokens"] >= window[i - 1]["total_tokens"]
    ]
    avg_burn = int(sum(deltas) / len(deltas)) if deltas else 0

    pct = (total_tokens / max(context_limit, 1)) * 100.0
    tier = classify_tier(pct)

    target_50 = int(context_limit * 0.50)
    target_70 = int(context_limit * 0.70)
    est_to_50 = (
        max(0, (target_50 - total_tokens) // avg_burn) if avg_burn > 0 else None
    )
    est_to_70 = (
        max(0, (target_70 - total_tokens) // avg_burn) if avg_burn > 0 else None
    )

    return {
        "available": True,
        "estimated": False,
        "context_tokens": total_tokens,
        "context_limit": context_limit,
        "usage_pct": round(pct, 2),
        "tier": tier,
        "input_tokens": latest["input_tokens"],
        "cache_read_tokens": latest["cache_read_tokens"],
        "output_tokens": latest["output_tokens"],
        "cache_hit_pct": round(cache_hit_pct, 1),
        "avg_tokens_per_turn": avg_burn,
        "est_turns_to_50pct": est_to_50,
        "est_turns_to_70pct": est_to_70,
        "step_index": last_step_idx,
        "last_checkpoint_step": last_checkpoint_step,
        "planner_turns": len(planner_steps),
        "recommendation": format_recommendation(tier, pct, total_tokens, context_limit),
    }


def classify_tier(pct: float) -> str:
    """Map usage percentage to severity tier."""
    for name, lo, hi, _, _ in TIERS:
        if lo <= pct < hi:
            return name
    return "CRITICAL"


def format_recommendation(tier: str, pct: float, tokens: int, limit: int) -> str:
    """Return concise guidance message for the given severity tier."""
    tok_k = f"{tokens / 1000:.1f}k"
    lim_k = f"{limit / 1000:.0f}k"
    if tier == "OK":
        return (
            f"[Context Guard: {pct:.1f}% ({tok_k}/{lim_k} tok) - OK] "
            "Context usage is healthy (0-30%). Proceed normally."
        )
    if tier == "ADVISORY":
        return (
            f"[Context Guard: {pct:.1f}% ({tok_k}/{lim_k} tok) - ADVISORY (30-50%)] "
            "Consider documenting progress and calling `context_guard_compact` to self-trigger "
            "compaction unless focused on finishing an active task."
        )
    if tier == "WARNING":
        return (
            f"[Context Guard: {pct:.1f}% ({tok_k}/{lim_k} tok) - WARNING (50-70%)] "
            "Above the 50% efficiency target. Finish the current step, update continuity notes, and "
            "call `context_guard_compact` to self-trigger compaction and continue."
        )
    if tier == "HIGH":
        return (
            f"[Context Guard: {pct:.1f}% ({tok_k}/{lim_k} tok) - HIGH (70-80%)] "
            "Come to a documenting/check-in point now and call `context_guard_compact`."
        )
    return (
        f"[Context Guard: {pct:.1f}% ({tok_k}/{lim_k} tok) - CRITICAL (80%+)] "
        "URGENT: Do not start new tasks. Document current state immediately and call `context_guard_compact`."
    )


def format_compact_cli_command(req: dict[str, Any]) -> str:
    """Format the single-line `/compact` command string for native `ParseCompactRequest`.

    - `/compact <continuation_prompt>` triggers `CompactRequest_COMPACT_AND_CONTINUE` (2).
    - `/compact ` triggers `CompactRequest_COMPACT_ONLY` (1).
    """
    auto_continue = bool(req.get("auto_continue", True))
    cont_prompt = " ".join(str(req.get("continuation_prompt", "")).split()).strip()
    summary = " ".join(str(req.get("session_summary", "")).split()).strip()
    if not auto_continue:
        return "/compact "
    if not cont_prompt:
        cont_prompt = "Context compacted. Continue from the checkpoint summary above."
    if summary:
        cont_prompt = f"[Context Note: {summary}] {cont_prompt}"
    return f"/compact {cont_prompt}"


def queue_compaction(
    session_summary: str = "",
    continuation_prompt: str = "Context compacted. Continue from the checkpoint summary above.",
    conversation_id: str = "",
    auto_continue: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Self-trigger native compaction at the end of the current agent turn (`PostInvocation`).

    Writes a one-shot trigger file for the active conversation ID. When the current agent
    invocation finishes, `PostInvocation` immediately consumes and deletes the trigger file,
    injecting `{"userMessage": "/compact ..."}` with `"terminationBehavior": "force_continue"`.
    Antigravity's built-in `ContextSummaryHook` then compacts the trajectory into a `CHECKPOINT`
    step and continues with `continuation_prompt`.
    """
    resolved_cid = get_active_conversation_id(conversation_id)
    req = {
        "conversation_id": resolved_cid,
        "session_summary": session_summary.strip(),
        "continuation_prompt": continuation_prompt.strip(),
        "auto_continue": bool(auto_continue),
        "queued_at": time.time(),
    }
    cmd_text = format_compact_cli_command(req)

    triggered = False
    trigger_path = _trigger_file_for_conversation(resolved_cid)
    if not dry_run:
        try:
            trigger_path.write_text(json.dumps(req, indent=2), encoding="utf-8")
            triggered = True
        except OSError:
            triggered = False

    return {
        "triggered": triggered,
        "conversation_id": resolved_cid,
        "auto_continue": bool(auto_continue),
        "mode": "compact_and_continue" if auto_continue else "compact_only",
        "compact_command": cmd_text,
        "message": (
            "Self-triggered compaction queued for the end of this turn (`PostInvocation`). "
            "Finish your response without calling more tools; `PostInvocation` will immediately "
            "inject `/compact` and continue in the compacted context."
        ),
    }


def consume_post_invocation_compact_trigger(conversation_id: str) -> dict[str, Any]:
    """Consume the one-shot self-compaction trigger on `PostInvocation` and return the hook response."""
    trigger_path = _trigger_file_for_conversation(conversation_id)
    if not trigger_path.is_file():
        return {}
    try:
        req = json.loads(trigger_path.read_text(encoding="utf-8"))
    except Exception:
        req = None
    finally:
        try:
            trigger_path.unlink(missing_ok=True)
        except OSError:
            pass

    if not isinstance(req, dict):
        return {}
    queued_at = float(req.get("queued_at", 0.0))
    if queued_at > 0 and (time.time() - queued_at) > TRIGGER_TTL_SECONDS:
        return {}

    cmd_text = format_compact_cli_command(req)
    auto_continue = bool(req.get("auto_continue", True))
    resp: dict[str, Any] = {
        "injectSteps": [
            {
                "userMessage": cmd_text,
            }
        ],
    }
    if auto_continue:
        resp["terminationBehavior"] = "force_continue"
    return resp


def should_emit_notification(
    conversation_id: str,
    metrics: dict[str, Any],
    invocation_num: int,
) -> bool:
    """Decide whether to inject a notification on this turn to keep overhead minimal."""
    tier = metrics["tier"]
    pct = float(metrics["usage_pct"])
    tokens = int(metrics["context_tokens"])

    state_path = _state_file_for_conversation(conversation_id)
    prev: dict[str, Any] = {}
    if state_path.is_file():
        try:
            prev = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    # Detect compaction (token count dropped by >20%)
    prev_tokens = int(prev.get("last_seen_tokens", 0))
    if prev_tokens > 0 and tokens < prev_tokens * 0.8:
        prev = {}

    if tier == "OK":
        state_path.write_text(
            json.dumps(
                {
                    "last_notified_tier": "OK",
                    "last_notified_pct": pct,
                    "last_notified_invocation": invocation_num,
                    "last_seen_tokens": tokens,
                }
            ),
            encoding="utf-8",
        )
        return False

    tier_order = {"OK": 0, "ADVISORY": 1, "WARNING": 2, "HIGH": 3, "CRITICAL": 4}
    prev_tier = prev.get("last_notified_tier", "OK")
    prev_notified_pct = float(prev.get("last_notified_pct", 0.0))
    prev_inv = int(prev.get("last_notified_invocation", -999))

    min_steps = 1
    min_pct_delta = 0.0
    for name, _, _, step_iv, pct_iv in TIERS:
        if name == tier:
            min_steps = step_iv
            min_pct_delta = pct_iv
            break

    emit = False
    if tier_order.get(tier, 0) > tier_order.get(prev_tier, 0):
        emit = True
    elif (invocation_num - prev_inv) >= min_steps and (pct - prev_notified_pct) >= min_pct_delta:
        emit = True
    elif tier == "CRITICAL":
        emit = True

    next_state = {
        "last_notified_tier": tier if emit else prev_tier,
        "last_notified_pct": pct if emit else prev_notified_pct,
        "last_notified_invocation": invocation_num if emit else prev_inv,
        "last_seen_tokens": tokens,
    }
    try:
        state_path.write_text(json.dumps(next_state), encoding="utf-8")
    except OSError:
        pass
    return emit


def handle_hook(payload: dict[str, Any], is_post: bool = False) -> dict[str, Any]:
    """Process a PreInvocation or PostInvocation hook payload and return JSON response."""
    raw_cid = str(
        payload.get("conversationId")
        or os.environ.get("ANTIGRAVITY_CONVERSATION_ID")
        or "default"
    ).strip()
    transcript_str = str(payload.get("transcriptPath", "")).strip()
    transcript_path = (
        Path(transcript_str)
        if transcript_str
        else find_transcript_for_conversation(raw_cid)
    )
    conversation_id = get_active_conversation_id(raw_cid)
    record_active_conversation(conversation_id, transcript_path)

    if is_post or "PostInvocation" in str(payload.get("hookEventName", "")):
        return consume_post_invocation_compact_trigger(conversation_id)

    # Clear any stale unconsumed trigger file at the start of a new turn
    stale_trigger = _trigger_file_for_conversation(conversation_id)
    if stale_trigger.is_file():
        try:
            stale_trigger.unlink(missing_ok=True)
        except OSError:
            pass

    if os.environ.get("AGY_CONTEXT_NOTIFICATIONS", "1") == "0":
        return {}

    model_name = str(payload.get("modelName", "auto"))
    invocation_num = int(payload.get("invocationNum", payload.get("stepIdx", 1)))
    limit = resolve_context_limit(model_name)
    metrics = parse_transcript_metrics(transcript_path, context_limit=limit)

    if not metrics.get("available") or not should_emit_notification(
        conversation_id, metrics, invocation_num
    ):
        return {}

    msg = metrics["recommendation"]
    return {
        "injectSteps": [
            {
                "systemMessage": {
                    "systemMessage": msg,
                    "metadata": {
                        "user_markdown": f"*{msg}*",
                    },
                }
            }
        ]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Antigravity Context Guard")
    parser.add_argument("--hook", action="store_true", help="Run as an Antigravity lifecycle hook")
    parser.add_argument("--post", action="store_true", help="Run as a PostInvocation hook")
    parser.add_argument("--status", action="store_true", help="Print current context metrics as JSON")
    parser.add_argument("--compact", action="store_true", help="Self-trigger compaction at the end of this turn")
    parser.add_argument("--summary", type=str, default="", help="Optional session summary for --compact")
    parser.add_argument(
        "--continue-prompt",
        type=str,
        default="Context compacted. Continue from the checkpoint summary above.",
        help="Continuation prompt after compaction",
    )
    parser.add_argument("--conversation-id", type=str, default="", help="Optional conversationId for --compact or --status")
    parser.add_argument("--no-continue", action="store_true", help="Compact without auto-continuation")
    parser.add_argument("--transcript", type=str, default="", help="Explicit path to transcript.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Override context token limit")
    args = parser.parse_args(argv)

    if args.hook:
        raw = sys.stdin.read().strip()
        payload = json.loads(raw) if raw else {}
        result = handle_hook(payload, is_post=args.post)
        sys.stdout.write(json.dumps(result) + "\n")
        return 0

    if args.compact:
        res = queue_compaction(
            session_summary=args.summary,
            continuation_prompt=args.continue_prompt,
            conversation_id=args.conversation_id,
            auto_continue=not args.no_continue,
        )
        sys.stdout.write(json.dumps(res, indent=2) + "\n")
        return 0

    transcript_path = (
        Path(args.transcript)
        if args.transcript
        else find_transcript_for_conversation(args.conversation_id)
    )
    limit = args.limit if args.limit > 0 else resolve_context_limit()
    resolved_cid = get_active_conversation_id(args.conversation_id)
    metrics = parse_transcript_metrics(transcript_path, context_limit=limit)
    metrics["conversation_id"] = resolved_cid
    sys.stdout.write(json.dumps(metrics, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
