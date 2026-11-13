#!/usr/bin/env python3
"""Zero-dependency Stdio MCP server exposing `context_guard_status` and `context_guard_compact` to Antigravity agents."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Any

import context_monitor

TOOLS = [
    {
        "name": "context_guard_status",
        "description": (
            "Check the current conversation's context window usage, token count, "
            "cache hit rate, per-turn token burn rate, and severity tier "
            "(OK 0-30%, ADVISORY 30-50%, WARNING 50-70%, HIGH 70-80%, CRITICAL 80%+). "
            "Can also self-trigger compaction (`compact: true`) at the end of the current turn."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "Optional conversation ID (auto-detected from active session if omitted).",
                },
                "transcript_path": {
                    "type": "string",
                    "description": "Optional path to transcript.jsonl (auto-discovered for active conversation if omitted).",
                },
                "context_limit": {
                    "type": "integer",
                    "description": "Optional context limit override in tokens (default 200000 soft target).",
                },
                "compact": {
                    "type": "boolean",
                    "description": "Set to true to self-trigger compaction at the end of this turn.",
                },
                "session_summary": {
                    "type": "string",
                    "description": "Optional summary notes to include when `compact: true`.",
                },
                "continuation_prompt": {
                    "type": "string",
                    "description": "Continuation instruction passed to `/compact <prompt>` when `compact: true`.",
                },
                "auto_continue": {
                    "type": "boolean",
                    "description": "Whether to continue immediately after compaction (default: true).",
                },
            },
        },
    },
    {
        "name": "context_guard_compact",
        "description": (
            "Self-trigger native trajectory compaction and continuation at the end of the current turn. "
            "When called, `PostInvocation` injects `/compact <continuation_prompt>` with "
            "`terminationBehavior: force_continue`, causing Antigravity's built-in `ContextSummaryHook` "
            "to compact the trajectory into a `CHECKPOINT` step and immediately continue."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_summary": {
                    "type": "string",
                    "description": "Optional brief context note prepended to the continuation prompt.",
                },
                "continuation_prompt": {
                    "type": "string",
                    "description": "Instruction to execute immediately after compaction completes.",
                },
                "conversation_id": {
                    "type": "string",
                    "description": "Optional conversation ID (auto-detected from active session if omitted).",
                },
                "auto_continue": {
                    "type": "boolean",
                    "description": "Set to true (default) to compact and continue immediately, or false to compact only.",
                },
            },
        },
    },
]


def handle_request(req: dict[str, Any]) -> dict[str, Any] | None:
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "agy-context-guard", "version": "0.2.0"},
            },
        }

    if method.startswith("notifications/"):
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        importlib.reload(context_monitor)
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "context_guard_compact":
            res = context_monitor.queue_compaction(
                session_summary=str(args.get("session_summary", "")),
                continuation_prompt=str(
                    args.get(
                        "continuation_prompt",
                        "Context compacted. Continue from the checkpoint summary above.",
                    )
                ),
                conversation_id=str(args.get("conversation_id", "")),
                auto_continue=bool(args.get("auto_continue", True)),
            )
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(res, indent=2)}],
                    "isError": False,
                },
            }
        if name != "context_guard_status":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {name}"},
            }
        cid = str(args.get("conversation_id", "")).strip()
        tp_str = args.get("transcript_path")
        transcript_path = (
            Path(tp_str)
            if tp_str
            else context_monitor.find_transcript_for_conversation(cid)
        )
        limit = int(args.get("context_limit") or context_monitor.resolve_context_limit())
        resolved_cid = context_monitor.get_active_conversation_id(cid)
        metrics = context_monitor.parse_transcript_metrics(transcript_path, context_limit=limit)
        metrics["conversation_id"] = resolved_cid
        if args.get("compact"):
            metrics["compaction"] = context_monitor.queue_compaction(
                session_summary=str(args.get("session_summary", "")),
                continuation_prompt=str(
                    args.get(
                        "continuation_prompt",
                        "Context compacted. Continue from the checkpoint summary above.",
                    )
                ),
                conversation_id=resolved_cid,
                auto_continue=bool(args.get("auto_continue", True)),
            )
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(metrics, indent=2),
                    }
                ],
                "isError": False,
            },
        }

    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    return None


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
