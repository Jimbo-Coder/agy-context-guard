# agy-context-guard

Lightweight context window monitor and self-triggered compaction plugin for Google Antigravity (`agy`).

## Features

1. **Zero-Overhead Progressive Notifications (`PreInvocation` hook)**:
   Reads exact token counts (`input_tokens + cache_read_tokens + output_tokens`) from the tail of the active conversation's `transcript.jsonl` in $O(1)$ time (resolved deterministically via `ANTIGRAVITY_CONVERSATION_ID`) and injects brief, throttled notifications only when crossing severity thresholds:
   - **0–30% (`OK`)**: Completely silent (0 tokens added).
   - **30–50% (`ADVISORY`)**: Advises documenting progress and self-triggering compaction unless finishing an active task.
   - **50–70% (`WARNING`)**: Warns that cost scales past 50%; prompts the agent to finish the current step, update continuity notes, and call `context_guard_compact`.
   - **70–80%+ (`HIGH` / `CRITICAL`)**: Instructs the agent to immediately come to a check-in/documenting point and call `context_guard_compact`.

2. **On-Demand Agent Self-Check & Self-Triggered Compaction (`context_guard_status` + `context_guard_compact` MCP tools + CLI)**:
   Exposes a zero-dependency Stdio MCP server (`scripts/mcp_context_server.py`) so the agent can:
   - Inspect its exact token count, prompt cache hit rate, average token burn rate per turn, and estimated turns until 50% / 70% before starting large tasks.
   - Self-trigger native trajectory compaction and continuation via `context_guard_compact`. At the end of the turn (`PostInvocation`), the hook consumes the one-shot trigger and returns `{"injectSteps": [{"userMessage": "/compact ..."}], "terminationBehavior": "force_continue"}`, causing Antigravity's built-in `ContextSummaryHook` to compact the trajectory into a `CHECKPOINT` step and continue—with zero `tmux` or terminal dependencies.

## Installation

```bash
git clone https://github.com/Jimbo-Coder/agy-context-guard ~/.gemini/config/plugins/agy-context-guard
```

Companion plugin: [agy-permission-reviewer](https://github.com/Jimbo-Coder/agy-permission-reviewer) (3-tier context-aware `PreToolUse` permission reviewer).

## Optional Local Configuration

Copy [context_guard.local.example.json](context_guard.local.example.json) to `~/.gemini/config/context_guard.local.json` (or set `AGY_CONTEXT_LIMIT` / `AGY_CONTEXT_NOTIFICATIONS` in your environment):

```json
{
  "context_limit": 200000,
  "notifications": true
}
```
