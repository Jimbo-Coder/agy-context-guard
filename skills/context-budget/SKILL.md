---
name: context-budget
description: >-
  Check current context window utilization, token burn rate, and self-trigger
  native compaction (`context_guard_compact`). Use when deciding whether to start
  a new task, delegate to a subagent, or compact the active conversation.
---

# Context Budget & Self-Triggered Compaction

Use this skill to inspect the conversation's exact token footprint and self-trigger native compaction via `context_guard_compact`.

## How to Check Context Usage

1. When the user invokes `/context-budget` with no additional instructions, immediately call the `context_guard_status` MCP tool (`context_limit: 200000`) and print the token readout directly—do not run shell commands or take other actions.
2. Inspect `usage_pct`, `tier`, `avg_tokens_per_turn`, and `est_turns_to_50pct`:
   - **`OK` (0–30%)**: Safe to execute directly.
   - **`ADVISORY` (30–50%)**: If starting broad research, delegate to a `research` or `self` subagent so intermediate file reads stay out of the main context; otherwise call `context_guard_compact` once the active step completes.
   - **`WARNING` (50–70%)**: Complete the active edit/test, record state in continuity docs, and call `context_guard_compact`.
   - **`HIGH` / `CRITICAL` (70%+)**: Document current state immediately and call `context_guard_compact`.

## How Self-Triggered Compaction Works

1. Call `context_guard_compact` (via `call_mcp_tool` on `agy-context-guard_agy-context-guard`, or `context_guard_status` with `compact: true`) with `session_summary` and `continuation_prompt`, then end your turn.
2. On `PostInvocation`, the hook consumes the one-shot trigger and emits `{"injectSteps": [{"userMessage": "/compact ..."}], "terminationBehavior": "force_continue"}`, which triggers Antigravity's built-in `ContextSummaryHook` to compact the trajectory and immediately continue.
