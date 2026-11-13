# Context Guard Policy & Self-Triggered Compaction

Keep context usage under **50%** of the 200,000-token soft target whenever practical to minimize token cost and latency:

- **0–30% (`OK`)**: Normal operation.
- **30–50% (`ADVISORY`)**: Consider documenting progress and self-triggering compaction via `context_guard_compact` unless you are in the middle of a focused task that genuinely needs completion.
- **50–70% (`WARNING`)**: Finish the current step, update continuity notes, and call `context_guard_compact` to compact and continue.
- **70–80%+ (`HIGH` / `CRITICAL`)**: Do not start new tasks. Immediately come to a documenting/check-in point and call `context_guard_compact`.

## Agent Self-Triggered Compaction (`context_guard_compact`)

1. Call the `context_guard_compact` MCP tool (or `context_guard_status` with `compact: true`) with your `session_summary` and `continuation_prompt`, then finish your turn without calling additional tools.
2. At the end of that turn, the `PostInvocation` hook consumes the one-shot trigger and injects `{"userMessage": "/compact ..."}` with `"terminationBehavior": "force_continue"`, causing Antigravity's built-in `ContextSummaryHook` to immediately compact the trajectory into a `CHECKPOINT` step and continue execution with your `continuation_prompt`.
3. Before starting a large multi-step task, check exact context usage and burn rate by calling the `context_guard_status` MCP tool.
