#!/usr/bin/env python3
"""Unit tests for agy-context-guard."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import context_monitor  # noqa: E402


class ContextMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.env_patch = mock.patch.dict(
            os.environ,
            {"PLUGIN_DATA": self.tmp.name, "ANTIGRAVITY_CONVERSATION_ID": ""},
            clear=False,
        )
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.tmp.cleanup()

    def test_resolve_context_limit_ignores_legacy_1m_config(self) -> None:
        cfg_path = Path(self.tmp.name) / "config.json"
        cfg_path.write_text(json.dumps({"context_limit": 1_000_000}), encoding="utf-8")
        self.assertEqual(context_monitor.resolve_context_limit("auto"), 200_000)

        cfg_path.write_text(json.dumps({"context_limit": 150_000}), encoding="utf-8")
        self.assertEqual(context_monitor.resolve_context_limit("auto"), 150_000)

    def test_parse_transcript_metrics_and_hysteresis(self) -> None:
        transcript = Path(self.tmp.name) / "transcript.jsonl"
        rows = [
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "input_tokens": 2000,
                "cache_read_tokens": 18000,
                "output_tokens": 500,
            },
            {
                "step_index": 4,
                "type": "PLANNER_RESPONSE",
                "input_tokens": 4000,
                "cache_read_tokens": 60000,
                "output_tokens": 1000,
            },
        ]
        transcript.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        metrics = context_monitor.parse_transcript_metrics(transcript, context_limit=200_000)
        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["context_tokens"], 65_000)
        self.assertEqual(metrics["usage_pct"], 32.5)
        self.assertEqual(metrics["tier"], "ADVISORY")
        self.assertEqual(metrics["avg_tokens_per_turn"], 44_500)

        conv_id = "test-conv-hysteresis"
        # Crossing from OK -> ADVISORY emits immediately
        self.assertTrue(context_monitor.should_emit_notification(conv_id, metrics, invocation_num=1))
        # Consecutive turn in same tier with small delta is suppressed
        self.assertFalse(context_monitor.should_emit_notification(conv_id, metrics, invocation_num=2))
        # Escalating to WARNING emits immediately
        warn_metrics = dict(metrics, context_tokens=110_000, usage_pct=55.0, tier="WARNING")
        self.assertTrue(context_monitor.should_emit_notification(conv_id, warn_metrics, invocation_num=3))
        # Token drop >20% (after compaction) resets state
        post_compact = dict(metrics, context_tokens=20_000, usage_pct=10.0, tier="OK")
        self.assertFalse(context_monitor.should_emit_notification(conv_id, post_compact, invocation_num=4))
        self.assertTrue(context_monitor.should_emit_notification(conv_id, metrics, invocation_num=5))

    def test_self_triggered_post_invocation_compaction(self) -> None:
        """Verify agent self-triggered compaction fires on PostInvocation and deletes the trigger immediately."""
        conv_id = "conv-self-compact-test"
        transcript = Path(self.tmp.name) / "transcript.jsonl"
        rows = [
            {
                "step_index": 10,
                "type": "PLANNER_RESPONSE",
                "input_tokens": 5000,
                "cache_read_tokens": 105000,
                "output_tokens": 1000,
            },
        ]
        transcript.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        # 1. PreInvocation records active conversation from ANTIGRAVITY_CONVERSATION_ID
        with mock.patch.dict(os.environ, {"ANTIGRAVITY_CONVERSATION_ID": conv_id}, clear=False):
            pre1 = context_monitor.handle_hook(
                {"transcriptPath": str(transcript), "hookEventName": "PreInvocation", "invocationNum": 1},
                is_post=False,
            )
        self.assertEqual(context_monitor.get_active_conversation_id(), conv_id)
        self.assertIn("WARNING", json.dumps(pre1))

        # 2. Agent calls queue_compaction during the turn
        res = context_monitor.queue_compaction(
            session_summary="Completed module refactor.",
            continuation_prompt="Run verification suite next.",
            auto_continue=True,
        )
        self.assertTrue(res["triggered"])
        self.assertEqual(res["conversation_id"], conv_id)

        # 3. PostInvocation at the end of that turn consumes the trigger once and emits /compact + force_continue
        post1 = context_monitor.handle_hook(
            {"conversationId": conv_id, "transcriptPath": str(transcript), "hookEventName": "PostInvocation"},
            is_post=True,
        )
        self.assertEqual(post1.get("terminationBehavior"), "force_continue")
        self.assertEqual(
            post1.get("injectSteps"),
            [{"userMessage": "/compact [Context Note: Completed module refactor.] Run verification suite next."}],
        )

        # 4. Trigger file is deleted immediately; subsequent PostInvocation is a no-op
        post2 = context_monitor.handle_hook(
            {"conversationId": conv_id, "transcriptPath": str(transcript), "hookEventName": "PostInvocation"},
            is_post=True,
        )
        self.assertEqual(post2, {})

    def test_format_compact_cli_command_modes(self) -> None:
        cmd_stop = context_monitor.format_compact_cli_command(
            {"auto_continue": False, "continuation_prompt": "ignored"}
        )
        self.assertEqual(cmd_stop, "/compact ")

        cmd_cont = context_monitor.format_compact_cli_command(
            {
                "auto_continue": True,
                "session_summary": " plugins done ",
                "continuation_prompt": "Check in with user",
            }
        )
        self.assertEqual(cmd_cont, "/compact [Context Note: plugins done] Check in with user")


if __name__ == "__main__":
    unittest.main()
