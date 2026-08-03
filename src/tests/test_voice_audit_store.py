import tempfile
import unittest
from pathlib import Path

from voice_agent.audit_store import VoiceAuditStore


class VoiceAuditStoreTests(unittest.TestCase):
    def test_tool_calls_are_idempotent_and_session_usage_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = VoiceAuditStore(Path(temporary) / "state.sqlite")
            store.start_session("voice-1", run_id="run-1", operator_id="Milan")
            store.connect_session("voice-1", "rtc_1")
            self.assertIsNone(
                store.start_tool_call(
                    "call-1",
                    voice_session_id="voice-1",
                    run_id="run-1",
                    tool_name="report_missing_entry",
                    arguments={},
                )
            )
            store.finish_tool_call(
                "call-1",
                status="completed",
                result={"ok": True},
                annotation_id=9,
            )
            duplicate = store.start_tool_call(
                "call-1",
                voice_session_id="voice-1",
                run_id="run-1",
                tool_name="report_missing_entry",
                arguments={},
            )
            self.assertEqual(duplicate["result"], {"ok": True})
            self.assertEqual(duplicate["annotationId"], 9)
            store.add_usage("voice-1", input_tokens=10, output_tokens=4)
            store.end_session("voice-1", "test_complete")


if __name__ == "__main__":
    unittest.main()
