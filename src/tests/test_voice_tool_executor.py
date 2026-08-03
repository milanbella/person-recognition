import unittest

from voice_agent.event_bridge import VoiceEventBridge
from voice_agent.tool_executor import VoiceToolError, VoiceToolExecutor


class FakeOperatorClient:
    def __init__(self) -> None:
        self.annotations = []
        self.context = {
            "subjects": [{"subjectId": "milan"}],
            "events": [
                {
                    "eventId": 11,
                    "eventType": "entry_accepted",
                    "visitId": 4,
                    "payload": {},
                }
            ],
            "verdicts": {},
        }

    def state(self):
        return {
            "activeRun": {"runId": "run-1"},
            "shelves": [{"shelfId": 3}],
            "cameras": [],
            "visits": [],
        }

    def voice_context(self, run_id):
        return self.context

    def create_annotation(self, run_id, payload):
        annotation = {"annotationId": len(self.annotations) + 1, **payload}
        self.annotations.append(annotation)
        return {"annotation": annotation}

    def observation(self, camera_index):
        return {"camera": {"id": camera_index}, "observations": []}


class VoiceToolExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeOperatorClient()
        self.bridge = VoiceEventBridge()
        self.bridge.refresh(self.client.context)
        self.bridge.next_event()
        self.executor = VoiceToolExecutor(
            self.client,
            self.bridge,
            expected_run_id="run-1",
        )

    def test_confirms_existing_event_through_annotation_api(self) -> None:
        result = self.executor.execute("confirm_system_event", {"event_id": 11})
        self.assertTrue(result.payload["ok"])
        self.assertEqual(self.client.annotations[0]["annotationType"], "system_event_correct")
        self.assertIsNone(self.bridge.current_event())

    def test_records_valid_missing_shelf_approach(self) -> None:
        result = self.executor.execute(
            "report_missing_shelf_approach",
            {"shelf_id": 3},
        )
        self.assertEqual(result.payload["annotation"]["annotationType"], "shelf_approach")
        self.assertEqual(result.payload["annotation"]["shelfId"], 3)

    def test_rejects_unknown_shelf_and_changed_run(self) -> None:
        with self.assertRaises(VoiceToolError) as shelf_error:
            self.executor.execute("report_missing_shelf_leave", {"shelf_id": 99})
        self.assertEqual(shelf_error.exception.code, "unknown_shelf")
        self.client.state = lambda: {"activeRun": {"runId": "run-2"}, "shelves": []}
        with self.assertRaises(VoiceToolError) as run_error:
            self.executor.execute("report_missing_entry", {})
        self.assertEqual(run_error.exception.code, "run_changed")


if __name__ == "__main__":
    unittest.main()
