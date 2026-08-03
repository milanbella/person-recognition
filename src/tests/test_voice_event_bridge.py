import unittest

from voice_agent.event_bridge import VoiceEventBridge, event_summary


class VoiceEventBridgeTests(unittest.TestCase):
    def test_orders_filters_and_reconstructs_pending_events(self) -> None:
        bridge = VoiceEventBridge()
        bridge.refresh(
            {
                "events": [
                    {"eventId": 3, "eventType": "track_appeared", "occurredAtUnixMilliseconds": 1},
                    {"eventId": 2, "eventType": "leave_accepted", "occurredAtUnixMilliseconds": 30, "visitId": 7, "cameraIndex": 1},
                    {"eventId": 1, "eventType": "entry_accepted", "occurredAtUnixMilliseconds": 20, "visitId": 7, "cameraIndex": 0},
                ],
                "verdicts": {"1": {"annotationType": "system_event_correct"}},
            }
        )
        event = bridge.next_event()
        self.assertEqual(event["eventId"], 2)
        self.assertEqual(event_summary(event), "LEAVE detected for visit 7 on camera 2. Is that correct?")
        bridge.resolve(2)
        self.assertIsNone(bridge.next_event())

    def test_refresh_deduplicates_and_skip_advances(self) -> None:
        bridge = VoiceEventBridge()
        context = {
            "events": [
                {"eventId": 4, "eventType": "shelf_approach", "occurredAtUnixMilliseconds": 1, "visitId": 2, "payload": {"shelfId": 3}},
                {"eventId": 5, "eventType": "shelf_departure", "occurredAtUnixMilliseconds": 2, "visitId": 2, "payload": {"shelfId": 3}},
            ],
            "verdicts": {},
        }
        bridge.refresh(context)
        bridge.refresh(context)
        self.assertEqual(bridge.next_event()["eventId"], 4)
        self.assertEqual(bridge.skip_current()["eventId"], 4)
        self.assertEqual(bridge.next_event()["eventId"], 5)


if __name__ == "__main__":
    unittest.main()
