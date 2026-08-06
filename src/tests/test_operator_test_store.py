import tempfile
import unittest
from pathlib import Path

from pipeline.operator_models import ObservationReference, OperatorEvent
from pipeline.operator_test_store import OperatorTestStore


class OperatorTestStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store = OperatorTestStore(
            root / "state.sqlite",
            runs_root=root / "runs",
            runtime_configuration={"gitCommit": "abc123"},
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def start_run(self) -> dict:
        return self.store.start_run(
            {
                "scenario": "shop-route",
                "verifier": "Milan",
                "subjects": [
                    {
                        "subjectId": "milan",
                        "displayName": "Milan",
                        "expectedCustomerId": "184",
                    }
                ],
            }
        )

    def test_run_annotation_and_physical_visit_lifecycle(self) -> None:
        run = self.start_run()
        with self.assertRaises(RuntimeError):
            self.start_run()

        entry = self.store.create_annotation(
            run["runId"],
            {"annotationType": "physical_entry", "subjectId": "milan"},
            system_snapshot={},
            observation_reference=None,
        )
        reference = ObservationReference(
            camera_index=2,
            device_id="camera-c",
            rgb_sequence_number=10,
            host_synced_seconds=1.0,
            track_id=7,
            observed_visit_id=12,
            observed_customer_id="184",
        )
        confirmation = self.store.create_annotation(
            run["runId"],
            {
                "annotationType": "observation_is_subject",
                "subjectId": "milan",
            },
            system_snapshot={"camera": 2},
            observation_reference=reference,
        )
        leave = self.store.create_annotation(
            run["runId"],
            {"annotationType": "physical_leave", "subjectId": "milan"},
            system_snapshot={},
            observation_reference=None,
        )

        self.assertEqual(entry["physicalVisitId"], confirmation["physicalVisitId"])
        self.assertEqual(entry["physicalVisitId"], leave["physicalVisitId"])
        visits = self.store.physical_visits(run["runId"])
        self.assertIsNotNone(visits[0]["leftAtUnixMilliseconds"])

        stopped = self.store.stop_run(run["runId"])
        self.assertEqual(stopped["status"], "stopped")
        self.assertTrue(
            (self.store.runs_root / run["runId"] / "annotations.jsonl").exists()
        )

    def test_events_are_persisted_only_during_active_run(self) -> None:
        before = OperatorEvent(
            event_id=1,
            event_type="entry_accepted",
            occurred_at_unix_milliseconds=100,
            source="recognition_pipeline",
        )
        self.store.append_event(before)
        run = self.start_run()
        during = OperatorEvent(
            event_id=2,
            event_type="entry_accepted",
            occurred_at_unix_milliseconds=200,
            source="recognition_pipeline",
            visit_id=4,
        )
        self.store.append_event(during)
        _annotations, events = self.store.run_rows(run["runId"])
        self.assertEqual([event["eventId"] for event in events], [2])

    def test_enqueued_events_are_durable_after_flush(self) -> None:
        run = self.start_run()
        self.store.enqueue_event(
            OperatorEvent(
                event_id=3,
                event_type="track_appeared",
                occurred_at_unix_milliseconds=300,
                source="observer_snapshot",
                camera_index=2,
                track_id=7,
                visit_id=4,
            )
        )

        self.store.flush_events()

        _annotations, events = self.store.run_rows(run["runId"])
        self.assertEqual([event["eventId"] for event in events], [3])

    def test_system_event_verdict_requires_and_preserves_event_id(self) -> None:
        run = self.start_run()
        with self.assertRaises(ValueError):
            self.store.create_annotation(
                run["runId"],
                {
                    "annotationType": "system_event_correct",
                    "subjectId": "milan",
                },
                system_snapshot={},
                observation_reference=None,
            )

        annotation = self.store.create_annotation(
            run["runId"],
            {
                "annotationType": "system_event_correct",
                "subjectId": "milan",
                "systemEventId": 23,
                "systemEventType": "entry_accepted",
            },
            system_snapshot={},
            observation_reference=None,
        )

        self.assertEqual(annotation["payload"]["systemEventId"], 23)

    def test_subject_mapping_does_not_require_a_physical_entry(self) -> None:
        run = self.start_run()

        annotation = self.store.create_annotation(
            run["runId"],
            {
                "annotationType": "subject_visit_mapping",
                "subjectId": "milan",
                "visitId": 7,
            },
            system_snapshot={},
            observation_reference=None,
        )

        self.assertIsNone(annotation["physicalVisitId"])
        self.assertEqual(self.store.subject_visit_mapping(run["runId"], "milan"), 7)
        self.assertEqual(self.store.physical_visits(run["runId"]), [])


if __name__ == "__main__":
    unittest.main()
