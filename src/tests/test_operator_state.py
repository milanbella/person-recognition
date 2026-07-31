import time
import unittest

from pipeline.observer_api import (
    ObservedBody,
    ObservedPerson,
    ObserverCameraSnapshot,
)
from pipeline.operator_models import ObservationReference
from pipeline.operator_state import OperatorState


def person(
    *,
    track_id: int,
    visit_id: int | None,
    customer_id: str | None = None,
) -> ObservedPerson:
    return ObservedPerson(
        track_id=track_id,
        track_status="TRACKED",
        detection_score=0.8,
        visit_id=visit_id,
        visit_origin=None if visit_id is None else "entrance_confirmed",
        customer_id=customer_id,
        customer_binding_status="bound" if customer_id else "pending",
        bounding_box=(10, 20, 110, 220),
        centroid=(60.0, 120.0),
        depth=None,
        face_identity_ids=(),
        body=ObservedBody(
            has_appearance=True,
            aspect_ratio=0.5,
            height_pixels=200,
        ),
        matched_score=0.7,
        match_state="matched",
    )


def snapshot(
    *,
    sequence: int,
    people: tuple[ObservedPerson, ...],
) -> ObserverCameraSnapshot:
    return ObserverCameraSnapshot(
        camera_index=0,
        device_id="camera-a",
        camera_role="observer",
        rgb_sequence_number=sequence,
        host_synced_seconds=float(sequence),
        published_at_unix_milliseconds=10**15,
        frame_width=640,
        frame_height=360,
        observations=people,
    )


class OperatorStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = OperatorState(
            camera_device_ids=["camera-a"],
            camera_roles=["observer"],
            camera_timeout_seconds=3.0,
            event_ring_size=3,
        )

    def test_observer_changes_emit_transitions(self) -> None:
        self.state.publish_observer_snapshot(
            snapshot(sequence=1, people=(person(track_id=7, visit_id=None),))
        )
        self.state.publish_observer_snapshot(
            snapshot(
                sequence=2,
                people=(person(track_id=7, visit_id=12, customer_id="184"),),
            )
        )
        self.state.publish_observer_snapshot(snapshot(sequence=3, people=()))

        events, resync = self.state.events_after(0)
        self.assertTrue(resync)
        self.assertEqual(
            [event.event_type for event in events],
            [
                "visit_assignment_changed",
                "customer_binding_changed",
                "track_disappeared",
            ],
        )

    def test_resolves_recent_observation_after_new_frame(self) -> None:
        current = snapshot(
            sequence=8,
            people=(person(track_id=3, visit_id=4),),
        )
        self.state.publish_observer_snapshot(current)
        reference = ObservationReference(
            camera_index=0,
            device_id="camera-a",
            rgb_sequence_number=8,
            host_synced_seconds=8.0,
            track_id=3,
            observed_visit_id=4,
        )
        payload = self.state.resolve_observation(reference)
        self.assertEqual(payload["observation"]["visitId"], 4)

        self.state.publish_observer_snapshot(
            snapshot(sequence=9, people=())
        )
        payload = self.state.resolve_observation(reference)
        self.assertEqual(payload["frame"]["rgbSequenceNumber"], 8)

    def test_rejects_expired_recent_observation(self) -> None:
        current = snapshot(
            sequence=8,
            people=(person(track_id=3, visit_id=4),),
        )
        current = ObserverCameraSnapshot(
            **{
                **current.__dict__,
                "published_at_unix_milliseconds": (
                    time.time_ns() // 1_000_000 - 50
                ),
            }
        )
        self.state.publish_observer_snapshot(current)
        reference = ObservationReference(
            camera_index=0,
            device_id="camera-a",
            rgb_sequence_number=8,
            host_synced_seconds=8.0,
            track_id=3,
        )
        with self.assertRaises(LookupError):
            self.state.resolve_observation(reference, max_age_milliseconds=10)

    def test_visit_state_is_available_from_first_track_snapshot(self) -> None:
        self.state.publish_observer_snapshot(
            snapshot(sequence=1, people=(person(track_id=7, visit_id=12),))
        )
        payload = self.state.state_payload(active_run=None)
        self.assertEqual(payload["visits"][0]["visitId"], 12)
        self.assertEqual(payload["visits"][0]["lastTrackId"], 7)

    def test_state_includes_recent_events_for_mobile_reload(self) -> None:
        event = self.state.publish_event(
            event_type="entry_accepted",
            visit_id=12,
            payload={"reason": "direct_crossing"},
        )

        payload = self.state.state_payload(active_run=None)

        self.assertEqual(payload["recentEvents"][-1]["eventId"], event.event_id)
        self.assertEqual(
            payload["recentEvents"][-1]["eventType"],
            "entry_accepted",
        )


if __name__ == "__main__":
    unittest.main()
