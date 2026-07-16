import math
import unittest

import numpy as np

from pipeline.visit_identity import BodyAppearance
from pipeline.visit_registry import (
    CAMERA_ROLE_ENTRANCE_OBSERVER,
    CAMERA_ROLE_OBSERVER,
    TrackVisitEvidence,
    VisitRegistry,
    VISIT_ORIGIN_OBSERVER,
)


def appearance(values: list[float]) -> BodyAppearance:
    histogram = np.asarray(values, dtype=np.float32)
    histogram /= np.linalg.norm(histogram)
    return BodyAppearance(histogram=histogram, aspect_ratio=0.5, height_px=200)


def evidence(
    *,
    device_id: str,
    track_id: int,
    host_seconds: float,
    body: BodyAppearance | None,
    depth_mm: float | None = None,
    faces: tuple[str, ...] = (),
    camera_role: str = CAMERA_ROLE_OBSERVER,
    track_status: str = "NEW",
) -> TrackVisitEvidence:
    return TrackVisitEvidence(
        camera_role=camera_role,
        device_id=device_id,
        track_id=track_id,
        host_seconds=host_seconds,
        track_bbox=(0, 0, 100, 200),
        track_status=track_status,
        face_identity_ids=faces,
        body_appearance=body,
        depth_mm=depth_mm,
    )


def registry() -> VisitRegistry:
    return VisitRegistry(
        observer_match_threshold=0.45,
        observer_handoff_max_delay_seconds=0.0,
        observer_single_active_fallback_threshold=0.25,
        observer_provisional_seconds=3.0,
    )


class VisitRegistryProvisionalTests(unittest.TestCase):
    def test_single_eligible_entrance_visit_uses_lower_fallback_threshold(self) -> None:
        visits = registry()
        entrance = visits.resolve_entrance_track(
            evidence(
                device_id="entrance",
                track_id=1,
                host_seconds=0.0,
                body=appearance([1.0, 0.0]),
                depth_mm=1000.0,
                camera_role=CAMERA_ROLE_ENTRANCE_OBSERVER,
            )
        )
        observed = appearance([0.3, math.sqrt(0.91)])

        decision = visits.resolve_observer_track(
            evidence(
                device_id="remote-room",
                track_id=2,
                host_seconds=10.0,
                body=observed,
                depth_mm=2000.0,
            )
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.assignment.visit_id, entrance.assignment.visit_id)
        self.assertEqual(decision.decision, "observer_single_active_fallback")
        self.assertGreaterEqual(decision.score or 0.0, 0.25)
        self.assertLess(decision.score or 1.0, 0.45)

    def test_multiple_entrance_visits_do_not_use_low_fallback(self) -> None:
        visits = registry()
        visits.resolve_entrance_track(
            evidence(
                device_id="entrance-a",
                track_id=1,
                host_seconds=0.0,
                body=appearance([1.0, 0.0, 0.0]),
                depth_mm=1000.0,
                camera_role=CAMERA_ROLE_ENTRANCE_OBSERVER,
            )
        )
        visits.resolve_entrance_track(
            evidence(
                device_id="entrance-b",
                track_id=2,
                host_seconds=10.0,
                body=appearance([0.0, 1.0, 0.0]),
                depth_mm=1000.0,
                camera_role=CAMERA_ROLE_ENTRANCE_OBSERVER,
            )
        )
        observed = appearance([0.3, 0.3, math.sqrt(0.82)])

        decision = visits.resolve_observer_track(
            evidence(
                device_id="remote-room",
                track_id=3,
                host_seconds=20.0,
                body=observed,
                depth_mm=2000.0,
            )
        )

        self.assertIsNone(decision)
        self.assertTrue(
            visits.is_observer_track_provisional(device_id="remote-room", track_id=3)
        )
        self.assertEqual(len(visits.visits), 2)

    def test_provisional_track_retries_and_matches_when_evidence_improves(self) -> None:
        visits = registry()
        first = visits.resolve_entrance_track(
            evidence(
                device_id="entrance-a",
                track_id=1,
                host_seconds=0.0,
                body=appearance([1.0, 0.0, 0.0]),
                depth_mm=1000.0,
                camera_role=CAMERA_ROLE_ENTRANCE_OBSERVER,
            )
        )
        visits.resolve_entrance_track(
            evidence(
                device_id="entrance-b",
                track_id=2,
                host_seconds=10.0,
                body=appearance([0.0, 1.0, 0.0]),
                depth_mm=1000.0,
                camera_role=CAMERA_ROLE_ENTRANCE_OBSERVER,
            )
        )
        low = appearance([0.3, 0.3, math.sqrt(0.82)])
        self.assertIsNone(
            visits.resolve_observer_track(
                evidence(
                    device_id="remote-room",
                    track_id=3,
                    host_seconds=20.0,
                    body=low,
                    depth_mm=2000.0,
                )
            )
        )

        decision = visits.resolve_observer_track(
            evidence(
                device_id="remote-room",
                track_id=3,
                host_seconds=21.0,
                body=appearance([1.0, 0.0, 0.0]),
                depth_mm=1000.0,
                track_status="TRACKED",
            )
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.assignment.visit_id, first.assignment.visit_id)
        self.assertFalse(
            visits.is_observer_track_provisional(device_id="remote-room", track_id=3)
        )

    def test_temporal_handoff_can_resolve_an_existing_provisional_track(self) -> None:
        visits = VisitRegistry(
            observer_match_threshold=0.45,
            observer_handoff_max_delay_seconds=8.0,
            observer_handoff_threshold=0.35,
            observer_single_active_fallback_threshold=0.25,
            observer_provisional_seconds=3.0,
        )
        observer_evidence = evidence(
            device_id="remote-room",
            track_id=4,
            host_seconds=0.0,
            body=None,
        )
        self.assertIsNone(visits.resolve_observer_track(observer_evidence))
        entrance = visits.resolve_entrance_track(
            evidence(
                device_id="entrance",
                track_id=1,
                host_seconds=1.0,
                body=None,
                camera_role=CAMERA_ROLE_ENTRANCE_OBSERVER,
            )
        )

        decision = visits.resolve_observer_track(
            evidence(
                device_id="remote-room",
                track_id=4,
                host_seconds=1.5,
                body=None,
                track_status="TRACKED",
            )
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.assignment.visit_id, entrance.assignment.visit_id)
        self.assertEqual(decision.decision, "observer_handoff_reused")

    def test_entrance_event_clears_provisional_state_for_same_track(self) -> None:
        visits = registry()
        observation = evidence(
            device_id="entrance-observer",
            track_id=8,
            host_seconds=0.0,
            body=None,
            camera_role=CAMERA_ROLE_ENTRANCE_OBSERVER,
        )
        self.assertIsNone(visits.resolve_observer_track(observation))

        decision = visits.resolve_entrance_track(observation)

        self.assertEqual(decision.assignment.origin, "entrance_confirmed")
        self.assertFalse(
            visits.is_observer_track_provisional(
                device_id="entrance-observer",
                track_id=8,
            )
        )

    def test_provisional_timeout_creates_observer_only_visit(self) -> None:
        visits = registry()
        self.assertIsNone(
            visits.resolve_observer_track(
                evidence(
                    device_id="observer",
                    track_id=5,
                    host_seconds=0.0,
                    body=None,
                )
            )
        )

        decision = visits.resolve_observer_track(
            evidence(
                device_id="observer",
                track_id=5,
                host_seconds=3.1,
                body=None,
                track_status="TRACKED",
            )
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.decision, "new_observer_only_visit")
        self.assertEqual(decision.reason, "provisional_timeout")
        self.assertEqual(decision.assignment.origin, VISIT_ORIGIN_OBSERVER)

    def test_lost_provisional_track_does_not_create_visit(self) -> None:
        visits = registry()
        self.assertIsNone(
            visits.resolve_observer_track(
                evidence(
                    device_id="observer",
                    track_id=6,
                    host_seconds=0.0,
                    body=None,
                )
            )
        )

        decision = visits.resolve_observer_track(
            evidence(
                device_id="observer",
                track_id=6,
                host_seconds=3.1,
                body=None,
                track_status="LOST",
            )
        )

        self.assertIsNone(decision)
        self.assertEqual(visits.visits, {})
        self.assertFalse(
            visits.is_observer_track_provisional(device_id="observer", track_id=6)
        )

    def test_conflicting_mapped_faces_remain_provisional(self) -> None:
        visits = registry()
        visits.resolve_entrance_track(
            evidence(
                device_id="entrance-a",
                track_id=1,
                host_seconds=0.0,
                body=appearance([1.0, 0.0]),
                faces=("face-a",),
                camera_role=CAMERA_ROLE_ENTRANCE_OBSERVER,
            )
        )
        visits.resolve_entrance_track(
            evidence(
                device_id="entrance-b",
                track_id=2,
                host_seconds=10.0,
                body=appearance([0.0, 1.0]),
                faces=("face-b",),
                camera_role=CAMERA_ROLE_ENTRANCE_OBSERVER,
            )
        )

        decision = visits.resolve_observer_track(
            evidence(
                device_id="observer",
                track_id=7,
                host_seconds=20.0,
                body=appearance([1.0, 0.0]),
                faces=("face-a", "face-b"),
            )
        )

        self.assertIsNone(decision)
        self.assertTrue(
            visits.is_observer_track_provisional(device_id="observer", track_id=7)
        )


if __name__ == "__main__":
    unittest.main()
