import unittest

import numpy as np

from pipeline.face_capture import (
    FaceCaptureCoordinator,
    FaceCaptureError,
    FaceCaptureQualityConfig,
    FaceQualityResult,
    _feedback_for_quality,
    evaluate_face_quality,
)
from pipeline.face_identity import RecognizedFace
from pipeline.tracking import Track
from pipeline.visit_identity import VisitAssignment
from pipeline.visit_registry import (
    VISIT_ORIGIN_ENTRANCE,
    VISIT_ORIGIN_OBSERVER,
    VISIT_STATUS_ACTIVE,
    VISIT_STATUS_CLOSED,
)


def acceptable_frame_and_face() -> tuple[np.ndarray, RecognizedFace]:
    generator = np.random.default_rng(7)
    frame = generator.integers(65, 190, (700, 800, 3), dtype=np.uint8)
    return frame, RecognizedFace(
        bbox=(280, 180, 500, 430),
        det_score=0.96,
        identity_id="face_person_001",
        best_score=0.9,
        track_id=4,
        landmarks=(
            (335.0, 255.0),
            (445.0, 255.0),
            (390.0, 315.0),
            (350.0, 370.0),
            (430.0, 370.0),
        ),
    )


class FaceCaptureQualityTests(unittest.TestCase):
    def test_accepts_large_frontal_sharp_face(self) -> None:
        frame, face = acceptable_frame_and_face()
        result = evaluate_face_quality(
            frame,
            face,
            config=FaceCaptureQualityConfig(),
        )
        self.assertTrue(result.acceptable, result.failures)
        self.assertGreater(result.crop.shape[0], result.face_height_pixels)
        self.assertGreater(result.crop.shape[1], result.face_width_pixels)
        self.assertAlmostEqual(result.roll_degrees or 0.0, 0.0)

    def test_reports_small_face(self) -> None:
        frame, face = acceptable_frame_and_face()
        face.bbox = (350, 250, 430, 340)
        result = evaluate_face_quality(
            frame,
            face,
            config=FaceCaptureQualityConfig(),
        )
        self.assertIn("face_too_small", result.failures)

    def test_pose_feedback_gives_a_directional_correction(self) -> None:
        quality = FaceQualityResult(
            crop=np.zeros((200, 160, 3), dtype=np.uint8),
            crop_box=(0, 0, 160, 200),
            detection_score=0.99,
            face_width_pixels=160,
            face_height_pixels=200,
            yaw_degrees=25.0,
            pitch_degrees=0.0,
            roll_degrees=0.0,
            sharpness=100.0,
            brightness=120.0,
            dark_fraction=0.0,
            bright_fraction=0.0,
            failures=("yaw_out_of_range",),
            quality_score=0.8,
        )

        code, message = _feedback_for_quality(quality, 2)

        self.assertEqual(code, "TURN_FACE")
        self.assertIn("your right", message)
        self.assertIn("camera 2", message)


class FaceCaptureCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.coordinator = FaceCaptureCoordinator(
            camera_device_ids=["camera-a"],
            event_sink=lambda event_type, payload: self.events.append(
                (event_type, dict(payload))
            ),
        )
        self.coordinator.update_visit_state(
            {7: (VISIT_ORIGIN_ENTRANCE, "customer-7", VISIT_STATUS_ACTIVE)}
        )

    def test_customer_only_capture_resolves_and_completes_after_three_frames(self) -> None:
        created = self.coordinator.create(
            customer_id="customer-7",
            visit_id=None,
            purpose="identity_verification",
            timeout_seconds=30.0,
            idempotency_key="request-1",
        )
        self.assertEqual(
            created["statusUrl"],
            f"/face-captures/{created['captureId']}",
        )
        self.assertEqual(created["visitId"], 7)
        frame, face = acceptable_frame_and_face()
        track = Track(4, 180, 80, 620, 650, 0.9, "TRACKED")
        assignment = VisitAssignment(
            visit_id=7,
            track_id=4,
            device_id="camera-a",
            face_identity_ids=("face_person_001",),
            matched_score=0.9,
            origin=VISIT_ORIGIN_ENTRANCE,
        )
        for sequence in (10, 11, 12):
            self.coordinator.observe_frame(
                camera_index=0,
                device_id="camera-a",
                rgb_sequence_number=sequence,
                observed_at_unix_milliseconds=1000 + sequence,
                frame=frame,
                tracks=[track],
                recognized_faces=[face],
                assignments={4: assignment},
                customer_ids_by_visit={7: "customer-7"},
            )

        ready = self.coordinator.get(created["captureId"])
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["cameraNumber"], 1)
        self.assertEqual(ready["rgbSequenceNumber"], 12)
        image, digest = self.coordinator.image(created["captureId"])
        self.assertTrue(image.startswith(b"\xff\xd8"))
        self.assertEqual(len(digest), 64)
        self.assertIn("face_capture_ready", [event[0] for event in self.events])

    def test_idempotency_and_conflict(self) -> None:
        first = self.coordinator.create(
            customer_id="customer-7",
            visit_id=7,
            purpose="identity_verification",
            timeout_seconds=30.0,
            idempotency_key="same-key",
        )
        repeated = self.coordinator.create(
            customer_id="customer-7",
            visit_id=7,
            purpose="identity_verification",
            timeout_seconds=30.0,
            idempotency_key="same-key",
        )
        self.assertEqual(first["captureId"], repeated["captureId"])
        with self.assertRaisesRegex(FaceCaptureError, "different request"):
            self.coordinator.create(
                customer_id="customer-7",
                visit_id=8,
                purpose="identity_verification",
                timeout_seconds=30.0,
                idempotency_key="same-key",
            )

    def test_rejects_known_customer_visit_mismatch(self) -> None:
        with self.assertRaisesRegex(FaceCaptureError, "different customer"):
            self.coordinator.create(
                customer_id="another-customer",
                visit_id=7,
                purpose="identity_verification",
                timeout_seconds=30.0,
                idempotency_key=None,
            )

    def test_active_capture_fails_when_target_visit_closes(self) -> None:
        created = self.coordinator.create(
            customer_id="customer-7",
            visit_id=7,
            purpose="identity_verification",
            timeout_seconds=30.0,
            idempotency_key=None,
        )
        self.coordinator.update_visit_state(
            {7: (VISIT_ORIGIN_ENTRANCE, "customer-7", VISIT_STATUS_CLOSED)}
        )
        status = self.coordinator.get(created["captureId"])
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["feedback"]["code"], "TARGET_VISIT_INACTIVE")

    def test_operator_capture_allows_active_observer_only_visit(self) -> None:
        self.coordinator.update_visit_state(
            {25: (VISIT_ORIGIN_OBSERVER, None, VISIT_STATUS_ACTIVE)}
        )

        created = self.coordinator.create(
            customer_id=None,
            visit_id=25,
            purpose="operator_test",
            timeout_seconds=30.0,
            idempotency_key=None,
        )

        self.assertEqual(created["status"], "locating_customer")
        self.assertEqual(created["visitId"], 25)
        self.coordinator.update_visit_state(
            {25: (VISIT_ORIGIN_OBSERVER, None, VISIT_STATUS_ACTIVE)}
        )
        self.assertNotEqual(self.coordinator.get(created["captureId"])["status"], "failed")

    def test_identity_capture_rejects_observer_only_visit(self) -> None:
        self.coordinator.update_visit_state(
            {25: (VISIT_ORIGIN_OBSERVER, None, VISIT_STATUS_ACTIVE)}
        )

        with self.assertRaisesRegex(FaceCaptureError, "entrance-confirmed"):
            self.coordinator.create(
                customer_id=None,
                visit_id=25,
                purpose="identity_verification",
                timeout_seconds=30.0,
                idempotency_key=None,
            )

    def test_delete_removes_image_and_is_idempotent(self) -> None:
        created = self.coordinator.create(
            customer_id=None,
            visit_id=7,
            purpose="operator_test",
            timeout_seconds=30.0,
            idempotency_key=None,
        )
        self.assertEqual(
            self.coordinator.delete(created["captureId"])["status"],
            "deleted",
        )
        self.assertEqual(
            self.coordinator.delete(created["captureId"])["status"],
            "deleted",
        )
        with self.assertRaises(KeyError):
            self.coordinator.get(created["captureId"])


if __name__ == "__main__":
    unittest.main()
