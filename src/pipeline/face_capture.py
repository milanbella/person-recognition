from __future__ import annotations

import hashlib
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from pipeline.face_identity import RecognizedFace
from pipeline.tracking import Track
from pipeline.visit_identity import VisitAssignment
from pipeline.visit_registry import VISIT_ORIGIN_ENTRANCE, VISIT_STATUS_ACTIVE


TERMINAL_FACE_CAPTURE_STATUSES = {"ready", "expired", "cancelled", "failed"}
ACTIVE_FACE_CAPTURE_STATUSES = {
    "locating_customer",
    "adjust_position",
    "capturing",
}


class FaceCaptureError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FaceCaptureQualityConfig:
    minimum_detection_score: float = 0.80
    minimum_face_width_pixels: int = 160
    minimum_face_height_pixels: int = 180
    maximum_absolute_yaw_degrees: float = 20.0
    maximum_absolute_pitch_degrees: float = 20.0
    maximum_absolute_roll_degrees: float = 15.0
    minimum_sharpness: float = 50.0
    minimum_brightness: float = 45.0
    maximum_brightness: float = 215.0
    maximum_dark_fraction: float = 0.35
    maximum_bright_fraction: float = 0.25
    required_acceptable_observations: int = 3
    acceptance_window_seconds: float = 1.5
    recent_signal_seconds: float = 1.25
    jpeg_quality: int = 95


@dataclass(frozen=True)
class FaceQualityResult:
    crop: np.ndarray
    crop_box: tuple[int, int, int, int]
    detection_score: float
    face_width_pixels: int
    face_height_pixels: int
    yaw_degrees: float | None
    pitch_degrees: float | None
    roll_degrees: float | None
    sharpness: float
    brightness: float
    dark_fraction: float
    bright_fraction: float
    failures: tuple[str, ...]
    quality_score: float

    @property
    def acceptable(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class FaceCaptureSignal:
    observed_monotonic: float
    camera_index: int
    device_id: str
    track_id: int
    visit_id: int
    rgb_sequence_number: int
    feedback_code: str
    feedback_message: str
    signal_score: float
    quality: FaceQualityResult | None = None


@dataclass
class FaceCaptureSession:
    capture_id: str
    customer_id: str | None
    requested_visit_id: int | None
    visit_id: int | None
    purpose: str
    idempotency_key: str | None
    request_fingerprint: tuple[object, ...]
    created_unix_milliseconds: int
    created_monotonic: float
    capture_deadline_monotonic: float
    status: str = "locating_customer"
    feedback_code: str = "CUSTOMER_NOT_VISIBLE"
    feedback_message: str = "Stand where a camera can see you."
    signals_by_camera: dict[int, FaceCaptureSignal] = field(default_factory=dict)
    accepted_observations: deque[tuple[float, tuple[int, int]]] = field(
        default_factory=lambda: deque(maxlen=5)
    )
    best_quality: FaceQualityResult | None = None
    best_signal: FaceCaptureSignal | None = None
    image_jpeg: bytes | None = None
    image_sha256: str | None = None
    image_expires_monotonic: float | None = None
    captured_at_unix_milliseconds: int | None = None
    terminal_at_unix_milliseconds: int | None = None


def _expanded_head_crop(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int], bool]:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    face_width = max(1, x2 - x1)
    face_height = max(1, y2 - y1)
    requested = (
        int(round(x1 - face_width * 0.35)),
        int(round(y1 - face_height * 0.45)),
        int(round(x2 + face_width * 0.35)),
        int(round(y2 + face_height * 0.70)),
    )
    clipped = (
        requested[0] < 0
        or requested[1] < 0
        or requested[2] > frame_width
        or requested[3] > frame_height
    )
    crop_box = (
        max(0, requested[0]),
        max(0, requested[1]),
        min(frame_width, requested[2]),
        min(frame_height, requested[3]),
    )
    crop = frame[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
    return crop, crop_box, clipped


def _pose_from_landmarks(
    landmarks: Sequence[tuple[float, float]],
    bbox: tuple[int, int, int, int],
) -> tuple[float | None, float | None, float | None]:
    if len(landmarks) != 5:
        return None, None, None
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    eye_distance = math.hypot(eye_dx, eye_dy)
    face_height = max(1.0, float(bbox[3] - bbox[1]))
    if eye_distance < 1.0:
        return None, None, None

    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
    mouth_mid_y = (left_mouth[1] + right_mouth[1]) / 2.0
    roll = math.degrees(math.atan2(eye_dy, eye_dx))
    yaw = max(-45.0, min(45.0, (nose[0] - eye_mid_x) / eye_distance * 60.0))
    upper = (nose[1] - eye_mid_y) / face_height
    lower = (mouth_mid_y - nose[1]) / face_height
    pitch = max(-45.0, min(45.0, (upper - lower) * 90.0))
    return yaw, pitch, roll


def evaluate_face_quality(
    frame: np.ndarray,
    face: RecognizedFace,
    *,
    config: FaceCaptureQualityConfig,
) -> FaceQualityResult:
    x1, y1, x2, y2 = face.bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    crop, crop_box, clipped = _expanded_head_crop(frame, face.bbox)
    if crop.size == 0:
        raise FaceCaptureError("INVALID_FACE_CROP", "The detected face crop is empty.")

    face_roi = frame[max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)]
    if face_roi.size == 0:
        face_roi = crop
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    brightness = float(np.mean(normalized))
    dark_fraction = float(np.mean(normalized <= 20))
    bright_fraction = float(np.mean(normalized >= 245))
    yaw, pitch, roll = _pose_from_landmarks(face.landmarks, face.bbox)

    failures: list[str] = []
    if face.det_score < config.minimum_detection_score:
        failures.append("detection_score_low")
    if width < config.minimum_face_width_pixels or height < config.minimum_face_height_pixels:
        failures.append("face_too_small")
    if len(face.landmarks) != 5:
        failures.append("landmarks_missing")
    if yaw is not None and abs(yaw) > config.maximum_absolute_yaw_degrees:
        failures.append("yaw_out_of_range")
    if pitch is not None and abs(pitch) > config.maximum_absolute_pitch_degrees:
        failures.append("pitch_out_of_range")
    if roll is not None and abs(roll) > config.maximum_absolute_roll_degrees:
        failures.append("roll_out_of_range")
    if sharpness < config.minimum_sharpness:
        failures.append("face_blurry")
    if brightness < config.minimum_brightness or dark_fraction > config.maximum_dark_fraction:
        failures.append("too_dark")
    if brightness > config.maximum_brightness or bright_fraction > config.maximum_bright_fraction:
        failures.append("too_bright")
    if clipped:
        failures.append("face_clipped")

    score_parts = [
        min(1.0, face.det_score),
        min(1.0, width / max(config.minimum_face_width_pixels, 1)),
        min(1.0, height / max(config.minimum_face_height_pixels, 1)),
        min(1.0, sharpness / max(config.minimum_sharpness * 2.0, 1.0)),
        max(0.0, 1.0 - abs(brightness - 130.0) / 130.0),
    ]
    if yaw is not None:
        score_parts.append(max(0.0, 1.0 - abs(yaw) / 45.0))
    if pitch is not None:
        score_parts.append(max(0.0, 1.0 - abs(pitch) / 45.0))
    if roll is not None:
        score_parts.append(max(0.0, 1.0 - abs(roll) / 45.0))
    quality_score = float(sum(score_parts) / len(score_parts))
    return FaceQualityResult(
        crop=crop.copy(),
        crop_box=crop_box,
        detection_score=float(face.det_score),
        face_width_pixels=width,
        face_height_pixels=height,
        yaw_degrees=yaw,
        pitch_degrees=pitch,
        roll_degrees=roll,
        sharpness=sharpness,
        brightness=brightness,
        dark_fraction=dark_fraction,
        bright_fraction=bright_fraction,
        failures=tuple(failures),
        quality_score=quality_score,
    )


def _feedback_for_quality(
    quality: FaceQualityResult,
    camera_number: int,
) -> tuple[str, str]:
    failures = set(quality.failures)
    if "face_too_small" in failures:
        return "MOVE_CLOSER", f"Step closer to camera {camera_number} and place your face in the oval."
    if "yaw_out_of_range" in failures and quality.yaw_degrees is not None:
        direction = "right" if quality.yaw_degrees > 0 else "left"
        return "TURN_FACE", f"Turn your face slightly to your {direction}, toward camera {camera_number}."
    if "pitch_out_of_range" in failures and quality.pitch_degrees is not None:
        if quality.pitch_degrees > 0:
            return "RAISE_CHIN", f"Raise your chin slightly and look at camera {camera_number}."
        return "LOWER_CHIN", f"Lower your chin slightly and look at camera {camera_number}."
    if "roll_out_of_range" in failures:
        return "KEEP_HEAD_UPRIGHT", f"Keep your head upright and look at camera {camera_number}."
    if "face_clipped" in failures:
        return "MOVE_BACK", f"Move slightly away from camera {camera_number}."
    if "face_blurry" in failures:
        return "HOLD_STILL", "Hold still."
    if "too_dark" in failures:
        return "TOO_DARK", "Move to a better-lit position."
    if "too_bright" in failures:
        return "TOO_BRIGHT", "Avoid strong light behind your face."
    if failures:
        return "FACE_NOT_VISIBLE", f"Make sure your face is visible to camera {camera_number}."
    return "CAPTURING", "Good position. Hold still."


class FaceCaptureCoordinator:
    def __init__(
        self,
        *,
        camera_device_ids: Sequence[str],
        default_timeout_seconds: float = 30.0,
        ready_ttl_seconds: float = 120.0,
        maximum_active_sessions: int = 8,
        quality_config: FaceCaptureQualityConfig | None = None,
        event_sink: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.camera_device_ids = tuple(camera_device_ids)
        self.default_timeout_seconds = default_timeout_seconds
        self.ready_ttl_seconds = ready_ttl_seconds
        self.maximum_active_sessions = maximum_active_sessions
        self.quality_config = quality_config or FaceCaptureQualityConfig()
        self.event_sink = event_sink
        self._lock = threading.RLock()
        self._sessions: dict[str, FaceCaptureSession] = {}
        self._idempotency: dict[str, str] = {}
        self._visit_state: dict[int, tuple[str, str | None, str]] = {}

    def has_active_sessions(self) -> bool:
        with self._lock:
            self._cleanup_locked(time.monotonic())
            return any(session.status in ACTIVE_FACE_CAPTURE_STATUSES for session in self._sessions.values())

    def update_visit_state(
        self,
        visits: Mapping[int, tuple[str, str | None, str]],
    ) -> None:
        failed_payloads: list[dict[str, Any]] = []
        with self._lock:
            self._visit_state = dict(visits)
            for session in self._sessions.values():
                if session.status not in ACTIVE_FACE_CAPTURE_STATUSES:
                    continue
                target_visit_id = session.requested_visit_id or session.visit_id
                target_state = (
                    None if target_visit_id is None else self._visit_state.get(target_visit_id)
                )
                if target_state is not None and target_state[2] != VISIT_STATUS_ACTIVE:
                    session.status = "failed"
                    session.feedback_code = "TARGET_VISIT_INACTIVE"
                    session.feedback_message = "The target visit is no longer active."
                    session.terminal_at_unix_milliseconds = time.time_ns() // 1_000_000
                    failed_payloads.append(
                        self._payload_locked(session, time.monotonic())
                    )
                    continue
                if (
                    target_state is not None
                    and session.purpose == "identity_verification"
                    and target_state[0] != VISIT_ORIGIN_ENTRANCE
                ):
                    session.status = "failed"
                    session.feedback_code = "TARGET_NOT_ENTRANCE_CONFIRMED"
                    session.feedback_message = (
                        "Identity verification requires an entrance-confirmed visit."
                    )
                    session.terminal_at_unix_milliseconds = time.time_ns() // 1_000_000
                    failed_payloads.append(
                        self._payload_locked(session, time.monotonic())
                    )
                    continue
                try:
                    self._resolve_session_target_locked(session)
                except FaceCaptureError as error:
                    session.status = "failed"
                    session.feedback_code = error.code
                    session.feedback_message = str(error)
                    session.terminal_at_unix_milliseconds = time.time_ns() // 1_000_000
                    failed_payloads.append(
                        self._payload_locked(session, time.monotonic())
                    )
        for payload in failed_payloads:
            self._emit("face_capture_failed", payload)
            print(
                f"FACE_CAPTURE_FAILED capture_id={payload['captureId']} "
                f"code={payload['feedback']['code']}"
            )

    def create(
        self,
        *,
        customer_id: str | None,
        visit_id: int | None,
        purpose: str,
        timeout_seconds: float | None,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        customer_id = None if customer_id is None else customer_id.strip()
        if not customer_id and visit_id is None:
            raise FaceCaptureError("TARGET_REQUIRED", "customerId or visitId is required.")
        if visit_id is not None and visit_id <= 0:
            raise FaceCaptureError("INVALID_VISIT_ID", "visitId must be greater than zero.")
        if purpose not in {"identity_verification", "operator_test"}:
            raise FaceCaptureError("INVALID_PURPOSE", "Unsupported face-capture purpose.")
        effective_timeout = self.default_timeout_seconds if timeout_seconds is None else timeout_seconds
        if effective_timeout < 5.0 or effective_timeout > 60.0:
            raise FaceCaptureError("INVALID_TIMEOUT", "timeoutSeconds must be between 5 and 60.")

        fingerprint = (customer_id, visit_id, purpose, float(effective_timeout))
        now_monotonic = time.monotonic()
        now_ms = time.time_ns() // 1_000_000
        with self._lock:
            self._cleanup_locked(now_monotonic)
            if idempotency_key:
                existing_id = self._idempotency.get(idempotency_key)
                if existing_id is not None and existing_id in self._sessions:
                    existing = self._sessions[existing_id]
                    if existing.request_fingerprint != fingerprint:
                        raise FaceCaptureError(
                            "IDEMPOTENCY_CONFLICT",
                            "Idempotency-Key was already used for a different request.",
                        )
                    return self._payload_locked(existing, now_monotonic)
            active_count = sum(
                session.status in ACTIVE_FACE_CAPTURE_STATUSES
                for session in self._sessions.values()
            )
            if active_count >= self.maximum_active_sessions:
                raise FaceCaptureError("TOO_MANY_CAPTURES", "Too many face captures are active.")
            for existing in self._sessions.values():
                if existing.status not in ACTIVE_FACE_CAPTURE_STATUSES:
                    continue
                same_target = (
                    customer_id is not None
                    and existing.customer_id == customer_id
                ) or (
                    visit_id is not None
                    and (existing.visit_id == visit_id or existing.requested_visit_id == visit_id)
                )
                if same_target:
                    raise FaceCaptureError(
                        "CAPTURE_ALREADY_ACTIVE",
                        "A face capture is already active for this customer or visit.",
                    )

            session = FaceCaptureSession(
                capture_id=f"fc_{uuid.uuid4().hex[:16]}",
                customer_id=customer_id,
                requested_visit_id=visit_id,
                visit_id=visit_id,
                purpose=purpose,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                created_unix_milliseconds=now_ms,
                created_monotonic=now_monotonic,
                capture_deadline_monotonic=now_monotonic + effective_timeout,
            )
            self._resolve_session_target_locked(session)
            self._sessions[session.capture_id] = session
            if idempotency_key:
                self._idempotency[idempotency_key] = session.capture_id
            payload = self._payload_locked(session, now_monotonic)
        self._emit("face_capture_created", payload)
        print(
            f"FACE_CAPTURE_CREATED capture_id={session.capture_id} "
            f"visit_id={session.visit_id} timeout_seconds={effective_timeout:.1f}"
        )
        return payload

    def observe_frame(
        self,
        *,
        camera_index: int,
        device_id: str,
        rgb_sequence_number: int,
        observed_at_unix_milliseconds: int,
        frame: np.ndarray,
        tracks: Sequence[Track],
        recognized_faces: Sequence[RecognizedFace],
        assignments: Mapping[int, VisitAssignment],
        customer_ids_by_visit: Mapping[int, str],
    ) -> None:
        now_monotonic = time.monotonic()
        with self._lock:
            self._cleanup_locked(now_monotonic)
            sessions = [
                session
                for session in self._sessions.values()
                if session.status in ACTIVE_FACE_CAPTURE_STATUSES and session.visit_id is not None
            ]
        if not sessions:
            return

        tracks_by_id = {track.track_id: track for track in tracks if track.status in {"NEW", "TRACKED"}}
        faces_by_track = {
            face.track_id: face
            for face in recognized_faces
            if face.track_id is not None
        }
        assignment_by_visit: dict[int, list[tuple[Track, VisitAssignment]]] = {}
        for track_id, assignment in assignments.items():
            track = tracks_by_id.get(track_id)
            if track is None:
                continue
            assignment_by_visit.setdefault(assignment.visit_id, []).append((track, assignment))

        for session in sessions:
            assert session.visit_id is not None
            matches = assignment_by_visit.get(session.visit_id, [])
            if not matches:
                continue
            if len(matches) > 1:
                self._record_simple_signal(
                    session.capture_id,
                    camera_index=camera_index,
                    device_id=device_id,
                    track_id=matches[0][0].track_id,
                    visit_id=session.visit_id,
                    rgb_sequence_number=rgb_sequence_number,
                    code="TARGET_AMBIGUOUS",
                    message="The target appears more than once in this camera.",
                    score=0.05,
                )
                continue
            track, assignment = matches[0]
            bound_customer = customer_ids_by_visit.get(assignment.visit_id)
            if session.customer_id is not None and bound_customer not in {None, session.customer_id}:
                self._fail_session(session.capture_id, "TARGET_MISMATCH", "Visit/customer binding changed.")
                continue
            face = faces_by_track.get(track.track_id)
            if face is None:
                self._record_simple_signal(
                    session.capture_id,
                    camera_index=camera_index,
                    device_id=device_id,
                    track_id=track.track_id,
                    visit_id=assignment.visit_id,
                    rgb_sequence_number=rgb_sequence_number,
                    code="FACE_NOT_VISIBLE",
                    message=f"Make sure your face is visible to camera {camera_index + 1}.",
                    score=0.10,
                )
                continue
            try:
                quality = evaluate_face_quality(frame, face, config=self.quality_config)
            except FaceCaptureError:
                continue
            code, message = _feedback_for_quality(quality, camera_index + 1)
            signal = FaceCaptureSignal(
                observed_monotonic=now_monotonic,
                camera_index=camera_index,
                device_id=device_id,
                track_id=track.track_id,
                visit_id=assignment.visit_id,
                rgb_sequence_number=rgb_sequence_number,
                feedback_code=code,
                feedback_message=message,
                signal_score=quality.quality_score,
                quality=quality,
            )
            self._record_quality_signal(
                session.capture_id,
                signal,
                observed_at_unix_milliseconds=observed_at_unix_milliseconds,
            )

    def get(self, capture_id: str) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            self._cleanup_locked(now)
            session = self._sessions.get(capture_id)
            if session is None:
                raise KeyError(capture_id)
            self._refresh_feedback_locked(session, now)
            return self._payload_locked(session, now)

    def image(self, capture_id: str) -> tuple[bytes, str]:
        now = time.monotonic()
        with self._lock:
            self._cleanup_locked(now)
            session = self._sessions.get(capture_id)
            if session is None:
                raise KeyError(capture_id)
            if session.status == "expired":
                raise FaceCaptureError("CAPTURE_EXPIRED", "The face capture expired.")
            if (
                session.status != "ready"
                or session.image_jpeg is None
                or session.image_sha256 is None
            ):
                raise FaceCaptureError("IMAGE_NOT_READY", "The face image is not ready.")
            return session.image_jpeg, session.image_sha256

    def delete(self, capture_id: str) -> dict[str, Any]:
        now_ms = time.time_ns() // 1_000_000
        with self._lock:
            session = self._sessions.get(capture_id)
            if session is None:
                return {"captureId": capture_id, "status": "deleted"}
            session.status = "cancelled" if session.status in ACTIVE_FACE_CAPTURE_STATUSES else "cancelled"
            session.feedback_code = "CANCELLED"
            session.feedback_message = "Face capture was cancelled."
            session.image_jpeg = None
            session.image_sha256 = None
            session.terminal_at_unix_milliseconds = now_ms
            payload = {"captureId": capture_id, "status": "deleted"}
            self._remove_session_locked(session)
        self._emit("face_capture_deleted", payload)
        print(f"FACE_CAPTURE_DELETED capture_id={capture_id} reason=caller_request")
        return payload

    def _resolve_session_target_locked(self, session: FaceCaptureSession) -> None:
        active_entrance = {
            visit_id: customer_id
            for visit_id, (origin, customer_id, status) in self._visit_state.items()
            if origin == VISIT_ORIGIN_ENTRANCE and status == VISIT_STATUS_ACTIVE
        }
        if session.requested_visit_id is not None:
            target_state = self._visit_state.get(session.requested_visit_id)
            if target_state is not None and target_state[2] != VISIT_STATUS_ACTIVE:
                raise FaceCaptureError(
                    "TARGET_VISIT_INACTIVE",
                    "The target visit is no longer active.",
                )
            if (
                target_state is not None
                and session.purpose == "identity_verification"
                and target_state[0] != VISIT_ORIGIN_ENTRANCE
            ):
                raise FaceCaptureError(
                    "TARGET_NOT_ENTRANCE_CONFIRMED",
                    "Identity verification requires an entrance-confirmed visit.",
                )
            known_customer = None if target_state is None else target_state[1]
            if known_customer is not None and session.customer_id is not None and known_customer != session.customer_id:
                raise FaceCaptureError(
                    "TARGET_MISMATCH",
                    "visitId is bound to a different customerId.",
                )
            session.visit_id = session.requested_visit_id
            return
        if session.customer_id is None:
            return
        matching = [
            visit_id
            for visit_id, customer_id in active_entrance.items()
            if customer_id == session.customer_id
        ]
        if len(matching) > 1:
            raise FaceCaptureError(
                "TARGET_AMBIGUOUS",
                "customerId is bound to multiple active entrance visits.",
            )
        if len(matching) == 1:
            session.visit_id = matching[0]
        else:
            session.visit_id = None
            session.feedback_code = "CUSTOMER_NOT_BOUND"
            session.feedback_message = "The customer is not bound to an active visit yet."

    def _record_simple_signal(
        self,
        capture_id: str,
        *,
        camera_index: int,
        device_id: str,
        track_id: int,
        visit_id: int,
        rgb_sequence_number: int,
        code: str,
        message: str,
        score: float,
    ) -> None:
        signal = FaceCaptureSignal(
            observed_monotonic=time.monotonic(),
            camera_index=camera_index,
            device_id=device_id,
            track_id=track_id,
            visit_id=visit_id,
            rgb_sequence_number=rgb_sequence_number,
            feedback_code=code,
            feedback_message=message,
            signal_score=score,
        )
        self._record_quality_signal(
            capture_id,
            signal,
            observed_at_unix_milliseconds=time.time_ns() // 1_000_000,
        )

    def _record_quality_signal(
        self,
        capture_id: str,
        signal: FaceCaptureSignal,
        *,
        observed_at_unix_milliseconds: int,
    ) -> None:
        event: tuple[str, dict[str, Any]] | None = None
        with self._lock:
            session = self._sessions.get(capture_id)
            if session is None or session.status not in ACTIVE_FACE_CAPTURE_STATUSES:
                return
            previous = (session.status, session.feedback_code, self._recommended_camera_locked(session))
            session.signals_by_camera[signal.camera_index] = signal
            quality = signal.quality
            if quality is not None and quality.acceptable:
                key = (signal.camera_index, signal.track_id)
                if session.accepted_observations:
                    last_time, last_key = session.accepted_observations[-1]
                    if (
                        signal.observed_monotonic - last_time > self.quality_config.acceptance_window_seconds
                        or last_key != key
                    ):
                        session.accepted_observations.clear()
                session.accepted_observations.append((signal.observed_monotonic, key))
                while (
                    session.accepted_observations
                    and signal.observed_monotonic - session.accepted_observations[0][0]
                    > self.quality_config.acceptance_window_seconds
                ):
                    session.accepted_observations.popleft()
                if session.best_quality is None or quality.quality_score >= session.best_quality.quality_score:
                    session.best_quality = quality
                    session.best_signal = signal
                session.status = "capturing"
                session.feedback_code = "CAPTURING"
                session.feedback_message = "Good position. Hold still."
                if len(session.accepted_observations) >= self.quality_config.required_acceptable_observations:
                    assert session.best_quality is not None and session.best_signal is not None
                    success, encoded = cv2.imencode(
                        ".jpg",
                        session.best_quality.crop,
                        [cv2.IMWRITE_JPEG_QUALITY, self.quality_config.jpeg_quality],
                    )
                    if not success:
                        session.accepted_observations.clear()
                        session.feedback_code = "CAPTURE_ENCODING_FAILED"
                        session.feedback_message = "Could not encode the face image; hold still and retry."
                    else:
                        session.image_jpeg = encoded.tobytes()
                        session.image_sha256 = hashlib.sha256(session.image_jpeg).hexdigest()
                        session.status = "ready"
                        session.feedback_code = "READY"
                        session.feedback_message = "Face photograph is ready."
                        session.captured_at_unix_milliseconds = observed_at_unix_milliseconds
                        session.terminal_at_unix_milliseconds = time.time_ns() // 1_000_000
                        session.image_expires_monotonic = time.monotonic() + self.ready_ttl_seconds
                        payload = self._payload_locked(session, time.monotonic())
                        event = ("face_capture_ready", payload)
            else:
                session.accepted_observations.clear()
                selected = self._best_recent_signal_locked(session, signal.observed_monotonic)
                if selected is not None:
                    session.status = "adjust_position"
                    session.feedback_code = selected.feedback_code
                    session.feedback_message = selected.feedback_message

            current = (session.status, session.feedback_code, self._recommended_camera_locked(session))
            if event is None and current != previous:
                event = (
                    "face_capture_feedback_changed",
                    self._payload_locked(session, time.monotonic()),
                )
        if event is not None:
            event_type, payload = event
            self._emit(event_type, payload)
            if event_type == "face_capture_ready":
                print(
                    f"FACE_CAPTURE_READY capture_id={capture_id} "
                    f"camera_index={payload.get('cameraIndex')} "
                    f"sequence={payload.get('rgbSequenceNumber')} "
                    f"quality={float(payload.get('qualityScore') or 0.0):.3f}"
                )
            elif event_type == "face_capture_feedback_changed":
                feedback = payload.get("feedback") or {}
                print(
                    f"FACE_CAPTURE_FEEDBACK capture_id={capture_id} "
                    f"code={feedback.get('code')} "
                    f"camera_index={feedback.get('cameraIndex')}"
                )

    def _best_recent_signal_locked(
        self,
        session: FaceCaptureSession,
        now: float,
    ) -> FaceCaptureSignal | None:
        recent = [
            signal
            for signal in session.signals_by_camera.values()
            if now - signal.observed_monotonic <= self.quality_config.recent_signal_seconds
        ]
        return max(recent, key=lambda item: item.signal_score, default=None)

    def _recommended_camera_locked(self, session: FaceCaptureSession) -> int | None:
        signal = self._best_recent_signal_locked(session, time.monotonic())
        return None if signal is None else signal.camera_index

    def _refresh_feedback_locked(
        self,
        session: FaceCaptureSession,
        now: float,
    ) -> None:
        if session.status not in ACTIVE_FACE_CAPTURE_STATUSES:
            return
        selected = self._best_recent_signal_locked(session, now)
        if selected is not None:
            return
        session.accepted_observations.clear()
        session.status = "locating_customer"
        if session.visit_id is None and session.customer_id is not None:
            session.feedback_code = "CUSTOMER_NOT_BOUND"
            session.feedback_message = "The customer is not bound to an active visit yet."
        else:
            session.feedback_code = "CUSTOMER_NOT_VISIBLE"
            session.feedback_message = "Stand where a camera can see you."

    def _quality_payload(self, quality: FaceQualityResult | None, accepted: int) -> dict[str, Any] | None:
        if quality is None:
            return None
        return {
            "detectionScore": quality.detection_score,
            "faceWidthPixels": quality.face_width_pixels,
            "faceHeightPixels": quality.face_height_pixels,
            "yawDegrees": quality.yaw_degrees,
            "pitchDegrees": quality.pitch_degrees,
            "rollDegrees": quality.roll_degrees,
            "sharpness": quality.sharpness,
            "brightness": quality.brightness,
            "darkFraction": quality.dark_fraction,
            "brightFraction": quality.bright_fraction,
            "acceptedObservations": accepted,
            "requiredObservations": self.quality_config.required_acceptable_observations,
            "failures": list(quality.failures),
        }

    def _payload_locked(self, session: FaceCaptureSession, now: float) -> dict[str, Any]:
        selected = session.best_signal if session.status == "ready" else self._best_recent_signal_locked(session, now)
        quality = session.best_quality if session.status == "ready" else (None if selected is None else selected.quality)
        camera_index = None if selected is None else selected.camera_index
        expires_in = (
            session.image_expires_monotonic - now
            if session.status == "ready" and session.image_expires_monotonic is not None
            else session.capture_deadline_monotonic - now
        )
        expires_at_ms = time.time_ns() // 1_000_000 + max(0, int(expires_in * 1000.0))
        return {
            "captureId": session.capture_id,
            "statusUrl": f"/face-captures/{session.capture_id}",
            "status": session.status,
            "purpose": session.purpose,
            "visitId": session.visit_id,
            "customerId": session.customer_id,
            "createdAtUnixMilliseconds": session.created_unix_milliseconds,
            "expiresAtUnixMilliseconds": expires_at_ms,
            "feedback": {
                "code": session.feedback_code,
                "message": session.feedback_message,
                "cameraIndex": camera_index,
                "cameraNumber": None if camera_index is None else camera_index + 1,
                "deviceId": None if selected is None else selected.device_id,
            },
            "quality": self._quality_payload(
                quality,
                len(session.accepted_observations),
            ),
            "cameraIndex": camera_index if session.status == "ready" else None,
            "cameraNumber": None if session.status != "ready" or camera_index is None else camera_index + 1,
            "deviceId": None if session.status != "ready" or selected is None else selected.device_id,
            "trackId": None if session.status != "ready" or selected is None else selected.track_id,
            "rgbSequenceNumber": None if session.status != "ready" or selected is None else selected.rgb_sequence_number,
            "capturedAtUnixMilliseconds": session.captured_at_unix_milliseconds,
            "qualityScore": None if session.status != "ready" or quality is None else quality.quality_score,
            "width": None if session.status != "ready" or quality is None else int(quality.crop.shape[1]),
            "height": None if session.status != "ready" or quality is None else int(quality.crop.shape[0]),
            "imageUrl": f"/face-captures/{session.capture_id}/image" if session.status == "ready" else None,
            "livenessPerformed": False,
        }

    def _cleanup_locked(self, now: float) -> None:
        now_ms = time.time_ns() // 1_000_000
        for session in list(self._sessions.values()):
            if session.status in ACTIVE_FACE_CAPTURE_STATUSES and now >= session.capture_deadline_monotonic:
                session.status = "expired"
                session.feedback_code = "EXPIRED"
                session.feedback_message = "Face capture timed out."
                session.terminal_at_unix_milliseconds = now_ms
                session.image_jpeg = None
                session.image_sha256 = None
                self._emit("face_capture_expired", {"captureId": session.capture_id, "status": "expired"})
                print(
                    f"FACE_CAPTURE_EXPIRED capture_id={session.capture_id} "
                    f"last_feedback={session.feedback_code}"
                )
            elif (
                session.status == "ready"
                and session.image_expires_monotonic is not None
                and now >= session.image_expires_monotonic
            ):
                session.status = "expired"
                session.feedback_code = "EXPIRED"
                session.feedback_message = "The face image expired."
                session.image_jpeg = None
                session.image_sha256 = None
                session.terminal_at_unix_milliseconds = now_ms

        expired_tombstone_cutoff_ms = now_ms - 300_000
        for session in list(self._sessions.values()):
            if (
                session.status in {"expired", "cancelled", "failed"}
                and session.terminal_at_unix_milliseconds is not None
                and session.terminal_at_unix_milliseconds < expired_tombstone_cutoff_ms
            ):
                self._remove_session_locked(session)

    def _remove_session_locked(self, session: FaceCaptureSession) -> None:
        self._sessions.pop(session.capture_id, None)
        if session.idempotency_key:
            self._idempotency.pop(session.idempotency_key, None)

    def _fail_session(self, capture_id: str, code: str, message: str) -> None:
        with self._lock:
            session = self._sessions.get(capture_id)
            if session is None or session.status not in ACTIVE_FACE_CAPTURE_STATUSES:
                return
            session.status = "failed"
            session.feedback_code = code
            session.feedback_message = message
            session.terminal_at_unix_milliseconds = time.time_ns() // 1_000_000
            payload = self._payload_locked(session, time.monotonic())
        self._emit("face_capture_failed", payload)

    def _emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(event_type, payload)
