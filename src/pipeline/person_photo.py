from __future__ import annotations

import threading
import uuid
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from pipeline.tracking import Track
from pipeline.visit_identity import VisitAssignment


class PersonPhotoError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PersonPhotoCapture:
    def __init__(self, *, camera_device_ids: Sequence[str], maximum_pending_requests: int = 8):
        self.camera_device_ids = tuple(camera_device_ids)
        self.maximum_pending_requests = maximum_pending_requests
        self._lock = threading.RLock()
        self._person_requests: dict[str, dict[str, Any]] = {}
        self._visit_state: dict[int, tuple[str, str | None, str]] = {}

    def capture_person(self, camera_index: int, track_id: int | None = None) -> tuple[bytes, int, int]:
        jpeg, evidence = self.capture_person_evidence(camera_index, track_id)
        return jpeg, evidence["trackId"], evidence["rgbSequenceNumber"]

    def capture_person_evidence(self, camera_index: int, track_id: int | None = None) -> tuple[bytes, dict[str, Any]]:
        if not 0 <= camera_index < len(self.camera_device_ids):
            raise PersonPhotoError("CAMERA_NOT_FOUND", "Unknown camera index.")
        request_id = uuid.uuid4().hex
        request: dict[str, Any] = {
            "camera": camera_index, "track": track_id, "event": threading.Event(),
        }
        with self._lock:
            if len(self._person_requests) >= self.maximum_pending_requests:
                raise PersonPhotoError("TOO_MANY_CAPTURES", "Too many captures are pending.")
            self._person_requests[request_id] = request
        try:
            if not request["event"].wait(5.0):
                raise PersonPhotoError("CAMERA_TIMEOUT", "No current camera frame arrived within five seconds.")
            if "error" in request:
                raise request["error"]
            crop, evidence = request["result"]
            success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise PersonPhotoError("CAPTURE_ENCODING_FAILED", "Could not encode the person image.")
            return encoded.tobytes(), evidence
        finally:
            with self._lock:
                self._person_requests.pop(request_id, None)

    def _observe_person_requests(self, camera_index: int, sequence: int,
                                 frame: np.ndarray, tracks: Sequence[Track], *,
                                 observed_at_unix_milliseconds: int,
                                 assignments: Mapping[int, VisitAssignment],
                                 customer_ids_by_visit: Mapping[int, str]) -> None:
        with self._lock:
            for request in self._person_requests.values():
                if request["camera"] != camera_index or request["event"].is_set():
                    continue
                candidates = [track for track in tracks if track.status in {"NEW", "TRACKED"}
                              and (request["track"] is None or track.track_id == request["track"])]
                if len(candidates) != 1:
                    request["error"] = PersonPhotoError(
                        "PERSON_NOT_VISIBLE" if not candidates else "MULTIPLE_PEOPLE",
                        "No matching person is visible." if not candidates else
                        "Several people are visible; specify track_id.",
                    )
                else:
                    track = candidates[0]
                    height, width = frame.shape[:2]
                    crop = frame[max(0, int(track.y1)):min(height, int(track.y2)),
                                 max(0, int(track.x1)):min(width, int(track.x2))]
                    if not crop.size:
                        request["error"] = PersonPhotoError("PERSON_NOT_VISIBLE", "Person crop is empty.")
                    else:
                        assignment = assignments.get(track.track_id)
                        visit_id = None if assignment is None else assignment.visit_id
                        visit_state = self._visit_state.get(visit_id)
                        evidence = {
                            "cameraIndex": camera_index,
                            "cameraNumber": camera_index + 1,
                            "deviceId": self.camera_device_ids[camera_index],
                            "rgbSequenceNumber": sequence,
                            "observedAtUnixMilliseconds": observed_at_unix_milliseconds,
                            "trackId": track.track_id,
                            "trackStatus": track.status,
                            "detectionScore": float(track.score),
                            "visitId": visit_id,
                            "customerId": customer_ids_by_visit.get(visit_id),
                            "visitOrigin": None if assignment is None else assignment.origin,
                            "visitStatus": None if visit_state is None else visit_state[2],
                            "sourceWidth": width,
                            "sourceHeight": height,
                            "cropBox": [max(0, int(track.x1)), max(0, int(track.y1)),
                                        min(width, int(track.x2)), min(height, int(track.y2))],
                            "width": int(crop.shape[1]),
                            "height": int(crop.shape[0]),
                        }
                        request["result"] = (crop.copy(), evidence)
                request["event"].set()


    def has_pending_requests(self) -> bool:
        with self._lock:
            return bool(self._person_requests)

    def update_visit_state(self, visits: Mapping[int, tuple[str, str | None, str]]) -> None:
        with self._lock:
            self._visit_state = dict(visits)

    def observe_frame(self, *, camera_index: int, device_id: str,
                      rgb_sequence_number: int, observed_at_unix_milliseconds: int,
                      frame: np.ndarray, tracks: Sequence[Track],
                      assignments: Mapping[int, VisitAssignment],
                      customer_ids_by_visit: Mapping[int, str]) -> None:
        self._observe_person_requests(
            camera_index, rgb_sequence_number, frame, tracks,
            observed_at_unix_milliseconds=observed_at_unix_milliseconds,
            assignments=assignments, customer_ids_by_visit=customer_ids_by_visit,
        )
