from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import cv2
import numpy as np

from pipeline.depth import DepthSample
from pipeline.face_identity import RecognizedFace
from pipeline.tracking import Track


DEFAULT_VISIT_APPEARANCE_THRESHOLD = 0.60
DEFAULT_VISIT_SAME_CAMERA_MAX_AGE_SECONDS = 8.0
DEFAULT_VISIT_CROSS_CAMERA_MAX_AGE_SECONDS = 4.0


@dataclass
class BodyAppearance:
    histogram: np.ndarray
    aspect_ratio: float
    height_px: int


@dataclass
class VisitIdentity:
    visit_id: int
    last_seen_host_seconds: float
    last_device_id: str
    last_track_id: int
    last_bbox: tuple[int, int, int, int]
    appearance: BodyAppearance | None
    depth_mm: float | None
    observation_count: int = 0
    face_identity_ids: set[str] = field(default_factory=set)


@dataclass
class VisitAssignment:
    visit_id: int
    track_id: int
    device_id: str
    face_identity_ids: tuple[str, ...]
    matched_score: float | None
    origin: str = "local"


def add_visit_identity_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--visit-match-threshold",
        type=float,
        default=DEFAULT_VISIT_APPEARANCE_THRESHOLD,
        help="Minimum within-visit body/depth/time score required to attach a new track to an existing visit.",
    )
    parser.add_argument(
        "--visit-same-camera-max-age-seconds",
        type=float,
        default=DEFAULT_VISIT_SAME_CAMERA_MAX_AGE_SECONDS,
        help="How long a same-camera visit remains eligible for new-track reattachment.",
    )
    parser.add_argument(
        "--visit-cross-camera-max-age-seconds",
        type=float,
        default=DEFAULT_VISIT_CROSS_CAMERA_MAX_AGE_SECONDS,
        help="How long a different-camera visit remains eligible for cross-camera reattachment.",
    )
    return parser


def extract_body_appearance(frame: np.ndarray, track: Track) -> BodyAppearance | None:
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(track.x1)))
    y1 = max(0, min(height - 1, int(track.y1)))
    x2 = max(0, min(width, int(track.x2)))
    y2 = max(0, min(height, int(track.y2)))
    if x2 <= x1 + 4 or y2 <= y1 + 8:
        return None

    crop = frame[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    if crop_h < 12 or crop_w < 6:
        return None

    upper = crop[int(crop_h * 0.15) : max(int(crop_h * 0.55), int(crop_h * 0.15) + 1), :]
    lower = crop[int(crop_h * 0.55) : max(int(crop_h * 0.95), int(crop_h * 0.55) + 1), :]
    upper_hist = _hsv_histogram(upper)
    lower_hist = _hsv_histogram(lower)
    histogram = np.concatenate([upper_hist, lower_hist]).astype(np.float32)
    norm = float(np.linalg.norm(histogram))
    if norm <= 1e-8:
        return None

    return BodyAppearance(
        histogram=histogram / norm,
        aspect_ratio=float(crop_w) / float(crop_h),
        height_px=crop_h,
    )


def draw_visit_labels(
    frame: np.ndarray,
    tracks: Sequence[Track],
    assignments: Mapping[int, VisitAssignment],
) -> None:
    for track in tracks:
        assignment = assignments.get(track.track_id)
        if assignment is None:
            continue
        face_suffix = ""
        if assignment.face_identity_ids:
            compact_faces = ",".join(face_id.replace("face_person_", "F") for face_id in assignment.face_identity_ids)
            face_suffix = f" {compact_faces}"
        origin_suffix = ""
        if assignment.origin == "entrance_confirmed":
            origin_suffix = "E"
        elif assignment.origin == "observer_only":
            origin_suffix = "O"
        label = f"V{assignment.visit_id}{origin_suffix}{face_suffix}"
        y = min(frame.shape[0] - 10, max(40, track.y1 + 22))
        cv2.putText(
            frame,
            label,
            (track.x1, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (track.x1, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


class VisitIdentityManager:
    def __init__(
        self,
        *,
        match_threshold: float = DEFAULT_VISIT_APPEARANCE_THRESHOLD,
        same_camera_max_age_seconds: float = DEFAULT_VISIT_SAME_CAMERA_MAX_AGE_SECONDS,
        cross_camera_max_age_seconds: float = DEFAULT_VISIT_CROSS_CAMERA_MAX_AGE_SECONDS,
    ) -> None:
        self.match_threshold = match_threshold
        self.same_camera_max_age_seconds = same_camera_max_age_seconds
        self.cross_camera_max_age_seconds = cross_camera_max_age_seconds
        self.next_visit_id = 1
        self.visits: dict[int, VisitIdentity] = {}
        self.track_to_visit: dict[tuple[str, int], int] = {}
        self.face_to_visit: dict[str, int] = {}

    def update(
        self,
        *,
        device_id: str,
        host_seconds: float,
        frame: np.ndarray,
        tracks: Sequence[Track],
        depth_samples: Mapping[int, DepthSample],
        recognized_faces: Sequence[RecognizedFace],
        body_appearances: Mapping[int, BodyAppearance] | None = None,
    ) -> dict[int, VisitAssignment]:
        faces_by_track: dict[int, set[str]] = {}
        for face in recognized_faces:
            if face.track_id is None:
                continue
            faces_by_track.setdefault(face.track_id, set()).add(face.identity_id)

        assignments: dict[int, VisitAssignment] = {}
        for track in tracks:
            if track.status == "REMOVED":
                continue

            track_key = (device_id, track.track_id)
            appearance = (
                body_appearances.get(track.track_id)
                if body_appearances is not None
                else extract_body_appearance(frame, track)
            )
            depth_sample = depth_samples.get(track.track_id)
            depth_mm = None if depth_sample is None else depth_sample.depth_mm
            bbox = (track.x1, track.y1, track.x2, track.y2)
            face_ids = faces_by_track.get(track.track_id, set())

            matched_score: float | None = None
            visit_id = self.track_to_visit.get(track_key)
            if visit_id is None:
                visit_id = self._visit_from_known_face(face_ids)
                if visit_id is None:
                    visit_id, matched_score = self._match_or_create_visit(
                        device_id=device_id,
                        track_id=track.track_id,
                        host_seconds=host_seconds,
                        bbox=bbox,
                        appearance=appearance,
                        depth_mm=depth_mm,
                    )
                self.track_to_visit[track_key] = visit_id

            visit = self.visits[visit_id]
            visit.face_identity_ids.update(face_ids)
            for face_id in face_ids:
                self.face_to_visit[face_id] = visit_id
            self._update_visit(
                visit,
                device_id=device_id,
                track_id=track.track_id,
                host_seconds=host_seconds,
                bbox=bbox,
                appearance=appearance,
                depth_mm=depth_mm,
            )
            assignments[track.track_id] = VisitAssignment(
                visit_id=visit.visit_id,
                track_id=track.track_id,
                device_id=device_id,
                face_identity_ids=tuple(sorted(visit.face_identity_ids)),
                matched_score=matched_score,
            )

        return assignments

    def _visit_from_known_face(self, face_ids: set[str]) -> int | None:
        for face_id in sorted(face_ids):
            visit_id = self.face_to_visit.get(face_id)
            if visit_id is not None and visit_id in self.visits:
                return visit_id
        return None

    def _match_or_create_visit(
        self,
        *,
        device_id: str,
        track_id: int,
        host_seconds: float,
        bbox: tuple[int, int, int, int],
        appearance: BodyAppearance | None,
        depth_mm: float | None,
    ) -> tuple[int, float | None]:
        best_visit_id: int | None = None
        best_score = -1.0
        for visit in self.visits.values():
            age_seconds = host_seconds - visit.last_seen_host_seconds
            max_age = (
                self.same_camera_max_age_seconds
                if visit.last_device_id == device_id
                else self.cross_camera_max_age_seconds
            )
            if age_seconds < 0.0 or age_seconds > max_age:
                continue
            score = self._score_candidate(
                visit=visit,
                device_id=device_id,
                age_seconds=age_seconds,
                bbox=bbox,
                appearance=appearance,
                depth_mm=depth_mm,
                max_age=max_age,
            )
            if score > best_score:
                best_score = score
                best_visit_id = visit.visit_id

        if best_visit_id is not None and best_score >= self.match_threshold:
            return best_visit_id, best_score

        visit_id = self.next_visit_id
        self.next_visit_id += 1
        self.visits[visit_id] = VisitIdentity(
            visit_id=visit_id,
            last_seen_host_seconds=host_seconds,
            last_device_id=device_id,
            last_track_id=track_id,
            last_bbox=bbox,
            appearance=None,
            depth_mm=None,
        )
        return visit_id, None if best_visit_id is None else best_score

    def _score_candidate(
        self,
        *,
        visit: VisitIdentity,
        device_id: str,
        age_seconds: float,
        bbox: tuple[int, int, int, int],
        appearance: BodyAppearance | None,
        depth_mm: float | None,
        max_age: float,
    ) -> float:
        appearance_score = _appearance_similarity(appearance, visit.appearance)
        time_score = max(0.0, 1.0 - (age_seconds / max(max_age, 1e-6)))
        depth_score = _depth_similarity(depth_mm, visit.depth_mm)
        aspect_score = _aspect_similarity(appearance, visit.appearance)

        score = (0.65 * appearance_score) + (0.15 * time_score) + (0.15 * depth_score) + (0.05 * aspect_score)
        if visit.last_device_id != device_id:
            score -= 0.05

        return max(0.0, min(1.0, score))

    def _update_visit(
        self,
        visit: VisitIdentity,
        *,
        device_id: str,
        track_id: int,
        host_seconds: float,
        bbox: tuple[int, int, int, int],
        appearance: BodyAppearance | None,
        depth_mm: float | None,
    ) -> None:
        count = visit.observation_count
        if appearance is not None:
            if visit.appearance is None:
                visit.appearance = appearance
            else:
                merged_hist = ((visit.appearance.histogram * count) + appearance.histogram) / (count + 1)
                norm = float(np.linalg.norm(merged_hist))
                if norm > 1e-8:
                    merged_hist = merged_hist / norm
                visit.appearance = BodyAppearance(
                    histogram=merged_hist.astype(np.float32),
                    aspect_ratio=((visit.appearance.aspect_ratio * count) + appearance.aspect_ratio) / (count + 1),
                    height_px=int(round(((visit.appearance.height_px * count) + appearance.height_px) / (count + 1))),
                )
        if depth_mm is not None:
            visit.depth_mm = depth_mm if visit.depth_mm is None else ((visit.depth_mm * count) + depth_mm) / (count + 1)

        visit.last_seen_host_seconds = host_seconds
        visit.last_device_id = device_id
        visit.last_track_id = track_id
        visit.last_bbox = bbox
        visit.observation_count += 1


def _hsv_histogram(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.zeros((16 * 8,), dtype=np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    hist = hist.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(hist))
    return hist if norm <= 1e-8 else hist / norm


def _appearance_similarity(left: BodyAppearance | None, right: BodyAppearance | None) -> float:
    if left is None or right is None:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left.histogram, right.histogram))))


def _depth_similarity(left_mm: float | None, right_mm: float | None) -> float:
    if left_mm is None or right_mm is None:
        return 0.0
    return max(0.0, 1.0 - (abs(left_mm - right_mm) / 800.0))


def _aspect_similarity(left: BodyAppearance | None, right: BodyAppearance | None) -> float:
    if left is None or right is None:
        return 0.0
    return max(0.0, 1.0 - (abs(left.aspect_ratio - right.aspect_ratio) / 0.5))
