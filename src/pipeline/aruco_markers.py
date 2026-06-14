from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


DEFAULT_ARUCO_DICTIONARY = "DICT_4X4_50"
DEFAULT_DOOR_MARKER_IDS = (0, 1, 2, 3)


@dataclass(frozen=True)
class ArucoMarkerDetection:
    marker_id: int
    corners_px: tuple[tuple[float, float], ...]
    center_px: tuple[float, float]


@dataclass(frozen=True)
class ArucoDetectionResult:
    detections: list[ArucoMarkerDetection]
    rejected_candidates: list[tuple[tuple[float, float], ...]]


def get_aruco_dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This OpenCV build does not include cv2.aruco. "
            "Install opencv-contrib-python in the active environment."
        )

    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        supported = sorted(
            name
            for name in dir(cv2.aruco)
            if name.startswith("DICT_") and isinstance(getattr(cv2.aruco, name), int)
        )
        raise ValueError(
            f"Unknown ArUco dictionary {dictionary_name!r}. "
            f"Supported examples: {', '.join(supported[:12])}"
        )

    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _create_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def _corners_to_tuple(corners: np.ndarray) -> tuple[tuple[float, float], ...]:
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    return tuple((float(x), float(y)) for x, y in points)


def _marker_center(corners_px: Sequence[tuple[float, float]]) -> tuple[float, float]:
    points = np.asarray(corners_px, dtype=np.float32)
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def detect_aruco_markers(
    frame: np.ndarray,
    *,
    dictionary_name: str = DEFAULT_ARUCO_DICTIONARY,
) -> ArucoDetectionResult:
    dictionary = get_aruco_dictionary(dictionary_name)
    parameters = _create_detector_parameters()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters,
        )

    detections: list[ArucoMarkerDetection] = []
    if marker_ids is not None:
        for raw_id, raw_corners in zip(marker_ids.flatten(), marker_corners):
            corners_px = _corners_to_tuple(raw_corners)
            detections.append(
                ArucoMarkerDetection(
                    marker_id=int(raw_id),
                    corners_px=corners_px,
                    center_px=_marker_center(corners_px),
                )
            )

    rejected = [
        _corners_to_tuple(candidate)
        for candidate in (rejected_candidates if rejected_candidates is not None else [])
    ]
    return ArucoDetectionResult(detections=detections, rejected_candidates=rejected)


def draw_aruco_detections(
    frame: np.ndarray,
    detections: Sequence[ArucoMarkerDetection],
    *,
    door_marker_ids: set[int],
) -> None:
    for detection in detections:
        is_door_marker = detection.marker_id in door_marker_ids
        color = (0, 220, 0) if is_door_marker else (220, 140, 0)
        corners = np.asarray(detection.corners_px, dtype=np.int32).reshape(-1, 1, 2)
        center_x, center_y = (int(round(v)) for v in detection.center_px)
        label = f"ID {detection.marker_id}"
        if is_door_marker:
            label += " DOOR"

        cv2.polylines(frame, [corners], isClosed=True, color=color, thickness=3)
        cv2.circle(frame, (center_x, center_y), radius=6, color=color, thickness=-1)
        cv2.putText(
            frame,
            label,
            (center_x + 10, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"center=({center_x},{center_y})",
            (center_x + 10, center_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_rejected_aruco_candidates(
    frame: np.ndarray,
    rejected_candidates: Sequence[Sequence[tuple[float, float]]],
) -> None:
    for candidate in rejected_candidates:
        corners = np.asarray(candidate, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 0, 220), thickness=1)


def format_marker_summary(
    detections: Sequence[ArucoMarkerDetection],
    *,
    door_marker_ids: set[int],
) -> str:
    if not detections:
        return "No ArUco markers detected."

    parts = []
    for detection in sorted(detections, key=lambda item: item.marker_id):
        center_x, center_y = (round(v, 1) for v in detection.center_px)
        role = "door" if detection.marker_id in door_marker_ids else "other"
        parts.append(f"id={detection.marker_id} {role} center=({center_x},{center_y})")
    return "Detected ArUco markers: " + "; ".join(parts)
