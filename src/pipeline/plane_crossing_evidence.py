from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from pipeline.tracking import Track


@dataclass(frozen=True)
class PlaneEvidenceFrame:
    sequence_num: int
    host_synced_seconds: float
    frame: np.ndarray
    tracks: tuple[Track, ...]


def select_plane_evidence_frames(
    frames: Sequence[PlaneEvidenceFrame],
    *,
    crossing_sequence_num: int,
    frame_count: int,
) -> tuple[PlaneEvidenceFrame, ...]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    ordered = sorted(frames, key=lambda item: item.sequence_num)
    if len(ordered) <= frame_count:
        return tuple(ordered)
    crossing_index = min(
        range(len(ordered)),
        key=lambda index: abs(ordered[index].sequence_num - crossing_sequence_num),
    )
    before = frame_count // 2
    start = max(0, crossing_index - before)
    end = min(len(ordered), start + frame_count)
    start = max(0, end - frame_count)
    return tuple(ordered[start:end])


def render_plane_crossing_contact_sheet(
    frames: Sequence[PlaneEvidenceFrame],
    *,
    crossing_sequence_num: int,
    track_id: int,
    visit_id: int | None,
    event_type: str,
    plane_signed_distance_mm: float | None,
    depth_mm: float,
    depth_roi: tuple[int, int, int, int],
    thumbnail_width: int = 640,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if not frames:
        raise ValueError("At least one evidence frame is required.")
    rendered: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for evidence_frame in frames:
        image = evidence_frame.frame.copy()
        track = next(
            (item for item in evidence_frame.tracks if item.track_id == track_id),
            None,
        )
        track_box = None
        if track is not None:
            track_box = (track.x1, track.y1, track.x2, track.y2)
            cv2.rectangle(
                image,
                (track.x1, track.y1),
                (track.x2, track.y2),
                (0, 255, 0),
                3,
            )
        is_crossing = evidence_frame.sequence_num == crossing_sequence_num
        if is_crossing:
            x1, y1, x2, y2 = depth_roi
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 3)
        plane_text = (
            "none"
            if plane_signed_distance_mm is None
            else f"{plane_signed_distance_mm:.0f}mm"
        )
        label = (
            f"{event_type.upper()} seq={evidence_frame.sequence_num} "
            f"track={track_id} visit={visit_id}"
        )
        cv2.putText(
            image,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255) if is_crossing else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if is_crossing:
            cv2.putText(
                image,
                f"plane={plane_text} depth={depth_mm:.0f}mm orange=depth ROI",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
        height, width = image.shape[:2]
        scale = thumbnail_width / width
        thumbnail = cv2.resize(
            image,
            (thumbnail_width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        rendered.append(thumbnail)
        metadata.append(
            {
                "rgbSequenceNumber": evidence_frame.sequence_num,
                "hostSyncedSeconds": evidence_frame.host_synced_seconds,
                "isCrossingFrame": is_crossing,
                "trackBoundingBox": track_box,
            }
        )

    columns = min(3, len(rendered))
    rows = math.ceil(len(rendered) / columns)
    tile_height = max(image.shape[0] for image in rendered)
    sheet = np.zeros(
        (rows * tile_height, columns * thumbnail_width, 3),
        dtype=np.uint8,
    )
    for index, image in enumerate(rendered):
        row, column = divmod(index, columns)
        y = row * tile_height
        x = column * thumbnail_width
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    return sheet, metadata
