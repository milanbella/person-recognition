from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Sequence, Tuple

import cv2
import numpy as np

from pipeline.config import (
    DEFAULT_PERSON_TRACKER_BACKEND,
    DEFAULT_TRACKING_IOU_THRESHOLD,
    DEFAULT_TRACKING_MAX_MISSED,
)
from pipeline.detection import Detection, add_detection_args


@dataclass
class Track:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    hits: int = 1
    missed_frames: int = 0
    status: str = "NEW"
    history: List[Tuple[float, float]] = field(default_factory=list)

    def centroid(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def update_from_detection(self, detection: Detection) -> None:
        self.x1 = detection.x1
        self.y1 = detection.y1
        self.x2 = detection.x2
        self.y2 = detection.y2
        self.score = detection.score
        self.hits += 1
        self.missed_frames = 0
        self.status = "TRACKED" if self.hits > 1 else "NEW"
        self.history.append(self.centroid())
        self.history = self.history[-20:]


class PersonTracker(Protocol):
    def update(self, detections: Sequence[Detection]) -> List[Track]:
        ...


def build_tracking_argparser(
    description: str = "Step 3/4: host-side tracking on top of host-side SCRFD detections.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_detection_args(parser)
    add_tracking_args(parser)
    return parser


def add_tracking_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--tracker-backend",
        choices=["iou"],
        default=DEFAULT_PERSON_TRACKER_BACKEND,
        help="Person tracker backend. Current default is simple IoU tracking.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_TRACKING_IOU_THRESHOLD,
        help="Minimum IoU for matching a detection to an existing track.",
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=DEFAULT_TRACKING_MAX_MISSED,
        help="How many consecutive frames a track may be unmatched before removal.",
    )
    return parser


def build_person_tracker(args: argparse.Namespace) -> PersonTracker:
    backend = getattr(args, "tracker_backend", DEFAULT_PERSON_TRACKER_BACKEND)
    if backend != "iou":
        raise ValueError(f"Unsupported tracker backend: {backend}")
    return SimpleIoUTracker(
        iou_threshold=args.iou_threshold,
        max_missed=args.max_missed,
    )


def compute_iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


class SimpleIoUTracker:
    def __init__(self, iou_threshold: float, max_missed: int) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}
        self.last_logged_status: Dict[int, str] = {}

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        unmatched_track_ids = set(self.tracks.keys())
        unmatched_detection_indices = set(range(len(detections)))
        matches: List[Tuple[int, int]] = []

        candidate_pairs: List[Tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            track_box = (track.x1, track.y1, track.x2, track.y2)
            for det_idx, detection in enumerate(detections):
                det_box = (detection.x1, detection.y1, detection.x2, detection.y2)
                iou = compute_iou(track_box, det_box)
                if iou >= self.iou_threshold:
                    candidate_pairs.append((iou, track_id, det_idx))

        candidate_pairs.sort(reverse=True, key=lambda item: item[0])

        for _iou, track_id, det_idx in candidate_pairs:
            if track_id not in unmatched_track_ids or det_idx not in unmatched_detection_indices:
                continue
            matches.append((track_id, det_idx))
            unmatched_track_ids.remove(track_id)
            unmatched_detection_indices.remove(det_idx)

        for track_id, det_idx in matches:
            self.tracks[track_id].update_from_detection(detections[det_idx])

        removed_track_ids: List[int] = []
        for track_id in unmatched_track_ids:
            track = self.tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missed:
                track.status = "REMOVED"
                removed_track_ids.append(track_id)
            else:
                track.status = "LOST"

        for det_idx in unmatched_detection_indices:
            detection = detections[det_idx]
            track = Track(
                track_id=self.next_track_id,
                x1=detection.x1,
                y1=detection.y1,
                x2=detection.x2,
                y2=detection.y2,
                score=detection.score,
            )
            track.history.append(track.centroid())
            self.tracks[track.track_id] = track
            self.next_track_id += 1

        self._log_status_changes(removed_track_ids)

        active_tracks = [track for track in self.tracks.values() if track.status != "REMOVED"]
        for track_id in removed_track_ids:
            self.tracks.pop(track_id, None)
            self.last_logged_status.pop(track_id, None)

        active_tracks.sort(key=lambda track: track.track_id)
        return active_tracks

    def _log_status_changes(self, removed_track_ids: Sequence[int]) -> None:
        for track in self.tracks.values():
            previous = self.last_logged_status.get(track.track_id)
            if previous != track.status:
                cx, cy = track.centroid()
                print(
                    f"Track id={track.track_id} status={track.status} "
                    f"centroid=({cx:.1f}, {cy:.1f}) score={track.score:.3f}"
                )
                self.last_logged_status[track.track_id] = track.status

        for track_id in removed_track_ids:
            print(f"Track id={track_id} status=REMOVED")


IouPersonTracker = SimpleIoUTracker


def draw_tracks(frame: np.ndarray, tracks: Sequence[Track]) -> None:
    for track in tracks:
        if track.status == "LOST":
            color = (0, 165, 255)
        else:
            color = (255, 200, 0) if track.status == "NEW" else (0, 255, 0)

        cv2.rectangle(frame, (track.x1, track.y1), (track.x2, track.y2), color, 2)
        cv2.putText(
            frame,
            f"ID {track.track_id} {track.status} {track.score:.2f}",
            (track.x1, max(20, track.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        for idx in range(1, len(track.history)):
            pt1 = (int(track.history[idx - 1][0]), int(track.history[idx - 1][1]))
            pt2 = (int(track.history[idx][0]), int(track.history[idx][1]))
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
