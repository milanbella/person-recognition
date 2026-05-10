from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pipeline.config import (
    DEFAULT_ENTRANCE_LINE_AXIS,
    DEFAULT_ENTRANCE_LINE_POSITION,
    DEFAULT_ENTRANCE_MIN_HISTORY,
    DEFAULT_OUTSIDE_SIDE,
)
from pipeline.detection import add_detection_args
from pipeline.tracking import add_tracking_args
from pipeline.tracking import Track


@dataclass
class EntranceState:
    last_side: Optional[str] = None
    entered: bool = False
    centroids: List[Tuple[float, float]] = field(default_factory=list)


def build_entrance_argparser(
    description: str = "Step 6: host-side entrance-line logic on top of SCRFD tracking.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_detection_args(parser)
    add_tracking_args(parser)
    add_entrance_args(parser)
    return parser


def add_entrance_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--line-axis",
        choices=["x", "y"],
        default=DEFAULT_ENTRANCE_LINE_AXIS,
        help="Axis for the entrance line. 'x' means horizontal line, 'y' means vertical line.",
    )
    parser.add_argument(
        "--line-position",
        type=float,
        default=DEFAULT_ENTRANCE_LINE_POSITION,
        help="Normalized line position in the frame, between 0.0 and 1.0.",
    )
    parser.add_argument(
        "--outside-side",
        choices=["less", "greater"],
        default=DEFAULT_OUTSIDE_SIDE,
        help="Which side of the line is considered outside.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=DEFAULT_ENTRANCE_MIN_HISTORY,
        help="Minimum number of centroids before an entry event may be emitted.",
    )
    parser.add_argument(
        "--debug-entrance",
        action="store_true",
        help="Show centroid/side debug overlays and print side transitions.",
    )
    return parser


def classify_side(
    centroid: Tuple[float, float],
    axis: str,
    line_position: float,
    frame_shape: Tuple[int, int],
) -> str:
    frame_height, frame_width = frame_shape
    threshold = line_position * (frame_height if axis == "x" else frame_width)
    value = centroid[1] if axis == "x" else centroid[0]
    return "less" if value < threshold else "greater"


def centroid_for_track(track: Track) -> Tuple[float, float]:
    return track.centroid()


def draw_entrance_line(
    frame: np.ndarray,
    axis: str,
    line_position: float,
    outside_side: str,
) -> None:
    height, width = frame.shape[:2]
    color = (255, 0, 255)

    if axis == "x":
        y = int(round(line_position * height))
        cv2.line(frame, (0, y), (width, y), color, 2, cv2.LINE_AA)
        outside_label = "outside" if outside_side == "less" else "inside"
        inside_label = "inside" if outside_side == "less" else "outside"
        cv2.putText(
            frame,
            outside_label,
            (20, max(30, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            inside_label,
            (20, min(height - 20, y + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    else:
        x = int(round(line_position * width))
        cv2.line(frame, (x, 0), (x, height), color, 2, cv2.LINE_AA)
        outside_label = "outside" if outside_side == "less" else "inside"
        inside_label = "inside" if outside_side == "less" else "outside"
        cv2.putText(
            frame,
            outside_label,
            (max(10, x - 110), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            inside_label,
            (min(width - 110, x + 12), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def process_entrance_logic(
    tracks: Sequence[Track],
    states: Dict[int, EntranceState],
    axis: str,
    line_position: float,
    frame_shape: Tuple[int, int],
    outside_side: str,
    min_history: int,
    debug_entrance: bool,
) -> List[int]:
    entered_track_ids: List[int] = []

    active_ids = {track.track_id for track in tracks if track.status != "REMOVED"}
    for track_id in list(states.keys()):
        if track_id not in active_ids:
            states.pop(track_id, None)

    for track in tracks:
        if track.status not in {"NEW", "TRACKED", "LOST"}:
            continue

        centroid = centroid_for_track(track)
        state = states.setdefault(track.track_id, EntranceState())
        state.centroids.append(centroid)
        state.centroids = state.centroids[-20:]

        current_side = classify_side(centroid, axis, line_position, frame_shape)
        if state.last_side is None:
            state.last_side = current_side
            if debug_entrance:
                print(
                    f"ENTRANCE_DEBUG track_id={track.track_id} "
                    f"init_side={current_side} history={len(state.centroids)} "
                    f"status={track.status}"
                )
            continue

        if debug_entrance and current_side != state.last_side:
            print(
                f"ENTRANCE_DEBUG track_id={track.track_id} "
                f"side_change={state.last_side}->{current_side} "
                f"history={len(state.centroids)} status={track.status} "
                f"outside_side={outside_side} entered={state.entered}"
            )

        if (
            not state.entered
            and len(state.centroids) >= min_history
            and state.last_side == outside_side
            and current_side != outside_side
            and track.status in {"TRACKED", "LOST"}
        ):
            state.entered = True
            entered_track_ids.append(track.track_id)
        elif debug_entrance and not state.entered and current_side != state.last_side:
            print(
                f"ENTRANCE_DEBUG track_id={track.track_id} "
                f"event_suppressed history={len(state.centroids)} "
                f"last_side={state.last_side} current_side={current_side} "
                f"required_outside_side={outside_side} status={track.status}"
            )

        state.last_side = current_side

    return entered_track_ids


def draw_entry_events(
    frame: np.ndarray,
    entered_track_ids: Sequence[int],
) -> None:
    if not entered_track_ids:
        return

    text = "Entered: " + ", ".join(str(track_id) for track_id in entered_track_ids)
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_entrance_debug(
    frame: np.ndarray,
    tracks: Sequence[Track],
    states: Dict[int, EntranceState],
    axis: str,
    line_position: float,
    outside_side: str,
) -> None:
    for track in tracks:
        state = states.get(track.track_id)
        if state is None or not state.centroids:
            continue

        cx, cy = state.centroids[-1]
        center = (int(cx), int(cy))
        side = classify_side((cx, cy), axis, line_position, frame.shape[:2])
        color = (255, 0, 255) if side == outside_side else (0, 0, 255)

        # Draw a high-contrast centroid marker that stays visible on bright clothes/backgrounds.
        cv2.circle(frame, center, 10, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, center, 7, color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, 14, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (center[0] - 16, center[1]), (center[0] + 16, center[1]), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (center[0], center[1] - 16), (center[0], center[1] + 16), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (center[0] - 16, center[1]), (center[0] + 16, center[1]), color, 1, cv2.LINE_AA)
        cv2.line(frame, (center[0], center[1] - 16), (center[0], center[1] + 16), color, 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            f"S:{side} H:{len(state.centroids)} E:{int(state.entered)}",
            (center[0] + 18, center[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"S:{side} H:{len(state.centroids)} E:{int(state.entered)}",
            (center[0] + 18, center[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    frame_height, frame_width = frame.shape[:2]
    threshold = line_position * (frame_height if axis == "x" else frame_width)
    info = (
        f"axis={axis} line={line_position:.2f} "
        f"threshold_px={threshold:.1f} "
        f"outside={outside_side}"
    )
    cv2.putText(
        frame,
        info,
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
