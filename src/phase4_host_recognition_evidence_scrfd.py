import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Sequence, Tuple

import cv2
import depthai as dai
import numpy as np

from phase1_host_detection_scrfd import (
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    ScrfdInsightFaceDetector,
)
from phase2_host_tracking_scrfd import SimpleIoUTracker, Track, draw_tracks
from phase3_host_entrance_line_scrfd import (
    EntranceState,
    build_argparser as build_entrance_argparser,
    draw_entrance_line,
    draw_entry_events,
    process_entrance_logic,
)


@dataclass
class CropSnapshot:
    frame_index: int
    crop: np.ndarray
    score: float


@dataclass
class PendingEvidenceEvent:
    event_id: int
    output_dir: Path
    remaining_post_frames: int
    saved_post_frames: int = 0


@dataclass
class TrackEvidenceState:
    recent_crops: Deque[CropSnapshot] = field(default_factory=lambda: deque(maxlen=8))
    pending_events: List[PendingEvidenceEvent] = field(default_factory=list)


def build_argparser() -> argparse.ArgumentParser:
    parser = build_entrance_argparser()
    parser.description = (
        "Step 7: host-side recognition evidence capture on top of SCRFD entrance events."
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "evidence",
        help="Directory where evidence crops will be saved.",
    )
    parser.add_argument(
        "--pre-frames",
        type=int,
        default=4,
        help="How many recent crops to keep and save before the entry event.",
    )
    parser.add_argument(
        "--post-frames",
        type=int,
        default=4,
        help="How many crops to save after the entry event.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.15,
        help="Extra crop margin as a fraction of box width/height.",
    )
    return parser


def crop_track(frame: np.ndarray, track: Track, margin: float) -> np.ndarray | None:
    height, width = frame.shape[:2]
    box_w = track.x2 - track.x1
    box_h = track.y2 - track.y1
    margin_x = int(round(box_w * margin))
    margin_y = int(round(box_h * margin))

    x1 = max(0, track.x1 - margin_x)
    y1 = max(0, track.y1 - margin_y)
    x2 = min(width, track.x2 + margin_x)
    y2 = min(height, track.y2 + margin_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None
    return crop


class EvidenceCollector:
    def __init__(self, evidence_dir: Path, pre_frames: int, post_frames: int) -> None:
        self.evidence_dir = evidence_dir
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.states: Dict[int, TrackEvidenceState] = {}
        self.next_event_id = 1
        self.last_saved_messages: List[str] = []
        evidence_dir.mkdir(parents=True, exist_ok=True)

    def update_buffers(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        frame_index: int,
        crop_margin: float,
    ) -> None:
        active_ids = {track.track_id for track in tracks if track.status != "REMOVED"}
        for track_id in list(self.states.keys()):
            if track_id not in active_ids:
                self.states.pop(track_id, None)

        for track in tracks:
            if track.status not in {"NEW", "TRACKED", "LOST"}:
                continue

            crop = crop_track(frame, track, crop_margin)
            if crop is None:
                continue

            state = self.states.setdefault(track.track_id, TrackEvidenceState())
            state.recent_crops.append(
                CropSnapshot(
                    frame_index=frame_index,
                    crop=crop,
                    score=track.score,
                )
            )

            self._save_pending_post_crops(track.track_id, state)

    def record_entry_event(self, track_id: int) -> None:
        state = self.states.get(track_id)
        if state is None or not state.recent_crops:
            print(f"EVIDENCE_WARNING track_id={track_id} no crops available at entry event")
            return

        event_id = self.next_event_id
        self.next_event_id += 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.evidence_dir / f"track_{track_id:03d}_event_{event_id:03d}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        recent = list(state.recent_crops)[-self.pre_frames :]
        for index, snapshot in enumerate(recent):
            filename = output_dir / (
                f"pre_{index:02d}_frame_{snapshot.frame_index:06d}_score_{snapshot.score:.2f}.jpg"
            )
            cv2.imwrite(str(filename), snapshot.crop)

        current = recent[-1]
        current_filename = output_dir / (
            f"event_frame_{current.frame_index:06d}_score_{current.score:.2f}.jpg"
        )
        cv2.imwrite(str(current_filename), current.crop)

        pending = PendingEvidenceEvent(
            event_id=event_id,
            output_dir=output_dir,
            remaining_post_frames=self.post_frames,
        )
        state.pending_events.append(pending)

        message = f"EVIDENCE_SAVED track_id={track_id} event_id={event_id} dir={output_dir}"
        print(message)
        self.last_saved_messages.append(message)
        self.last_saved_messages = self.last_saved_messages[-5:]

    def _save_pending_post_crops(self, track_id: int, state: TrackEvidenceState) -> None:
        if not state.pending_events or not state.recent_crops:
            return

        current = state.recent_crops[-1]
        remaining_events: List[PendingEvidenceEvent] = []
        for pending in state.pending_events:
            if pending.remaining_post_frames <= 0:
                continue

            filename = pending.output_dir / (
                f"post_{pending.saved_post_frames:02d}_frame_{current.frame_index:06d}_score_{current.score:.2f}.jpg"
            )
            cv2.imwrite(str(filename), current.crop)
            pending.saved_post_frames += 1
            pending.remaining_post_frames -= 1

            if pending.remaining_post_frames > 0:
                remaining_events.append(pending)
            else:
                print(
                    f"EVIDENCE_COMPLETE track_id={track_id} "
                    f"event_id={pending.event_id} dir={pending.output_dir}"
                )

        state.pending_events = remaining_events


def draw_evidence_status(frame: np.ndarray, collector: EvidenceCollector) -> None:
    if not collector.last_saved_messages:
        return

    y = 70
    for message in collector.last_saved_messages[-3:]:
        short_text = message.split(" dir=")[0]
        cv2.putText(
            frame,
            short_text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 24


def main() -> None:
    args = build_argparser().parse_args()

    detector = ScrfdInsightFaceDetector(
        model_path=args.model,
        input_size=(args.input_width, args.input_height),
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    tracker = SimpleIoUTracker(
        iou_threshold=args.iou_threshold,
        max_missed=args.max_missed,
    )
    collector = EvidenceCollector(
        evidence_dir=args.evidence_dir,
        pre_frames=args.pre_frames,
        post_frames=args.post_frames,
    )

    device = dai.Device()
    platform = device.getPlatform().name
    print(f"Device: {device.getDeviceId()} Platform: {platform}")

    entrance_states: Dict[int, EntranceState] = {}
    frame_index = 0

    with dai.Pipeline(device) as pipeline:
        print("Step 7: host-side recognition evidence capture on top of SCRFD entrance events.")

        camera = pipeline.create(dai.node.Camera).build()
        camera_out = camera.requestOutput(
            size=(PREVIEW_WIDTH, PREVIEW_HEIGHT),
            type=dai.ImgFrame.Type.BGR888p,
            fps=args.fps,
        )
        queue = camera_out.createOutputQueue(maxSize=4, blocking=False)

        print("Pipeline created. Starting...")
        pipeline.start()

        while pipeline.isRunning():
            frame_index += 1
            msg = queue.get()
            frame = msg.getCvFrame()

            tracks = tracker.update(detector.detect(frame))
            collector.update_buffers(
                frame=frame,
                tracks=tracks,
                frame_index=frame_index,
                crop_margin=args.crop_margin,
            )

            entered_track_ids = process_entrance_logic(
                tracks=tracks,
                states=entrance_states,
                axis=args.line_axis,
                line_position=args.line_position,
                frame_shape=frame.shape[:2],
                outside_side=args.outside_side,
                min_history=args.min_history,
                debug_entrance=args.debug_entrance,
            )

            for track_id in entered_track_ids:
                print(f"ENTRY_EVENT track_id={track_id}")
                collector.record_entry_event(track_id)

            draw_tracks(frame, tracks)
            draw_entrance_line(
                frame,
                axis=args.line_axis,
                line_position=args.line_position,
                outside_side=args.outside_side,
            )
            draw_entry_events(frame, entered_track_ids)
            draw_evidence_status(frame, collector)

            cv2.imshow("OAK Host SCRFD Recognition Evidence", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
