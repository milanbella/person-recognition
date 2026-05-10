from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameStamp:
    frame_index: int
    sequence_num: int
    host_synced_seconds: float
    device_monotonic_seconds: float
    received_utc: str


@dataclass
class RecordingInfo:
    video_path: Path
    timestamps_path: Path
    device_id: str
    fps: int
    width: int
    height: int
    frames: list[FrameStamp]


def resolve_timestamps_path(video_path: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    return video_path.with_suffix(".timestamps.jsonl")


def load_recording(video_path: Path, timestamps_path: Path) -> RecordingInfo:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not timestamps_path.exists():
        raise FileNotFoundError(f"Timestamps file not found: {timestamps_path}")

    header: dict[str, object] | None = None
    frames: list[FrameStamp] = []
    for line in timestamps_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("type") == "recording_header":
            header = payload
            continue
        if payload.get("type") != "frame":
            continue
        frames.append(
            FrameStamp(
                frame_index=int(payload["frame_index"]),
                sequence_num=int(payload["sequence_num"]),
                host_synced_seconds=float(payload["host_synced_seconds"]),
                device_monotonic_seconds=float(payload["device_monotonic_seconds"]),
                received_utc=str(payload["received_utc"]),
            )
        )

    if header is None:
        raise RuntimeError(f"No recording header found in {timestamps_path}")
    if not frames:
        raise RuntimeError(f"No frame timestamps found in {timestamps_path}")

    return RecordingInfo(
        video_path=video_path,
        timestamps_path=timestamps_path,
        device_id=str(header["device_id"]),
        fps=int(header["fps"]),
        width=int(header["width"]),
        height=int(header["height"]),
        frames=frames,
    )


class ReplayStream:
    def __init__(self, info: RecordingInfo) -> None:
        self.info = info
        self.capture = cv2.VideoCapture(str(info.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video file: {info.video_path}")

        self.current_index = -1
        self.current_frame: np.ndarray | None = None
        self.current_stamp: FrameStamp | None = None
        self.next_index = 0
        self.advance()

    def advance(self) -> bool:
        ok, frame = self.capture.read()
        if not ok or self.next_index >= len(self.info.frames):
            self.current_frame = None
            self.current_stamp = None
            return False

        self.current_frame = frame
        self.current_stamp = self.info.frames[self.next_index]
        self.current_index = self.next_index
        self.next_index += 1
        return True

    def advance_until(self, target_host_seconds: float) -> None:
        while self.next_index < len(self.info.frames):
            next_stamp = self.info.frames[self.next_index]
            if next_stamp.host_synced_seconds > target_host_seconds:
                break
            if not self.advance():
                break

    def reset(self) -> None:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_index = -1
        self.current_frame = None
        self.current_stamp = None
        self.next_index = 0
        self.advance()

    def close(self) -> None:
        self.capture.release()
