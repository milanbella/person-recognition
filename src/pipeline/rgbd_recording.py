from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

DEFAULT_RGBD_RECORDINGS_DIR = Path(__file__).resolve().parents[1] / "recordings"
DEFAULT_PLANE_CALIBRATIONS_DIR = Path(__file__).resolve().parents[1] / "plane_calibrations"


@dataclass
class RGBDRecordedFrame:
    frame_index: int
    rgb_sequence_num: int
    rgb_host_synced_seconds: float
    rgb_device_monotonic_seconds: float
    depth_sequence_num: int
    depth_host_synced_seconds: float
    depth_device_monotonic_seconds: float
    matched_depth_delta_ms: float
    received_utc: str
    depth_png_relpath: str


@dataclass
class RGBDRecordingInfo:
    recording_dir: Path
    rgb_video_path: Path
    frames_path: Path
    depth_frames_dir: Path
    device_id: str
    fps: int
    width: int
    height: int
    rgb_intrinsics: dict[str, float] | None
    frames: List[RGBDRecordedFrame]


class RGBDReplayStream:
    def __init__(self, info: RGBDRecordingInfo) -> None:
        self.info = info
        self.capture = cv2.VideoCapture(str(info.rgb_video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open RGB video file: {info.rgb_video_path}")

        self.current_index = -1
        self.current_rgb_frame: np.ndarray | None = None
        self.current_depth_frame: np.ndarray | None = None
        self.current_frame_meta: RGBDRecordedFrame | None = None
        self.next_index = 0
        self.advance()

    def advance(self) -> bool:
        ok, frame = self.capture.read()
        if not ok or self.next_index >= len(self.info.frames):
            self.current_rgb_frame = None
            self.current_depth_frame = None
            self.current_frame_meta = None
            return False

        frame_meta = self.info.frames[self.next_index]
        self.current_rgb_frame = frame
        self.current_depth_frame = load_depth_png(self.info, frame_meta)
        self.current_frame_meta = frame_meta
        self.current_index = self.next_index
        self.next_index += 1
        return True

    def advance_until(self, target_host_seconds: float) -> None:
        while self.next_index < len(self.info.frames):
            next_meta = self.info.frames[self.next_index]
            if next_meta.rgb_host_synced_seconds > target_host_seconds:
                break
            if not self.advance():
                break

    def reset(self) -> None:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_index = -1
        self.current_rgb_frame = None
        self.current_depth_frame = None
        self.current_frame_meta = None
        self.next_index = 0
        self.advance()

    def close(self) -> None:
        self.capture.release()


def build_recording_dir(output_root: Path, device_id: str) -> Path:
    return output_root / f"oak_{device_id}.rgbd"


def build_plane_calibration_path(calibrations_root: Path, device_id: str) -> Path:
    calibrations_root.mkdir(parents=True, exist_ok=True)
    return calibrations_root / f"plane_fit_{device_id}.json"


def add_rgbd_recording_lookup_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--device-id",
        type=str,
        default=None,
        help="OAK device id/MXID used to derive the default RGBD recording folder.",
    )
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=None,
        help="Optional explicit RGBD recording directory override.",
    )
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=DEFAULT_RGBD_RECORDINGS_DIR,
        help="Root directory containing RGBD recording folders named oak_<device-id>.rgbd.",
    )
    return parser


def resolve_recording_dir(
    *,
    recording_dir: Path | None,
    device_id: str | None,
    recordings_root: Path,
) -> Path:
    if recording_dir is not None:
        return recording_dir
    if device_id is None:
        raise ValueError("Either --device-id or --recording-dir is required.")
    return build_recording_dir(recordings_root, device_id)


def build_recording_paths(recording_dir: Path) -> Dict[str, Path]:
    return {
        "recording_dir": recording_dir,
        "rgb_video_path": recording_dir / "rgb.avi",
        "frames_path": recording_dir / "frames.jsonl",
        "depth_frames_dir": recording_dir / "depth_frames",
    }


def load_rgbd_recording(recording_dir: Path) -> RGBDRecordingInfo:
    paths = build_recording_paths(recording_dir)
    rgb_video_path = paths["rgb_video_path"]
    frames_path = paths["frames_path"]
    depth_frames_dir = paths["depth_frames_dir"]

    if not recording_dir.exists():
        raise FileNotFoundError(f"Recording directory not found: {recording_dir}")
    if not rgb_video_path.exists():
        raise FileNotFoundError(f"RGB video not found: {rgb_video_path}")
    if not frames_path.exists():
        raise FileNotFoundError(f"Frame metadata not found: {frames_path}")
    if not depth_frames_dir.exists():
        raise FileNotFoundError(f"Depth frame directory not found: {depth_frames_dir}")

    header: Dict[str, Any] | None = None
    frames: List[RGBDRecordedFrame] = []
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("type") == "recording_header":
            header = payload
            continue
        if payload.get("type") != "frame":
            continue

        frames.append(
            RGBDRecordedFrame(
                frame_index=int(payload["frame_index"]),
                rgb_sequence_num=int(payload["rgb_sequence_num"]),
                rgb_host_synced_seconds=float(payload["rgb_host_synced_seconds"]),
                rgb_device_monotonic_seconds=float(payload["rgb_device_monotonic_seconds"]),
                depth_sequence_num=int(payload["depth_sequence_num"]),
                depth_host_synced_seconds=float(payload["depth_host_synced_seconds"]),
                depth_device_monotonic_seconds=float(payload["depth_device_monotonic_seconds"]),
                matched_depth_delta_ms=float(payload["matched_depth_delta_ms"]),
                received_utc=str(payload["received_utc"]),
                depth_png_relpath=str(payload["depth_png_relpath"]),
            )
        )

    if header is None:
        raise RuntimeError(f"No recording header found in {frames_path}")
    if not frames:
        raise RuntimeError(f"No RGBD frame records found in {frames_path}")

    return RGBDRecordingInfo(
        recording_dir=recording_dir,
        rgb_video_path=rgb_video_path,
        frames_path=frames_path,
        depth_frames_dir=depth_frames_dir,
        device_id=str(header["device_id"]),
        fps=int(header["fps"]),
        width=int(header["width"]),
        height=int(header["height"]),
        rgb_intrinsics=header.get("rgb_intrinsics"),
        frames=frames,
    )


def load_depth_png(recording: RGBDRecordingInfo, frame: RGBDRecordedFrame) -> np.ndarray:
    path = recording.recording_dir / frame.depth_png_relpath
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth PNG: {path}")
    if depth.dtype != np.uint16:
        raise RuntimeError(f"Expected uint16 depth PNG at {path}, got {depth.dtype}")
    return depth
