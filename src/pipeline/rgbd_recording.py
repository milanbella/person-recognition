from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


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


def build_recording_dir(output_root: Path, device_id: str) -> Path:
    return output_root / f"oak_{device_id}.rgbd"


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
