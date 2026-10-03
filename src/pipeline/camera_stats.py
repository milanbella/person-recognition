from __future__ import annotations

import threading
import time
from datetime import datetime, timezone


def _iso(timestamp: float | None) -> str | None:
    return None if timestamp is None else datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


class CameraStats:
    """Checkout-compatible counters measured when raw RGB packets reach Python."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self._lock = threading.Lock()
        self._frame_time: float | None = None
        self._frame_unix: float | None = None
        self._sequence: int | None = None
        self._count = 0
        self._dropped = 0
        self._gaps = 0
        self._fps = 0.0
        self._detections = 0
        self._inference_ms: float | None = None
        self._inference_unix: float | None = None
        self._exposure_us: float | None = None
        self._telemetry_time: float | None = None
        self._telemetry: dict[str, float | None] = {}

    def record_telemetry(self, message: object) -> None:
        ddr = message.ddrMemoryUsage
        with self._lock:
            self._telemetry = {
                "temperature": round(float(message.chipTemperature.average), 1),
                "cpuLoad": round(float(message.leonCssCpuUsage.average) * 100, 1),
                "memoryLoad": round(float(ddr.used) / ddr.total * 100, 1) if ddr.total else None,
            }
            self._telemetry_time = time.monotonic()

    def record_received_frame(self, *, sequence: int, exposure_us: float | None) -> None:
        now = time.monotonic()
        unix = time.time()
        with self._lock:
            if sequence == self._sequence:
                return
            if self._sequence is not None and sequence > self._sequence + 1:
                self._dropped += sequence - self._sequence - 1
                self._gaps += 1
            self._frame_time = now
            self._frame_unix = unix
            self._sequence = sequence
            self._count += 1
            self._exposure_us = exposure_us

    def record_rgb_packet(self, message: object) -> None:
        try:
            exposure_us = message.getExposureTime().total_seconds() * 1_000_000
        except Exception:
            exposure_us = None
        self.record_received_frame(sequence=int(message.getSequenceNum()), exposure_us=exposure_us)

    def record_inference(self, *, detections: int, inference_ms: float) -> None:
        with self._lock:
            self._detections = detections
            self._inference_ms = round(inference_ms, 2)
            self._inference_unix = time.time()

    def set_processing_fps(self, fps: float) -> None:
        with self._lock:
            self._fps = fps

    @property
    def last_received_monotonic(self) -> float | None:
        with self._lock:
            return self._frame_time

    def payload(self, camera_index: int) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            age = None if self._frame_time is None else max(0, now - self._frame_time) * 1000
            status = "offline" if age is None or age >= 5000 else "stale" if age >= 1000 else "active"
            fresh_telemetry = self._telemetry_time is not None and now - self._telemetry_time < 10
            return {
                "camera": camera_index,
                "timestamp": _iso(time.time()),
                **{key: self._telemetry.get(key) if fresh_telemetry else None
                   for key in ("temperature", "cpuLoad", "memoryLoad")},
                "status": status,
                "checkoutActive": False,
                "deviceId": self.device_id,
                "frameAgeMs": None if age is None else round(age, 1),
                "frameCount": self._count,
                "sequenceNumber": self._sequence,
                "droppedFrames": self._dropped,
                "sequenceGaps": self._gaps,
                "exposureUs": self._exposure_us,
                "detections": self._detections,
                "processingFps": round(self._fps, 2),
                "inferenceMs": self._inference_ms,
                "lastFrameTimestamp": _iso(self._frame_unix),
                "lastDetectionTimestamp": _iso(self._inference_unix),
                "lastInferenceTimestamp": _iso(self._inference_unix),
            }
