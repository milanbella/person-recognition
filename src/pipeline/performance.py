from __future__ import annotations

import time
from dataclasses import dataclass, field


PROFILE_STAGE_ORDER = (
    "camera_iteration",
    "depth_drain",
    "rgb_poll",
    "rgb_decode",
    "yolo",
    "tracking",
    "shelf_marker",
    "depth_logic",
    "shelf_depth",
    "shelf_coordinator",
    "face",
    "body",
    "registry_io",
    "overlay",
    "depth_colorize",
    "stream_publish",
    "gui",
    "cycle",
)

PROFILE_METRIC_ORDER = (
    "raw_rgb_capture_age_ms",
    "processing_rgb_capture_age_ms",
    "rgb_pair_delta_ms",
    "preview_age_ms",
    "raw_rgb_queue_drained",
    "processing_rgb_queue_drained",
)


@dataclass
class LivePerformanceLogger:
    enabled: bool
    interval_seconds: float = 5.0
    window_started: float = field(default_factory=time.perf_counter)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    metric_totals: dict[str, float] = field(default_factory=dict)
    metric_counts: dict[str, int] = field(default_factory=dict)
    metric_maxima: dict[str, float] = field(default_factory=dict)
    cycle_count: int = 0
    camera_poll_count: int = 0
    rgb_frame_count: int = 0
    processed_frame_count: int = 0
    stream_frame_count: int = 0
    detection_count: int = 0
    track_count: int = 0

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0.0:
            raise ValueError("Performance logging interval must be greater than zero.")

    def start(self) -> float:
        return time.perf_counter() if self.enabled else 0.0

    def record_duration(self, stage: str, started: float) -> None:
        if not self.enabled:
            return
        self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + (
            time.perf_counter() - started
        )
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1

    def record_metric(self, metric: str, value: float) -> None:
        if not self.enabled:
            return
        self.metric_totals[metric] = self.metric_totals.get(metric, 0.0) + value
        self.metric_counts[metric] = self.metric_counts.get(metric, 0) + 1
        self.metric_maxima[metric] = max(
            value,
            self.metric_maxima.get(metric, value),
        )

    def record_camera_poll(self) -> None:
        if self.enabled:
            self.camera_poll_count += 1

    def record_rgb_frame(self) -> None:
        if self.enabled:
            self.rgb_frame_count += 1

    def record_processed_frame(self, *, detection_count: int, track_count: int) -> None:
        if not self.enabled:
            return
        self.processed_frame_count += 1
        self.detection_count += detection_count
        self.track_count += track_count

    def record_stream_frame(self) -> None:
        if self.enabled:
            self.stream_frame_count += 1

    def complete_cycle(self, started: float) -> None:
        if not self.enabled:
            return
        self.record_duration("cycle", started)
        self.cycle_count += 1
        now = time.perf_counter()
        if now - self.window_started >= self.interval_seconds:
            self.report(now)

    def report(self, now: float | None = None) -> None:
        if not self.enabled:
            return
        report_time = time.perf_counter() if now is None else now
        elapsed = max(report_time - self.window_started, 1e-9)
        processed = max(self.processed_frame_count, 1)
        average_detections = self.detection_count / processed
        average_tracks = self.track_count / processed
        stage_text = " ".join(
            f"{stage}_ms={self._average_stage_ms(stage):.1f}"
            for stage in PROFILE_STAGE_ORDER
        )
        metric_text = " ".join(
            f"{metric}={self._average_metric(metric):.1f} "
            f"{metric}_max={self.metric_maxima.get(metric, 0.0):.1f}"
            for metric in PROFILE_METRIC_ORDER
        )
        print(
            f"LIVE_PERF window_s={elapsed:.1f} cycles={self.cycle_count} "
            f"camera_polls={self.camera_poll_count} rgb_frames={self.rgb_frame_count} "
            f"processed_frames={self.processed_frame_count} "
            f"processed_fps={self.processed_frame_count / elapsed:.1f} "
            f"stream_frames={self.stream_frame_count} "
            f"avg_detections={average_detections:.2f} avg_tracks={average_tracks:.2f} "
            f"{stage_text} {metric_text}",
            flush=True,
        )
        self.window_started = report_time
        self.stage_seconds.clear()
        self.stage_counts.clear()
        self.metric_totals.clear()
        self.metric_counts.clear()
        self.metric_maxima.clear()
        self.cycle_count = 0
        self.camera_poll_count = 0
        self.rgb_frame_count = 0
        self.processed_frame_count = 0
        self.stream_frame_count = 0
        self.detection_count = 0
        self.track_count = 0

    def _average_stage_ms(self, stage: str) -> float:
        count = self.stage_counts.get(stage, 0)
        if count == 0:
            return 0.0
        return self.stage_seconds.get(stage, 0.0) * 1000.0 / count

    def _average_metric(self, metric: str) -> float:
        count = self.metric_counts.get(metric, 0)
        if count == 0:
            return 0.0
        return self.metric_totals.get(metric, 0.0) / count
