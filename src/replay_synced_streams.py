import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline.recording import ReplayStream, load_recording, resolve_timestamps_path


def render_stream(stream: ReplayStream, label: str, target_host_seconds: float) -> np.ndarray:
    if stream.current_frame is None or stream.current_stamp is None:
        frame = np.zeros((stream.info.height, stream.info.width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"{label}: no frame",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    frame = stream.current_frame.copy()
    delta_ms = (stream.current_stamp.host_synced_seconds - target_host_seconds) * 1000.0
    lines = [
        f"{label} device={stream.info.device_id}",
        f"frame_index={stream.current_stamp.frame_index} seq={stream.current_stamp.sequence_num}",
        f"host_synced={stream.current_stamp.host_synced_seconds:.3f}s",
        f"delta_to_target={delta_ms:+.1f} ms",
        f"received_utc={stream.current_stamp.received_utc}",
    ]
    y = 32
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay two recorded OAK streams side-by-side using recorded timestamps."
    )
    parser.add_argument("--video-a", type=Path, required=True, help="First recorded .avi file.")
    parser.add_argument("--video-b", type=Path, required=True, help="Second recorded .avi file.")
    parser.add_argument(
        "--timestamps-a",
        type=Path,
        default=None,
        help="Optional first .timestamps.jsonl file. Defaults to the video sidecar path.",
    )
    parser.add_argument(
        "--timestamps-b",
        type=Path,
        default=None,
        help="Optional second .timestamps.jsonl file. Defaults to the video sidecar path.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier. 1.0 is real-time based on recorded timestamps.",
    )
    parser.add_argument(
        "--start-mode",
        choices=["overlap", "full"],
        default="overlap",
        help="Start at the overlapping interval or include pre-overlap lead-in with blank/older frames.",
    )
    return parser.parse_args()

def fit_to_common_height(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_height = min(left.shape[0], right.shape[0])

    def resize(frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] == target_height:
            return frame
        scale = target_height / frame.shape[0]
        target_width = int(round(frame.shape[1] * scale))
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    return resize(left), resize(right)


def main() -> None:
    args = parse_args()
    info_a = load_recording(args.video_a, resolve_timestamps_path(args.video_a, args.timestamps_a))
    info_b = load_recording(args.video_b, resolve_timestamps_path(args.video_b, args.timestamps_b))

    stream_a = ReplayStream(info_a)
    stream_b = ReplayStream(info_b)

    try:
        start_a = info_a.frames[0].host_synced_seconds
        start_b = info_b.frames[0].host_synced_seconds
        end_a = info_a.frames[-1].host_synced_seconds
        end_b = info_b.frames[-1].host_synced_seconds

        if args.start_mode == "overlap":
            replay_start = max(start_a, start_b)
            replay_end = min(end_a, end_b)
        else:
            replay_start = min(start_a, start_b)
            replay_end = max(end_a, end_b)

        if replay_end <= replay_start:
            raise RuntimeError("The recordings do not have an overlapping replay interval.")

        stream_a.advance_until(replay_start)
        stream_b.advance_until(replay_start)

        print(f"Replay start host_synced_seconds={replay_start:.3f}")
        print(f"Replay end host_synced_seconds={replay_end:.3f}")
        print("Controls: q=quit, space=pause/resume, ]=faster, [=slower")

        paused = False
        speed = args.speed
        started_monotonic = time.monotonic()
        paused_elapsed = 0.0

        while True:
            if paused:
                target_time = replay_start + paused_elapsed * speed
            else:
                elapsed = time.monotonic() - started_monotonic
                paused_elapsed = elapsed
                target_time = replay_start + elapsed * speed

            if target_time > replay_end:
                print("Replay complete.")
                break

            stream_a.advance_until(target_time)
            stream_b.advance_until(target_time)

            left = render_stream(stream_a, "Camera A", target_time)
            right = render_stream(stream_b, "Camera B", target_time)
            left, right = fit_to_common_height(left, right)
            combined = np.hstack([left, right])

            cv2.putText(
                combined,
                f"target_host_synced={target_time:.3f}s speed={speed:.2f}x",
                (20, combined.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Synchronized Replay", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
                if not paused:
                    started_monotonic = time.monotonic() - paused_elapsed
            if key == ord("]"):
                speed = min(speed * 1.25, 16.0)
                started_monotonic = time.monotonic() - paused_elapsed
            if key == ord("["):
                speed = max(speed / 1.25, 0.1)
                started_monotonic = time.monotonic() - paused_elapsed
    finally:
        stream_a.close()
        stream_b.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
