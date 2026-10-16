"""Measure raw RGB sequence gaps without detection, tracking, or image decoding."""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass

import depthai as dai

from pipeline.camera import (
    configure_live_device,
    device_identifier,
    format_usb_connection,
    list_available_devices,
)
from pipeline.camera_stats import CameraStats
from pipeline.mjpeg_stream_server import MjpegStreamServer


@dataclass
class CameraQueues:
    device: dai.Device
    pipeline: dai.Pipeline
    raw: dai.MessageQueue
    processing: dai.MessageQueue | None
    depth: dai.MessageQueue | None
    telemetry: dai.MessageQueue


class CameraUnavailableError(RuntimeError):
    pass


class DiagnosticCameraStats(CameraStats):
    def payload(self, camera_index: int) -> dict[str, object]:
        payload = super().payload(camera_index)
        # The existing dashboard formats these values as numbers even when
        # inference is deliberately absent or a camera has not connected.
        for key in ("temperature", "cpuLoad", "memoryLoad", "frameAgeMs", "exposureUs", "inferenceMs"):
            if payload[key] is None:
                payload[key] = 0.0
        return payload


def connect(device_id: str) -> dai.Device:
    for attempt in range(1, 6):
        available = {device_identifier(info) for info in list_available_devices()}
        if device_id in available:
            try:
                device = dai.Device(device_id)
                configure_live_device(device)
                return device
            except Exception as error:
                print(f"CAMERA_CONNECT_RETRY device_id={device_id} attempt={attempt}/5 error={error}", flush=True)
        else:
            print(f"CAMERA_CONNECT_RETRY device_id={device_id} attempt={attempt}/5 not_found", flush=True)
        if attempt < 5:
            time.sleep(2)
    raise CameraUnavailableError(f"Could not connect to camera {device_id} after five attempts")


def open_camera(stack: ExitStack, device_id: str, args: argparse.Namespace) -> CameraQueues:
    device = connect(device_id)
    stack.callback(device.close)
    pipeline = stack.enter_context(dai.Pipeline(device))
    color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    if args.profile == "checkout-like":
        raw = color.requestOutput(
            size=(1280, 720),
            type=dai.ImgFrame.Type.BGR888p,
            resizeMode=dai.ImgResizeMode.LETTERBOX,
            fps=args.fps,
        ).createOutputQueue()
    else:
        raw = color.requestOutput(
            size=(args.frame_width, args.frame_height),
            type=dai.ImgFrame.Type.BGR888p,
            fps=args.fps,
        ).createOutputQueue(maxSize=8, blocking=False)

    processing = None
    depth = None
    if args.profile in ("full", "rgb-only"):
        processing = color.requestOutput(
            size=(args.processing_width, args.processing_height),
            type=dai.ImgFrame.Type.BGR888p,
            fps=args.fps,
        ).createOutputQueue(maxSize=8, blocking=False)
    elif args.extra_rgb:
        processing = color.requestOutput(
            size=(1920, 1080),
            type=dai.ImgFrame.Type.BGR888p,
            fps=args.fps,
        ).createOutputQueue(maxSize=8, blocking=False)
    if args.profile == "full" or (args.profile == "checkout-like" and not args.no_depth):
        if args.profile == "full" and not args.no_still_output:
            still = color.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
            color.inputControl.createInputQueue()
            still.createOutputQueue(maxSize=1, blocking=False)
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.ROBOTICS
            if args.profile == "checkout-like"
            else dai.node.StereoDepth.PresetMode.DEFAULT
        )
        if args.profile == "full":
            stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
            stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        if args.profile == "checkout-like":
            stereo.setOutputSize(640, 400)
            for camera, input_node in ((left, stereo.left), (right, stereo.right)):
                camera.requestOutput(
                    size=(640, 400),
                    type=dai.ImgFrame.Type.GRAY8,
                    fps=args.fps,
                ).link(input_node)
            depth = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
        else:
            stereo.setOutputSize(args.processing_width, args.processing_height)
            left.requestFullResolutionOutput(fps=args.fps).link(stereo.left)
            right.requestFullResolutionOutput(fps=args.fps).link(stereo.right)
            depth = stereo.depth.createOutputQueue(
                maxSize=max(8, math.ceil(args.fps * 6.0)), blocking=False,
            )

    logger = pipeline.create(dai.node.SystemLogger)
    logger.setRate(0.5)
    telemetry = logger.out.createOutputQueue(maxSize=1, blocking=False)
    pipeline.start()
    print(f"CAMERA_STARTED device_id={device_id} {format_usb_connection(device)}", flush=True)
    return CameraQueues(device, pipeline, raw, processing, depth, telemetry)


def read_queue(queue: dai.MessageQueue | None, mode: str, on_message=None) -> int:
    if queue is None:
        return 0
    count = 0
    while True:
        message = queue.tryGet()
        if message is None:
            break
        if on_message is not None:
            on_message(message)
        count += 1
        if mode == "one":
            break
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", nargs="+", required=True)
    parser.add_argument("--profile", choices=("full", "rgb-only", "raw-only", "checkout-like"), default="full")
    parser.add_argument("--queue-mode", choices=("drain", "one"), default="drain")
    parser.add_argument("--no-still-output", action="store_true")
    parser.add_argument("--no-depth", action="store_true", help="Omit stereo depth from checkout-like")
    parser.add_argument("--extra-rgb", action="store_true", help="Add a 1920x1080 RGB output to checkout-like")
    parser.add_argument("--skip-unavailable", action="store_true")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--frame-height", type=int, default=1080)
    parser.add_argument("--processing-width", type=int, default=1280)
    parser.add_argument("--processing-height", type=int, default=720)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--report-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.fps is None:
        args.fps = 15 if args.profile == "checkout-like" else 30
    if len(set(args.device_id)) != len(args.device_id):
        raise ValueError("Camera device IDs must be unique")
    if args.extra_rgb and args.profile != "checkout-like":
        raise ValueError("--extra-rgb requires --profile checkout-like")
    if args.no_depth and args.profile != "checkout-like":
        raise ValueError("--no-depth requires --profile checkout-like")
    if args.fps <= 0 or args.report_seconds <= 0 or args.duration_seconds < 0:
        raise ValueError("FPS and report interval must be positive; duration cannot be negative")
    for value in (args.frame_width, args.frame_height, args.processing_width, args.processing_height):
        if value <= 0:
            raise ValueError("Frame dimensions must be positive")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    server = MjpegStreamServer(
        camera_device_ids=list(args.device_id), host=args.host, port=args.port,
        enable_mjpeg_streaming=False,
    )
    server.camera_stats = {
        index: DiagnosticCameraStats(device_id)
        for index, device_id in enumerate(args.device_id)
    }
    with ExitStack() as stack:
        server.start()
        stack.callback(server.stop)
        cameras: list[CameraQueues | None] = []
        for index, device_id in enumerate(args.device_id):
            if stop.is_set():
                return
            try:
                cameras.append(open_camera(stack, device_id, args))
            except CameraUnavailableError as error:
                if not args.skip_unavailable:
                    raise
                print(f"CAMERA_UNAVAILABLE camera={index + 1} device_id={device_id} error={error}", flush=True)
                cameras.append(None)
            if index + 1 < len(args.device_id) and stop.wait(1.0):
                return
        if not any(camera is not None for camera in cameras):
            raise RuntimeError("No cameras connected")

        print(
            f"DROP_TEST_READY profile={args.profile} queue_mode={args.queue_mode} "
            f"fps={args.fps} "
            f"cameras={sum(camera is not None for camera in cameras)}/{len(cameras)} "
            f"extra_rgb={args.extra_rgb} "
            f"depth={args.profile == 'full' or (args.profile == 'checkout-like' and not args.no_depth)} "
            f"still_output={args.profile == 'full' and not args.no_still_output}",
            flush=True,
        )
        started = time.monotonic()
        last_report = started
        previous = [server.camera_stats[i].payload(i) for i in range(len(cameras))]
        while not stop.is_set():
            for index, camera in enumerate(cameras):
                if camera is None:
                    continue
                if camera.device.isClosed() or not camera.pipeline.isRunning():
                    raise RuntimeError(f"Camera {index + 1} pipeline stopped")
                stats = server.camera_stats[index]
                read_queue(camera.raw, args.queue_mode, stats.record_rgb_packet)
                read_queue(camera.processing, args.queue_mode)
                read_queue(camera.depth, args.queue_mode)
                read_queue(camera.telemetry, "drain", stats.record_telemetry)

            now = time.monotonic()
            if now - last_report >= args.report_seconds:
                elapsed = now - last_report
                for index, camera in enumerate(cameras):
                    if camera is None:
                        print(f"DROP_TEST camera={index + 1} device_id={args.device_id[index]} status=unavailable", flush=True)
                        continue
                    current = server.camera_stats[index].payload(index)
                    old = previous[index]
                    print(
                        f"DROP_TEST camera={index + 1} device_id={args.device_id[index]} "
                        f"profile={args.profile} queue_mode={args.queue_mode} "
                        f"received_fps={(current['frameCount'] - old['frameCount']) / elapsed:.2f} "
                        f"missing_fps={(current['droppedFramesTotal'] - old['droppedFramesTotal']) / elapsed:.2f} "
                        f"gaps={current['sequenceGapsTotal'] - old['sequenceGapsTotal']} "
                        f"frame_age_ms={current['frameAgeMs']}",
                        flush=True,
                    )
                    previous[index] = current
                last_report = now
            if args.duration_seconds and now - started >= args.duration_seconds:
                break
            stop.wait(0.001)


if __name__ == "__main__":
    main()
