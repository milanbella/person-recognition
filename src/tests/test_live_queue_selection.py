import unittest
from collections import deque
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from live_synced_rgbd_streams import (
    LiveDepthPacket,
    LiveRgbTrackSnapshot,
    drain_latest_depth_message_for_snapshots,
    drain_latest_message,
    drain_messages_into_buffer,
    pop_latest_matching_rgb_pair,
    pop_matching_rgb_track_snapshot,
    rgb_depth_pair_is_synchronized,
    select_preview_frame,
    update_cached_depth_overlay,
)
from pipeline.performance import LivePerformanceLogger


class FakeQueue:
    def __init__(self, messages):
        self.messages = list(messages)

    def tryGet(self):
        if not self.messages:
            return None
        return self.messages.pop(0)


class FakeDepthMessage:
    def __init__(self, *, sequence_num, host_seconds, frame_value):
        self.sequence_num = sequence_num
        self.host_seconds = host_seconds
        self.frame = np.full((2, 3), frame_value, dtype=np.uint16)
        self.get_frame_calls = 0

    def getSequenceNum(self):
        return self.sequence_num

    def getTimestamp(self):
        return timedelta(seconds=self.host_seconds)

    def getTimestampDevice(self):
        return timedelta(seconds=self.host_seconds - 0.1)

    def getFrame(self):
        self.get_frame_calls += 1
        return self.frame


class LiveQueueSelectionTests(unittest.TestCase):
    def test_drain_latest_message_discards_stale_messages(self) -> None:
        messages = [object(), object(), object()]
        queue = FakeQueue(messages)

        selected = drain_latest_message(queue)

        self.assertIs(selected, messages[-1])
        self.assertEqual(queue.messages, [])

    def test_messages_are_drained_into_bounded_processing_buffer(self) -> None:
        messages = [
            FakeDepthMessage(sequence_num=1, host_seconds=9.8, frame_value=1),
            FakeDepthMessage(sequence_num=2, host_seconds=10.02, frame_value=2),
            FakeDepthMessage(sequence_num=3, host_seconds=10.1, frame_value=3),
        ]
        buffered_messages = deque(maxlen=2)

        drained = drain_messages_into_buffer(
            FakeQueue(messages),
            buffered_messages,
        )

        self.assertEqual(drained, 3)
        self.assertEqual(
            [message.sequence_num for message in buffered_messages],
            [2, 3],
        )

    def test_current_rgb_outputs_pair_on_latest_common_sequence(self) -> None:
        raw_messages = deque(
            [
                FakeDepthMessage(sequence_num=100, host_seconds=10.0, frame_value=1),
                FakeDepthMessage(sequence_num=101, host_seconds=10.033, frame_value=2),
            ]
        )
        processing_messages = deque(
            [
                FakeDepthMessage(sequence_num=100, host_seconds=10.0, frame_value=3),
                FakeDepthMessage(sequence_num=101, host_seconds=10.033, frame_value=4),
            ]
        )

        selected = pop_latest_matching_rgb_pair(raw_messages, processing_messages)

        self.assertIsNotNone(selected)
        raw_message, processing_message = selected
        self.assertEqual(raw_message.sequence_num, 101)
        self.assertEqual(processing_message.sequence_num, 101)
        self.assertEqual(len(raw_messages), 0)
        self.assertEqual(len(processing_messages), 0)

    def test_depth_attaches_to_exact_track_snapshot_sequence(self) -> None:
        snapshots = deque(
            [
                LiveRgbTrackSnapshot(
                    sequence_num=200,
                    host_synced_seconds=20.0,
                    processing_frame=np.zeros((2, 3, 3), dtype=np.uint8),
                    processing_tracks=(),
                    display_tracks=(),
                    recognized_faces=(),
                    body_evidence_by_track={},
                ),
                LiveRgbTrackSnapshot(
                    sequence_num=201,
                    host_synced_seconds=20.033,
                    processing_frame=np.zeros((2, 3, 3), dtype=np.uint8),
                    processing_tracks=(),
                    display_tracks=(),
                    recognized_faces=(),
                    body_evidence_by_track={},
                ),
            ]
        )

        selected, delta_ms = pop_matching_rgb_track_snapshot(
            snapshots,
            depth_sequence_num=201,
            depth_host_synced_seconds=20.034,
            max_delta_ms=250.0,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.sequence_num, 201)
        self.assertAlmostEqual(delta_ms, 1.0)
        self.assertEqual(len(snapshots), 0)

    def test_depth_falls_back_to_nearby_snapshot_capture_timestamp(self) -> None:
        snapshots = deque(
            [
                LiveRgbTrackSnapshot(
                    sequence_num=300,
                    host_synced_seconds=30.0,
                    processing_frame=np.zeros((2, 3, 3), dtype=np.uint8),
                    processing_tracks=(),
                    display_tracks=(),
                    recognized_faces=(),
                    body_evidence_by_track={},
                )
            ]
        )

        selected, delta_ms = pop_matching_rgb_track_snapshot(
            snapshots,
            depth_sequence_num=301,
            depth_host_synced_seconds=30.033,
            max_delta_ms=250.0,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.sequence_num, 300)
        self.assertAlmostEqual(delta_ms, 33.0)
        self.assertEqual(len(snapshots), 0)

    def test_depth_rejects_snapshot_outside_capture_timestamp_limit(self) -> None:
        snapshots = deque(
            [
                LiveRgbTrackSnapshot(
                    sequence_num=300,
                    host_synced_seconds=30.0,
                    processing_frame=np.zeros((2, 3, 3), dtype=np.uint8),
                    processing_tracks=(),
                    display_tracks=(),
                    recognized_faces=(),
                    body_evidence_by_track={},
                )
            ]
        )

        selected, delta_ms = pop_matching_rgb_track_snapshot(
            snapshots,
            depth_sequence_num=301,
            depth_host_synced_seconds=31.0,
            max_delta_ms=250.0,
        )

        self.assertIsNone(selected)
        self.assertAlmostEqual(delta_ms, 1000.0)
        self.assertEqual(len(snapshots), 1)

    def test_depth_drain_selects_exact_pending_snapshot_without_copying_frames(self) -> None:
        snapshots = deque(
            [
                LiveRgbTrackSnapshot(
                    sequence_num=401,
                    host_synced_seconds=40.033,
                    processing_frame=np.zeros((2, 3, 3), dtype=np.uint8),
                    processing_tracks=(),
                    display_tracks=(),
                    recognized_faces=(),
                    body_evidence_by_track={},
                )
            ]
        )
        messages = [
            FakeDepthMessage(sequence_num=400, host_seconds=40.0, frame_value=1),
            FakeDepthMessage(sequence_num=401, host_seconds=40.033, frame_value=2),
            FakeDepthMessage(sequence_num=402, host_seconds=40.066, frame_value=3),
        ]

        selected, latest = drain_latest_depth_message_for_snapshots(
            FakeQueue(messages),
            snapshots,
            max_delta_ms=250.0,
        )

        self.assertEqual(selected.sequence_num, 401)
        self.assertEqual(latest.sequence_num, 402)
        self.assertEqual([message.get_frame_calls for message in messages], [0, 0, 0])

    def test_depth_drain_matches_independent_sequence_by_capture_timestamp(self) -> None:
        snapshots = deque(
            [
                LiveRgbTrackSnapshot(
                    sequence_num=900,
                    host_synced_seconds=50.0,
                    processing_frame=np.zeros((2, 3, 3), dtype=np.uint8),
                    processing_tracks=(),
                    display_tracks=(),
                    recognized_faces=(),
                    body_evidence_by_track={},
                )
            ]
        )
        messages = [
            FakeDepthMessage(sequence_num=100, host_seconds=49.7, frame_value=1),
            FakeDepthMessage(sequence_num=101, host_seconds=50.01, frame_value=2),
            FakeDepthMessage(sequence_num=102, host_seconds=50.4, frame_value=3),
        ]

        selected, latest = drain_latest_depth_message_for_snapshots(
            FakeQueue(messages),
            snapshots,
            max_delta_ms=250.0,
        )

        self.assertEqual(selected.sequence_num, 101)
        self.assertEqual(latest.sequence_num, 102)

    def test_hidden_depth_window_skips_colorization_and_releases_overlay(self) -> None:
        state = SimpleNamespace(cached_depth_overlay=np.ones((2, 3, 3), dtype=np.uint8))

        with patch("live_synced_rgbd_streams.colorize_depth") as colorize_depth:
            update_cached_depth_overlay(
                state=state,
                depth_frame_mm=np.ones((2, 3), dtype=np.uint16),
                hide_depth_window=True,
                performance=LivePerformanceLogger(enabled=False),
            )

        colorize_depth.assert_not_called()
        self.assertIsNone(state.cached_depth_overlay)

    def test_rgb_depth_pair_rejects_stale_depth(self) -> None:
        depth_packet = LiveDepthPacket(
            sequence_num=10,
            host_synced_seconds=7.5,
            device_monotonic_seconds=7.4,
            frame_mm=np.zeros((2, 3), dtype=np.uint16),
        )

        self.assertFalse(
            rgb_depth_pair_is_synchronized(
                depth_packet,
                rgb_host_synced_seconds=10.0,
                max_delta_ms=250.0,
            )
        )
        self.assertTrue(
            rgb_depth_pair_is_synchronized(
                depth_packet,
                rgb_host_synced_seconds=7.7,
                max_delta_ms=250.0,
            )
        )

    def test_preview_defaults_to_current_raw_frame(self) -> None:
        raw_frame = np.zeros((2, 3, 3), dtype=np.uint8)
        overlay = np.ones((2, 3, 3), dtype=np.uint8)
        state = SimpleNamespace(
            cached_rgb_raw=raw_frame,
            cached_rgb_overlay=overlay,
        )

        self.assertIs(
            select_preview_frame(state, show_annotated_preview=False),
            raw_frame,
        )
        self.assertIs(
            select_preview_frame(state, show_annotated_preview=True),
            overlay,
        )


if __name__ == "__main__":
    unittest.main()
