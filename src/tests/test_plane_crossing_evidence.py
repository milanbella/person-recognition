import unittest

import numpy as np

from pipeline.plane_crossing_evidence import (
    PlaneEvidenceFrame,
    render_plane_crossing_contact_sheet,
    select_plane_evidence_frames,
)
from pipeline.tracking import Track


class PlaneCrossingEvidenceTests(unittest.TestCase):
    def frame(self, sequence: int) -> PlaneEvidenceFrame:
        return PlaneEvidenceFrame(
            sequence_num=sequence,
            host_synced_seconds=sequence / 10.0,
            frame=np.zeros((180, 320, 3), dtype=np.uint8),
            tracks=(Track(7, 40 + sequence, 30, 140 + sequence, 170, 0.9),),
        )

    def test_selects_bounded_frames_around_crossing(self) -> None:
        selected = select_plane_evidence_frames(
            [self.frame(sequence) for sequence in range(10, 20)],
            crossing_sequence_num=15,
            frame_count=5,
        )

        self.assertEqual([frame.sequence_num for frame in selected], [13, 14, 15, 16, 17])

    def test_renders_track_and_crossing_roi_with_metadata(self) -> None:
        selected = tuple(self.frame(sequence) for sequence in (14, 15, 16))

        sheet, metadata = render_plane_crossing_contact_sheet(
            selected,
            crossing_sequence_num=15,
            track_id=7,
            visit_id=3,
            event_type="entry",
            plane_signed_distance_mm=-25.0,
            depth_mm=4311.0,
            depth_roi=(70, 100, 110, 150),
            thumbnail_width=320,
        )

        self.assertEqual(sheet.shape, (180, 960, 3))
        self.assertEqual([item["isCrossingFrame"] for item in metadata], [False, True, False])
        self.assertEqual(metadata[1]["trackBoundingBox"], (55, 30, 155, 170))
        self.assertGreater(int(sheet.sum()), 0)


if __name__ == "__main__":
    unittest.main()
