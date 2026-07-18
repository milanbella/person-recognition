import unittest

import numpy as np

from pipeline.body_evidence import BodyEvidence, scale_body_evidence_heights
from pipeline.depth import DepthSample, scale_depth_samples
from pipeline.tracking import Track, scale_tracks
from pipeline.visit_identity import BodyAppearance


class CoordinateScalingTests(unittest.TestCase):
    def test_tracks_scale_from_processing_to_display_resolution(self) -> None:
        track = Track(
            track_id=7,
            x1=10,
            y1=20,
            x2=110,
            y2=220,
            score=0.9,
            hits=3,
            missed_frames=1,
            status="TRACKED",
            history=[(60.0, 120.0)],
        )

        scaled = scale_tracks(
            [track],
            source_width=640,
            source_height=360,
            target_width=3840,
            target_height=2160,
        )[0]

        self.assertEqual((scaled.x1, scaled.y1, scaled.x2, scaled.y2), (60, 120, 660, 1320))
        self.assertEqual(scaled.history, [(360.0, 720.0)])
        self.assertEqual((scaled.track_id, scaled.status, scaled.hits), (7, "TRACKED", 3))

    def test_depth_sample_pixels_scale_without_changing_3d_evidence(self) -> None:
        sample = DepthSample(
            depth_mm=1750.0,
            valid_pixel_count=40,
            roi=(10, 20, 110, 220),
            anchor_px=(60, 120),
            point_3d_mm=(100.0, 200.0, 1750.0),
        )

        scaled = scale_depth_samples(
            {7: sample},
            source_width=640,
            source_height=360,
            target_width=3840,
            target_height=2160,
        )[7]

        self.assertEqual(scaled.roi, (60, 120, 660, 1320))
        self.assertEqual(scaled.anchor_px, (360, 720))
        self.assertEqual(scaled.point_3d_mm, sample.point_3d_mm)
        self.assertEqual(scaled.depth_mm, sample.depth_mm)

    def test_body_height_metadata_scales_to_display_coordinates(self) -> None:
        evidence = BodyEvidence(
            track_id=7,
            appearance=BodyAppearance(
                histogram=np.array([0.25, 0.75], dtype=np.float32),
                aspect_ratio=0.4,
                height_px=200,
            ),
        )

        scaled = scale_body_evidence_heights({7: evidence}, scale_y=6.0)[7]

        self.assertEqual(scaled.appearance.height_px, 1200)
        self.assertEqual(scaled.appearance.aspect_ratio, 0.4)
        np.testing.assert_array_equal(
            scaled.appearance.histogram,
            evidence.appearance.histogram,
        )


if __name__ == "__main__":
    unittest.main()
