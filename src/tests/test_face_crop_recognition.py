import unittest
from types import SimpleNamespace

import numpy as np

from pipeline.face_identity import InsightFaceFaceRecognizer
from pipeline.tracking import Track


class FakeAnalyzer:
    def __init__(self) -> None:
        self.crop_shapes = []

    def get(self, crop):
        self.crop_shapes.append(crop.shape)
        return [
            SimpleNamespace(
                det_score=0.9,
                bbox=np.array([10.0, 20.0, 80.0, 90.0]),
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            )
        ]


class FaceCropRecognitionTests(unittest.TestCase):
    def test_face_crop_is_extracted_and_bbox_is_remapped_to_full_frame(self) -> None:
        recognizer = InsightFaceFaceRecognizer.__new__(InsightFaceFaceRecognizer)
        recognizer.analyzer = FakeAnalyzer()
        recognizer.match_threshold = 0.68
        recognizer.min_det_score = 0.45
        recognizer.identities = []
        frame = np.zeros((400, 500, 3), dtype=np.uint8)
        track = Track(
            track_id=7,
            x1=100,
            y1=50,
            x2=300,
            y2=350,
            score=0.8,
            status="TRACKED",
        )

        faces = recognizer.recognize_crops(frame, tracks=[track])

        self.assertEqual(recognizer.analyzer.crop_shapes, [(300, 200, 3)])
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].track_id, 7)
        self.assertEqual(faces[0].bbox, (110, 70, 180, 140))
        self.assertEqual(faces[0].identity_id, "face_person_001")


if __name__ == "__main__":
    unittest.main()
