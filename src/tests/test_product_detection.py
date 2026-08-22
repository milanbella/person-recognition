import unittest

import numpy as np

from pipeline.product_detection import (
    ProductDetection,
    decode_product_crop,
    decode_end_to_end_product_output,
    encode_lossless_product_crop,
    expanded_person_crop,
    parse_yolo_class_names,
    translate_product_detection,
)


class ProductDetectionTests(unittest.TestCase):
    def test_lossless_frozen_crop_round_trip_preserves_every_pixel(self) -> None:
        crop = np.random.default_rng(7).integers(
            0,
            256,
            size=(73, 119, 3),
            dtype=np.uint8,
        )

        encoded = encode_lossless_product_crop(crop)
        restored = decode_product_crop(encoded)

        self.assertTrue(encoded.startswith(b"\x89PNG\r\n\x1a\n"))
        np.testing.assert_array_equal(restored, crop)

    def test_expands_and_clips_person_crop(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        crop, crop_box, person_box = expanded_person_crop(
            frame,
            (10, 20, 110, 80),
            margin_fraction=0.25,
        )

        self.assertEqual(crop_box, (0, 5, 135, 95))
        self.assertEqual(person_box, (10, 15, 110, 75))
        self.assertEqual(crop.shape, (90, 135, 3))

    def test_parses_ultralytics_class_names(self) -> None:
        names = parse_yolo_class_names(
            {"names": "{0: 'cola', 1: 'rice'}"}
        )

        self.assertEqual(names, {0: "cola", 1: "rice"})

    def test_translates_crop_detection_to_source_frame(self) -> None:
        detection = ProductDetection(5, 10, 45, 70, 0.8, 3, "juice")

        translated = translate_product_detection(
            detection,
            (100, 200, 500, 800),
        )

        self.assertEqual(
            translated,
            ProductDetection(105, 210, 145, 270, 0.8, 3, "juice"),
        )

    def test_decodes_selected_batch_and_filters_score(self) -> None:
        output = np.zeros((3, 3, 6), dtype=np.float32)
        output[0, 0] = [10, 20, 110, 220, 0.9, 1]
        output[0, 1] = [20, 30, 40, 50, 0.1, 0]
        output[1, 0] = [1, 2, 3, 4, 0.95, 0]

        boxes, scores, class_ids = decode_end_to_end_product_output(
            output,
            class_names={0: "cola", 1: "rice"},
            score_threshold=0.2,
        )

        np.testing.assert_array_equal(boxes, [[10, 20, 110, 220]])
        np.testing.assert_allclose(scores, [0.9])
        np.testing.assert_array_equal(class_ids, [1])


if __name__ == "__main__":
    unittest.main()
