import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException

from model_training.api import ModelTrainingService
from model_training.capture import CaptureRegistrar
from model_training.exporter import YoloDatasetExporter
from model_training.store import ModelTrainingStore


class _LiveClient:
    def __init__(self, capture: dict[str, object] | None = None) -> None:
        self.capture_payload = capture

    def cameras(self):
        return [{"id": 0, "deviceId": "camera-a", "status": "active"}]

    def capture(self, camera_index: int):
        assert camera_index == 0
        assert self.capture_payload is not None
        return dict(self.capture_payload)


class _CatalogClient:
    def products(self):
        return [{"id": 7, "code": "oil-1l", "name": "Cooking oil 1L", "barCode": "123"}]


class ModelTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = ModelTrainingStore(self.root / "model_training.sqlite")
        self.store.replace_products(1, _CatalogClient().products())
        self.session = self.store.create_session(
            shop_id=1,
            product_code="oil-1l",
            scenario="hand",
            camera_index=0,
            device_id="camera-a",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def source_capture(self) -> dict[str, object]:
        spool = self.root / "spool"
        spool.mkdir(exist_ok=True)
        image_path = spool / "capture.jpg"
        metadata_path = spool / "capture.json"
        image = np.zeros((2160, 3840, 3), dtype=np.uint8)
        image[300:1600, 1000:2200] = (40, 120, 220)
        self.assertTrue(cv2.imwrite(str(image_path), image))
        metadata_path.write_text(json.dumps({"cameraIndex": 0}), encoding="utf-8")
        return {
            "cameraIndex": 0,
            "cameraNumber": 1,
            "deviceId": "camera-a",
            "rgbSequenceNumber": 42,
            "capturedAtUnixMilliseconds": 123456,
            "width": 3840,
            "height": 2160,
            "imagePath": str(image_path),
            "metadataPath": str(metadata_path),
        }

    def test_capture_is_owned_and_proxy_is_mobile_sized(self) -> None:
        source = self.source_capture()
        frame = CaptureRegistrar(self.root / "state", self.store).register(
            self.session, source, capture_request_id="request-1"
        )
        owned = self.store.frame_image_path(frame["frameId"], "original")
        proxy = cv2.imread(str(self.store.frame_image_path(frame["frameId"], "review")))
        self.assertEqual(proxy.shape[:2], (720, 1280))

        Path(str(source["imagePath"])).unlink()
        Path(str(source["metadataPath"])).unlink()
        self.assertTrue(owned.is_file())
        self.assertEqual(frame["width"], 3840)
        self.assertEqual(len(frame["perceptualHash"]), 16)

    def test_annotations_export_to_normalized_yolo_label(self) -> None:
        frame = CaptureRegistrar(self.root / "state", self.store).register(
            self.session, self.source_capture(), capture_request_id="request-2"
        )
        self.store.save_annotations(
            frame["frameId"],
            [{"x1": 0.25, "y1": 0.2, "x2": 0.75, "y2": 0.8}],
        )
        self.store.finalize_frame(frame["frameId"], "accepted")

        dataset = YoloDatasetExporter(self.root / "state", self.store).export()
        label = Path(dataset["path"]) / "labels" / "train" / f"{frame['frameId']}.txt"
        self.assertEqual(label.read_text(encoding="ascii"), "0 0.50000000 0.50000000 0.50000000 0.60000000\n")
        self.assertIn("train-only", dataset["splitWarning"])

    def test_not_visible_is_exported_as_empty_label(self) -> None:
        frame = CaptureRegistrar(self.root / "state", self.store).register(
            self.session, self.source_capture(), capture_request_id="request-3"
        )
        self.store.finalize_frame(frame["frameId"], "not_visible")
        dataset = YoloDatasetExporter(self.root / "state", self.store).export()
        label = Path(dataset["path"]) / "labels" / "train" / f"{frame['frameId']}.txt"
        self.assertEqual(label.read_text(encoding="ascii"), "")

    def test_api_requires_auth_and_refreshes_products(self) -> None:
        service = ModelTrainingService(
            state_root=self.root / "api-state",
            api_token="secret",
            shop_id=1,
            live_client=_LiveClient(),
            catalog_client=_CatalogClient(),
            assets_root=Path(__file__).resolve().parent.parent / "model_training_ui",
        )
        app_routes = list(service.app().routes)
        for route in list(app_routes):
            original_router = getattr(route, "original_router", None)
            if original_router is not None:
                app_routes.extend(original_router.routes)
        routes = {
            getattr(route, "path", ""): route.endpoint
            for route in app_routes
            if hasattr(route, "endpoint")
        }
        with self.assertRaises(HTTPException) as unauthorized:
            routes["/model-training/api/products/refresh"](None)
        self.assertEqual(unauthorized.exception.status_code, 401)
        refreshed = routes["/model-training/api/products/refresh"]("Bearer secret")
        self.assertEqual(refreshed["products"][0]["code"], "oil-1l")
        cameras = routes["/model-training/api/cameras"]()
        self.assertEqual(cameras["cameras"][0]["cameraNumber"], 1)



if __name__ == "__main__":
    unittest.main()
