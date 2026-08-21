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


def _route_endpoint(service: ModelTrainingService, path: str, method: str):
    for route in service.router().routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"Route not found: {method} {path}")


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

        ui_root = Path(__file__).resolve().parent.parent / "model_training_ui"
        page = (ui_root / "index.html").read_text(encoding="utf-8")
        script = (ui_root / "model-training.js").read_text(encoding="utf-8")
        self.assertIn('id="product-search"', page)
        self.assertIn("function renderProducts()", script)
        self.assertIn('item.barCode', script)
        self.assertIn("function restoreCameraSelection()", script)
        self.assertIn('localStorage.setItem("modelTrainingCameraIndex"', script)
        self.assertIn("refreshQueueState", script)
        self.assertIn("window.setInterval(refreshQueueState, 2000)", script)
        self.assertIn("function sessionMatchesSelection()", script)
        self.assertIn("await createSessionFromSelection();", script)
        self.assertIn('id="working-session"', page)
        self.assertIn('id="exported-session"', page)
        self.assertIn("function refreshSessionLists", script)
        self.assertIn("exportedOnly=true", script)
        self.assertGreater(page.index('class="panel preview-panel"'), page.index('class="panel setup-panel"'))
        preview_start = page.index('class="panel preview-panel"')
        review_start = page.index('class="panel review-panel"')
        self.assertIn('id="capture"', page[preview_start:review_start])

        stylesheet = (ui_root / "model-training.css").read_text(encoding="utf-8")
        self.assertIn("max-width: 100%", stylesheet)
        self.assertNotIn("object-fit: contain; touch-action", stylesheet)

    def test_review_queue_state_tracks_pending_frame_ids(self) -> None:
        frame = CaptureRegistrar(self.root / "state", self.store).register(
            self.session, self.source_capture(), capture_request_id="queue-state"
        )
        pending = self.store.frame_queue_state(status="needs_review")
        self.assertEqual(pending, {"count": 1, "frameIds": [frame["frameId"]]})

        self.store.save_annotations(
            frame["frameId"],
            [{"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}],
        )
        self.store.finalize_frame(frame["frameId"], "accepted")
        self.assertEqual(
            self.store.frame_queue_state(status="needs_review"),
            {"count": 0, "frameIds": []},
        )

    def test_sessions_track_pending_unexported_and_exported_frames(self) -> None:
        registrar = CaptureRegistrar(self.root / "state", self.store)
        exported_frame = registrar.register(
            self.session, self.source_capture(), capture_request_id="membership-exported"
        )
        self.store.save_annotations(
            exported_frame["frameId"],
            [{"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8}],
        )
        self.store.finalize_frame(exported_frame["frameId"], "accepted")
        YoloDatasetExporter(self.root / "state", self.store).export()

        pending_frame = registrar.register(
            self.session, self.source_capture(), capture_request_id="membership-pending"
        )
        summary = self.store.get_session(self.session["sessionId"])
        self.assertEqual(summary["frameCount"], 2)
        self.assertEqual(summary["pendingCount"], 1)
        self.assertEqual(summary["exportedCount"], 1)
        self.assertEqual(summary["unexportedCount"], 0)
        self.assertEqual(summary["datasetVersions"], ["dataset-v0001"])
        self.assertEqual(
            [item["sessionId"] for item in self.store.list_sessions(group="working")],
            [self.session["sessionId"]],
        )
        self.assertEqual(
            [item["sessionId"] for item in self.store.list_sessions(group="exported")],
            [self.session["sessionId"]],
        )
        self.assertEqual(
            [item["frameId"] for item in self.store.list_frames(
                session_id=self.session["sessionId"], exported_only=True
            )],
            [exported_frame["frameId"]],
        )
        self.assertEqual(
            self.store.frame_queue_state(
                status="needs_review", session_id=self.session["sessionId"]
            ),
            {"count": 1, "frameIds": [pending_frame["frameId"]]},
        )

        self.store.save_annotations(
            pending_frame["frameId"],
            [{"x1": 0.3, "y1": 0.3, "x2": 0.7, "y2": 0.7}],
        )
        self.store.finalize_frame(pending_frame["frameId"], "accepted")
        summary = self.store.get_session(self.session["sessionId"])
        self.assertEqual(summary["pendingCount"], 0)
        self.assertEqual(summary["unexportedCount"], 1)

    def test_existing_dataset_manifest_backfills_membership(self) -> None:
        frame = CaptureRegistrar(self.root / "state", self.store).register(
            self.session, self.source_capture(), capture_request_id="legacy-membership"
        )
        self.store.save_annotations(
            frame["frameId"],
            [{"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8}],
        )
        self.store.finalize_frame(frame["frameId"], "accepted")
        YoloDatasetExporter(self.root / "state", self.store).export()
        with self.store._connection() as connection:
            connection.execute("DELETE FROM mt_dataset_frames")

        reopened = ModelTrainingStore(self.root / "model_training.sqlite")
        summary = reopened.get_session(self.session["sessionId"])
        self.assertEqual(summary["exportedCount"], 1)
        self.assertEqual(summary["datasetVersions"], ["dataset-v0001"])

    def test_clear_current_session_removes_only_owned_session_files(self) -> None:
        state_root = self.root / "api-state"
        service = ModelTrainingService(
            state_root=state_root,
            api_token="secret",
            shop_id=1,
            live_client=_LiveClient(),
            catalog_client=_CatalogClient(),
            assets_root=Path(__file__).resolve().parent.parent / "model_training_ui",
        )
        service.store.replace_products(1, _CatalogClient().products())
        session = service.store.create_session(
            shop_id=1,
            product_code="oil-1l",
            scenario="hand",
            camera_index=0,
            device_id="camera-a",
        )
        source = self.source_capture()
        frame = service.registrar.register(session, source, capture_request_id="clear-current")
        owned_paths = [
            service.store.frame_image_path(frame["frameId"], variant)
            for variant in ("original", "review", "thumbnail")
        ]
        clear = _route_endpoint(service, "/model-training/api/sessions/active", "DELETE")

        with self.assertRaises(HTTPException) as invalid_confirmation:
            clear({"confirmation": "yes"}, "Bearer secret")
        self.assertEqual(invalid_confirmation.exception.status_code, 422)
        result = clear({"confirmation": "CLEAR CURRENT SESSION"}, "Bearer secret")

        self.assertEqual(result["frameCount"], 1)
        self.assertIsNone(service.store.active_session())
        self.assertEqual(service.store.list_frames(), [])
        self.assertTrue(Path(str(source["imagePath"])).is_file())
        self.assertTrue(all(not path.exists() for path in owned_paths))

    def test_clear_all_training_data_removes_datasets_and_resets_version(self) -> None:
        state_root = self.root / "api-state"
        service = ModelTrainingService(
            state_root=state_root,
            api_token="secret",
            shop_id=1,
            live_client=_LiveClient(),
            catalog_client=_CatalogClient(),
            assets_root=Path(__file__).resolve().parent.parent / "model_training_ui",
        )
        service.store.replace_products(1, _CatalogClient().products())

        def add_accepted_frame(request_id: str) -> None:
            session = service.store.create_session(
                shop_id=1,
                product_code="oil-1l",
                scenario="hand",
                camera_index=0,
                device_id="camera-a",
            )
            frame = service.registrar.register(session, self.source_capture(), capture_request_id=request_id)
            service.store.save_annotations(
                frame["frameId"],
                [{"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8}],
            )
            service.store.finalize_frame(frame["frameId"], "accepted")

        add_accepted_frame("before-clear")
        self.assertEqual(service.exporter.export()["datasetVersion"], "dataset-v0001")
        clear = _route_endpoint(service, "/model-training/api/training-data", "DELETE")

        with self.assertRaises(HTTPException) as unauthorized:
            clear({"confirmation": "CLEAR ALL TRAINING DATA"}, None)
        self.assertEqual(unauthorized.exception.status_code, 401)
        result = clear({"confirmation": "CLEAR ALL TRAINING DATA"}, "Bearer secret")

        self.assertEqual(result["sessions"], 1)
        self.assertEqual(result["frames"], 1)
        self.assertEqual(result["datasets"], 1)
        self.assertEqual(len(service.store.list_products()), 1)
        self.assertEqual(list((state_root / "datasets").iterdir()), [])
        self.assertTrue(service.registrar.captures_root.is_dir())

        add_accepted_frame("after-clear")
        self.assertEqual(service.exporter.export()["datasetVersion"], "dataset-v0001")



if __name__ == "__main__":
    unittest.main()
