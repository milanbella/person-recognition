import unittest
import base64
from unittest.mock import patch

from fastapi import HTTPException, Response

from pipeline.person_photo import PersonPhotoCapture, PersonPhotoError
from pipeline.person_photo_api import create_person_photo_router
from pipeline.visit_registry import VISIT_ORIGIN_ENTRANCE, VISIT_STATUS_ACTIVE


class PersonPhotoApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinator = PersonPhotoCapture(camera_device_ids=["camera-a"])
        self.coordinator.update_visit_state(
            {1: (VISIT_ORIGIN_ENTRANCE, "customer-1", VISIT_STATUS_ACTIVE)}
        )
        self.router = create_person_photo_router(
            coordinator=self.coordinator,
            api_token="face-secret",
            operator_api_token="operator-secret",
        )

    def route(self, path: str, method: str):
        return next(
            route.endpoint
            for route in self.router.routes
            if route.path == path and method in route.methods
        )

    def test_person_json_contains_image_and_evidence_and_requires_auth(self) -> None:
        endpoint = self.route("/cameras/{camera_index}/person", "GET")
        with patch.object(self.coordinator, "capture_person_evidence") as capture:
            with self.assertRaises(HTTPException) as error:
                endpoint(0, response=Response(), authorization=None)
            self.assertEqual(error.exception.status_code, 401)
            capture.assert_not_called()
            capture.return_value = (b"jpeg-bytes", {"visitId": 7, "customerId": None, "photoPath": "/photos/test.jpg"})
            response = Response()
            result = endpoint(0, response=response, track_id=4, authorization="Bearer operator-secret")
            capture.assert_called_once_with(0, 4)
            self.assertEqual(result["visitId"], 7)
            self.assertEqual(result["photoPath"], "/photos/test.jpg")
            self.assertIsNone(result["customerId"])
            self.assertEqual(base64.b64decode(result["image"]["base64"]), b"jpeg-bytes")
            self.assertEqual(response.headers["cache-control"], "no-store")

    def test_jpeg_response_includes_saved_path(self):
        endpoint = self.route("/cameras/{camera_index}/person.jpg", "GET")
        with patch.object(self.coordinator, "capture_person_evidence", return_value=(
            b"jpeg", {"trackId": 4, "rgbSequenceNumber": 99, "photoPath": "/photos/test.jpg"},
        )):
            result = endpoint(0, authorization="Bearer face-secret")
        self.assertEqual(result.body, b"jpeg")
        self.assertEqual(result.headers["x-photo-path"], "/photos/test.jpg")

    def test_storage_failure_returns_500(self):
        for path in ("/cameras/{camera_index}/person", "/cameras/{camera_index}/person.jpg"):
            with self.subTest(path=path), patch.object(
                self.coordinator, "capture_person_evidence",
                side_effect=PersonPhotoError("CAPTURE_STORAGE_FAILED", "Failed to save"),
            ):
                kwargs = {"response": Response()} if path.endswith("/person") else {}
                with self.assertRaises(HTTPException) as raised:
                    self.route(path, "GET")(0, authorization="Bearer face-secret", **kwargs)
                self.assertEqual(raised.exception.status_code, 500)
