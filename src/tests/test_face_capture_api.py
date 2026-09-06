import unittest

from fastapi import HTTPException, Response

from pipeline.face_capture import FaceCaptureCoordinator
from pipeline.face_capture_api import create_face_capture_router
from pipeline.visit_registry import VISIT_ORIGIN_ENTRANCE, VISIT_STATUS_ACTIVE


class FaceCaptureApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinator = FaceCaptureCoordinator(camera_device_ids=["camera-a"])
        self.coordinator.update_visit_state(
            {1: (VISIT_ORIGIN_ENTRANCE, "customer-1", VISIT_STATUS_ACTIVE)}
        )
        self.router = create_face_capture_router(
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

    def test_routes_require_auth_and_operator_token_is_accepted(self) -> None:
        create = self.route("/face-captures", "POST")
        with self.assertRaises(HTTPException) as unauthorized:
            create(
                response=Response(),
                payload={"visitId": 1, "purpose": "operator_test"},
                authorization=None,
                idempotency_key=None,
            )
        self.assertEqual(unauthorized.exception.status_code, 401)

        response = Response()
        payload = create(
            response=response,
            payload={"visitId": 1, "purpose": "operator_test"},
            authorization="Bearer operator-secret",
            idempotency_key="operator-request",
        )
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(payload["visitId"], 1)
        status = self.route("/face-captures/{capture_id}", "GET")
        self.assertEqual(
            status(
                payload["captureId"],
                response=Response(),
                authorization="Bearer face-secret",
            )["captureId"],
            payload["captureId"],
        )

    def test_image_is_conflict_until_ready_and_delete_is_idempotent(self) -> None:
        create = self.route("/face-captures", "POST")
        payload = create(
            response=Response(),
            payload={"customerId": "customer-1"},
            authorization="Bearer face-secret",
            idempotency_key=None,
        )
        image = self.route("/face-captures/{capture_id}/image", "GET")
        with self.assertRaises(HTTPException) as pending:
            image(payload["captureId"], authorization="Bearer face-secret")
        self.assertEqual(pending.exception.status_code, 409)

        delete = self.route("/face-captures/{capture_id}", "DELETE")
        self.assertEqual(
            delete(
                payload["captureId"],
                response=Response(),
                authorization="Bearer face-secret",
            )["status"],
            "deleted",
        )
        self.assertEqual(
            delete(
                payload["captureId"],
                response=Response(),
                authorization="Bearer face-secret",
            )["status"],
            "deleted",
        )


if __name__ == "__main__":
    unittest.main()
