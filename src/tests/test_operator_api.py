import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from pipeline.mjpeg_stream_server import MjpegStreamServer


class OperatorApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.shop_open_calls = 0

        def open_shop() -> dict[str, object]:
            self.shop_open_calls += 1
            return {
                "shopId": 1,
                "customerId": "TEST-CUSTOMER",
                "shopEnteredAt": "2026-08-11T17:00:00",
            }

        self.server = MjpegStreamServer(
            camera_device_ids=["camera-a"],
            camera_roles=["observer"],
            operator_state_db=root / "state.sqlite",
            operator_runs_root=root / "runs",
            operator_api_token="secret",
            shop_opener=open_shop,
            shop_shelf_sync_provider=lambda visit_id: {
                "visitId": visit_id,
                "status": "synced",
                "local": {"shelfId": 3, "distanceMm": 640},
                "cloud": {"shelfId": 3, "distanceMm": 640},
                "pendingCount": 0,
            },
        )

    def tearDown(self) -> None:
        self.server.stop()
        self.temporary.cleanup()

    def route(self, path: str):
        routes = list(self.server.app.routes)
        for route in list(routes):
            original_router = getattr(route, "original_router", None)
            if original_router is not None:
                routes.extend(original_router.routes)
        return next(
            route.endpoint
            for route in routes
            if getattr(route, "path", None) == path
        )

    def test_state_and_authenticated_run_lifecycle(self) -> None:
        state = self.route("/operator/api/state")()
        self.assertEqual(state["cameras"][0]["deviceId"], "camera-a")

        payload = {
            "scenario": "route",
            "verifier": "Milan",
            "subjects": [{"subjectId": "milan", "displayName": "Milan"}],
        }
        with self.assertRaises(HTTPException) as unauthorized:
            self.route("/operator/api/test-runs")(payload, None)
        self.assertEqual(unauthorized.exception.status_code, 401)

        started = self.route("/operator/api/test-runs")(
            payload,
            "Bearer secret",
        )
        run_id = started["runId"]

        annotation = self.route(
            "/operator/api/test-runs/{run_id}/annotations"
        )(
            run_id,
            {"annotationType": "physical_entry", "subjectId": "milan"},
            "Bearer secret",
        )
        self.assertEqual(
            annotation["annotation"]["annotationType"],
            "physical_entry",
        )
        with self.assertRaises(HTTPException) as unauthorized_open:
            self.route("/operator/api/shop/open")(None)
        self.assertEqual(unauthorized_open.exception.status_code, 401)
        opened = self.route("/operator/api/shop/open")("Bearer secret")
        self.assertEqual(opened["customerId"], "TEST-CUSTOMER")
        self.assertEqual(self.shop_open_calls, 1)

        shelf_sync = self.route(
            "/operator/api/shop-shelf-sync/{visit_id}"
        )(12)
        self.assertEqual(shelf_sync["visitId"], 12)
        self.assertEqual(shelf_sync["status"], "synced")
        self.assertEqual(shelf_sync["cloud"]["shelfId"], 3)

    def test_console_assets_are_local(self) -> None:
        page = self.route("/operator/")()
        self.assertTrue(Path(page.path).exists())
        self.assertIn("Shop Walk Test", Path(page.path).read_text(encoding="utf-8"))
        self.assertEqual(page.headers["cache-control"], "no-store")
        script = self.route("/operator/assets/{asset_name}")("operator.js")
        self.assertTrue(Path(script.path).exists())
        self.assertEqual(script.headers["cache-control"], "no-store")
        script_source = Path(script.path).read_text(encoding="utf-8")
        self.assertIn("renderObservationHitboxes", script_source)
        self.assertIn("renderObservationChoices", script_source)
        self.assertIn("annotateEventFeedback", script_source)
        self.assertIn("MONITORED_EVENT_TYPES", script_source)
        self.assertIn("bindObservationSelectionControl", script_source)
        self.assertIn("autoMapSingleEntranceCandidate", script_source)
        self.assertIn("automatic_single_entrance_candidate", script_source)
        self.assertIn("manual_override", script_source)
        self.assertIn('addEventListener("touchend"', script_source)
        self.assertIn("{passive: false}", script_source)
        self.assertIn('button.addEventListener("click"', script_source)
        self.assertIn('addEventListener("contextmenu"', script_source)
        stylesheet = self.route("/operator/assets/{asset_name}")("operator.css")
        stylesheet_source = Path(stylesheet.path).read_text(encoding="utf-8")
        self.assertIn("-webkit-touch-callout: none", stylesheet_source)
        self.assertIn("touch-action: manipulation", stylesheet_source)
        self.assertIn(".person-hitbox", stylesheet_source)
        self.assertIn(".observation-choice", stylesheet_source)
        self.assertIn(".event-card", stylesheet_source)
        page_source = Path(page.path).read_text(encoding="utf-8")
        self.assertNotIn('name="scenario"', page_source)
        self.assertNotIn('name="verifier"', page_source)
        self.assertNotIn('name="subjectId"', page_source)
        self.assertNotIn('name="displayName"', page_source)
        self.assertNotIn('name="expectedCustomerId"', page_source)
        self.assertNotIn('name="notes"', page_source)
        self.assertIn('name="token"', page_source)
        self.assertIn('scenario: "shop-walk"', script_source)
        self.assertIn('subjectId: "milan"', script_source)
        self.assertIn('elements.token.value = app.token', script_source)
        self.assertNotIn("Physical reality", page_source)
        self.assertNotIn("Missing ENTRY", page_source)
        self.assertNotIn("Missing LEAVE", page_source)
        self.assertNotIn("Physical shelf position", page_source)
        self.assertNotIn("record-shelf-position", script_source)
        self.assertNotIn("missing-shelf", script_source)
        self.assertNotIn("Check claim", page_source)
        self.assertNotIn("Physical value when wrong", page_source)
        self.assertNotIn("Current claim is correct", page_source)
        self.assertNotIn("Record correction", page_source)
        self.assertNotIn("annotateWorldClaim", script_source)
        self.assertNotIn("parsePhysicalValue", script_source)
        self.assertNotIn("analyze-run", page_source)
        self.assertNotIn("analyze-run", script_source)
        self.assertIn('["Shelf position"', script_source)
        self.assertIn("Shelf distance evidence", page_source)
        self.assertNotIn("Missing shelf APPROACH", page_source)
        self.assertIn("shelfMeasurements", script_source)
        self.assertIn("System belief", page_source)
        self.assertIn("Generated event diagnostics", page_source)
        self.assertNotIn("voice", page_source.lower())
        self.assertIn("Open shop", page_source)
        self.assertIn("/operator/api/shop/open", script_source)
        self.assertNotIn("/operator/voice/", script_source)
        self.assertNotIn("disconnectVoice", script_source)
        self.assertIn("visual-testing-2", page_source)

    def test_shop_leave_persistence_result_is_published_to_operator_events(self) -> None:
        self.server.publish_shop_api_leave_result(
            event_type="shop_leave_persisted",
            camera_index=0,
            device_id="camera-a",
            track_id=2,
            visit_id=12,
            host_synced_seconds=10.0,
            payload={
                "shopId": 1,
                "customerId": "customer-a",
                "visitId": 12,
                "shopLeftAt": "2026-08-12T12:00:00Z",
            },
        )

        events, resync = self.server.operator_state.events_after(0)
        self.assertFalse(resync)
        event = events[-1]
        self.assertEqual(event.event_type, "shop_leave_persisted")
        self.assertEqual(event.visit_id, 12)
        self.assertEqual(event.payload["customerId"], "customer-a")

    def test_shop_entry_binding_result_is_published_to_operator_events(self) -> None:
        self.server.publish_shop_api_entry_result(
            event_type="shop_entry_bound",
            camera_index=0,
            device_id="camera-a",
            track_id=2,
            visit_id=12,
            host_synced_seconds=10.0,
            payload={
                "shopId": 1,
                "customerId": "customer-a",
                "visitId": 12,
                "reason": None,
            },
        )

        events, resync = self.server.operator_state.events_after(0)
        self.assertFalse(resync)
        event = events[-1]
        self.assertEqual(event.event_type, "shop_entry_bound")
        self.assertEqual(event.visit_id, 12)
        self.assertEqual(event.payload["customerId"], "customer-a")

    def test_mutations_are_read_only_without_configured_token(self) -> None:
        self.server.stop()
        root = Path(self.temporary.name)
        self.server = MjpegStreamServer(
            camera_device_ids=["camera-a"],
            camera_roles=["observer"],
            operator_state_db=root / "read-only.sqlite",
            operator_runs_root=root / "read-only-runs",
        )
        with self.assertRaises(HTTPException) as error:
            self.route("/operator/api/test-runs")(
                {
                    "scenario": "route",
                    "verifier": "Milan",
                    "subjects": [
                        {"subjectId": "milan", "displayName": "Milan"}
                    ],
                },
                None,
            )
        self.assertEqual(error.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
