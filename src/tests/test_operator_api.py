import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from pipeline.mjpeg_stream_server import MjpegStreamServer


class OperatorApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.server = MjpegStreamServer(
            camera_device_ids=["camera-a"],
            camera_roles=["observer"],
            operator_state_db=root / "state.sqlite",
            operator_runs_root=root / "runs",
            operator_api_token="secret",
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
        context = self.route(
            "/operator/api/test-runs/{run_id}/voice-context"
        )(run_id)
        self.assertEqual(context["run"]["runId"], run_id)
        self.assertEqual(context["subjects"][0]["subjectId"], "milan")
        self.assertIn("events", context)
        self.assertEqual(context["verdicts"], {})

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
        self.assertIn("Physical reality", page_source)
        self.assertIn("Shelf position", page_source)
        self.assertIn("Shelf distance evidence", page_source)
        self.assertNotIn("Missing shelf APPROACH", page_source)
        self.assertIn("shelfMeasurements", script_source)
        self.assertIn("System belief", page_source)
        self.assertIn("Generated event diagnostics", page_source)
        self.assertIn("Start voice test", page_source)
        self.assertIn("/operator/voice/sessions", script_source)

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
