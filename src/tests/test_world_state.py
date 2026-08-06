import tempfile
import time
import unittest
import sqlite3
from pathlib import Path

from fastapi import HTTPException

from pipeline.mjpeg_stream_server import MjpegStreamServer
from pipeline.observer_api import ObservedBody, ObservedPerson, ObserverCameraSnapshot
from pipeline.shelf_anchors import ShelfAnchor
from pipeline.shelf_api import ShelfCameraSnapshot
from pipeline.shelf_config import ShelfDefinition
from pipeline.shelf_proximity import ShelfCameraObservation, ShelfProximityStatus
from pipeline.world_state import WorldStateProjector
from pipeline.world_state_api import _visit_claims
from pipeline.world_state_store import WorldStateStore


def _person(track_id: int, visit_id: int, *, origin: str = "entrance_confirmed") -> ObservedPerson:
    return ObservedPerson(
        track_id=track_id,
        track_status="TRACKED",
        detection_score=0.88,
        visit_id=visit_id,
        visit_origin=origin,
        customer_id=None,
        customer_binding_status="pending",
        bounding_box=(10, 20, 100, 200),
        centroid=(55.0, 110.0),
        depth=None,
        face_identity_ids=(),
        body=ObservedBody(has_appearance=True, aspect_ratio=0.5, height_pixels=180),
        matched_score=0.7,
        match_state="matched",
    )


def _snapshot(sequence: int, *people: ObservedPerson) -> ObserverCameraSnapshot:
    return ObserverCameraSnapshot(
        camera_index=0,
        device_id="camera-a",
        camera_role="observer",
        rgb_sequence_number=sequence,
        host_synced_seconds=float(sequence),
        published_at_unix_milliseconds=time.time_ns() // 1_000_000,
        frame_width=640,
        frame_height=360,
        observations=tuple(people),
    )


def _shelf(shelf_id: int, marker_id: int) -> ShelfDefinition:
    return ShelfDefinition(
        shelf_id=shelf_id,
        label=f"Shelf {shelf_id}",
        marker_id=marker_id,
        approach_distance_mm=900,
        departure_distance_mm=1100,
        approach_dwell_milliseconds=500,
        departure_dwell_milliseconds=500,
        lost_visit_grace_milliseconds=1000,
        owner_switch_margin_mm=100,
        owner_switch_dwell_milliseconds=300,
    )


def _shelf_observation(
    shelf: ShelfDefinition,
    *,
    camera_index: int,
    visit_id: int,
    distance_mm: float,
    observed_at_ms: int,
) -> ShelfCameraObservation:
    device_id = f"camera-{camera_index}"
    anchor = ShelfAnchor(
        shelf_id=shelf.shelf_id,
        marker_id=shelf.marker_id,
        device_id=device_id,
        point_3d_mm=(0.0, 0.0, 3000.0),
        sample_count=20,
        rms_spread_mm=5.0,
        updated_at_unix_milliseconds=observed_at_ms,
        source="operator_calibrated",
    )
    return ShelfCameraObservation(
        shelf_id=shelf.shelf_id,
        shelf_label=shelf.label,
        marker_id=shelf.marker_id,
        camera_index=camera_index,
        device_id=device_id,
        track_id=2,
        visit_id=visit_id,
        visit_origin="entrance_confirmed",
        customer_id=None,
        distance_mm=distance_mm,
        person_point_3d_mm=(0.0, 0.0, 3000.0 - distance_mm),
        anchor=anchor,
        host_synced_seconds=observed_at_ms / 1000.0,
        observed_at_unix_milliseconds=observed_at_ms,
        rgb_sequence_number=observed_at_ms,
        depth_sequence_number=observed_at_ms,
        track_bounding_box=(100, 200, 300, 700),
        person_depth_mm=3000.0 - distance_mm,
        person_depth_valid_pixel_count=48,
        person_depth_roi=(180, 350, 220, 410),
        person_depth_anchor_px=(200, 380),
    )


def _shelf_snapshot(
    camera_index: int,
    shelves: tuple[ShelfDefinition, ...],
    observations: tuple[ShelfCameraObservation, ...],
    *,
    observed_at_ms: int,
) -> ShelfCameraSnapshot:
    return ShelfCameraSnapshot(
        camera_index=camera_index,
        device_id=f"camera-{camera_index}",
        camera_role="observer",
        rgb_sequence_number=observed_at_ms,
        depth_sequence_number=observed_at_ms,
        host_synced_seconds=observed_at_ms / 1000.0,
        published_at_unix_milliseconds=observed_at_ms,
        shelves=shelves,
        anchors_by_shelf={},
        observations=observations,
        states_by_shelf={},
    )


class WorldStateProjectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "state.sqlite"
        self.store = WorldStateStore(self.path, flush_interval_seconds=0.01)
        self.projector = WorldStateProjector(
            camera_device_ids=["camera-0", "camera-1"],
            camera_roles=["observer", "observer"],
            camera_timeout_seconds=0.05,
            store=self.store,
            track_aging_seconds=0.02,
            track_stale_seconds=0.04,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def test_track_split_keeps_one_visit_and_ignores_older_frame(self) -> None:
        self.projector.mark_camera_frame(0, rgb_sequence_number=1)
        self.projector.publish_observer_snapshot(_snapshot(1, _person(4, 7)))
        self.projector.publish_observer_snapshot(_snapshot(2, _person(5, 7)))
        self.projector.publish_observer_snapshot(_snapshot(1, _person(99, 7)))

        state = self.projector.snapshot()
        self.assertEqual(state["occupancy"]["insideVisitCount"], 0)
        self.assertEqual(len(state["visits"]), 1)
        tracks = state["visits"][0]["currentTracks"]
        self.assertEqual([item["trackId"] for item in tracks], [5])

    def test_entry_leave_and_restart_preserve_lifecycle_but_stale_ephemeral_state(self) -> None:
        self.projector.publish_visit_state(
            9,
            status="inside",
            origin="entrance_confirmed",
            event_type="entry_accepted",
        )
        first_revision = self.projector.revision
        self.store.flush()
        self.store.close()

        self.store = WorldStateStore(self.path, flush_interval_seconds=0.01)
        self.projector = WorldStateProjector(
            camera_device_ids=["camera-a"],
            camera_roles=["observer"],
            camera_timeout_seconds=0.05,
            store=self.store,
        )
        restored = self.projector.snapshot()["visits"][0]
        self.assertGreater(self.projector.revision, first_revision)
        self.assertEqual(restored["status"], "inside")
        self.assertIn(restored["freshness"], {"unknown", "stale"})

    def test_camera_and_track_age_to_stale(self) -> None:
        self.projector.mark_camera_frame(0, rgb_sequence_number=1)
        self.projector.publish_observer_snapshot(_snapshot(1, _person(4, 7)))
        time.sleep(0.08)
        state = self.projector.snapshot()
        self.assertEqual(state["cameras"][0]["status"], "offline")
        self.assertEqual(state["visits"][0]["currentTracks"][0]["freshness"], "stale")
        self.assertEqual(state["visits"][0]["visibility"], "not_visible")

    def test_shelf_measurements_do_not_toggle_proximity_state(self) -> None:
        shelf = ShelfDefinition(
            shelf_id=1,
            label="Drinks",
            marker_id=10,
            approach_distance_mm=900,
            departure_distance_mm=1100,
            approach_dwell_milliseconds=500,
            departure_dwell_milliseconds=500,
            lost_visit_grace_milliseconds=1000,
            owner_switch_margin_mm=100,
            owner_switch_dwell_milliseconds=300,
        )
        occupied = ShelfProximityStatus(
            shelf_id=1,
            shelf_label="Drinks",
            marker_id=10,
            state="near",
            owner_visit_id=7,
            owner_customer_id=None,
            owner_track_id=2,
            source_camera_index=0,
            source_device_id="camera-a",
            distance_mm=700.0,
            measurement_age_milliseconds=0,
            proximity_session_id="session-1",
        )
        self.projector.publish_shelf_statuses((occupied,))
        self.projector.publish_shelf_snapshot(
            ShelfCameraSnapshot(
                camera_index=0,
                device_id="camera-a",
                camera_role="observer",
                rgb_sequence_number=2,
                depth_sequence_number=2,
                host_synced_seconds=2.0,
                published_at_unix_milliseconds=time.time_ns() // 1_000_000,
                shelves=(shelf,),
                anchors_by_shelf={},
                observations=(),
                states_by_shelf={1: "far"},
            )
        )
        self.projector.publish_shelf_statuses((occupied,))
        self.store.flush()

        state = self.projector.snapshot()
        self.assertEqual(state["shelves"][0]["state"], "occupied")
        connection = sqlite3.connect(self.path)
        try:
            change_count = connection.execute(
                "SELECT COUNT(*) FROM world_state_changes WHERE change_type='shelf_state_changed'"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(change_count, 0)

    def test_shelf_observations_from_other_cameras_are_not_overwritten(self) -> None:
        shelf = _shelf(1, 10)
        now_ms = time.time_ns() // 1_000_000
        self.projector.publish_observer_snapshot(_snapshot(1, _person(2, 7)))
        self.projector.publish_shelf_snapshot(
            _shelf_snapshot(
                0,
                (shelf,),
                (_shelf_observation(
                    shelf,
                    camera_index=0,
                    visit_id=7,
                    distance_mm=700,
                    observed_at_ms=now_ms,
                ),),
                observed_at_ms=now_ms,
            )
        )
        self.projector.publish_shelf_snapshot(
            _shelf_snapshot(
                1,
                (shelf,),
                (_shelf_observation(
                    shelf,
                    camera_index=1,
                    visit_id=7,
                    distance_mm=800,
                    observed_at_ms=now_ms + 1,
                ),),
                observed_at_ms=now_ms + 1,
            )
        )
        measurements = self.projector.snapshot()["visits"][0]["shelfMeasurements"]
        self.assertEqual(
            [
                (item["cameraIndex"], item["distanceMm"])
                for item in measurements
            ],
            [(0, 700), (1, 800)],
        )
        self.projector.publish_shelf_snapshot(
            _shelf_snapshot(1, (shelf,), (), observed_at_ms=now_ms + 2)
        )

        state = self.projector.snapshot()
        candidate = state["visits"][0]["nearestShelfCandidate"]
        self.assertEqual(candidate["distanceMm"], 700)
        self.assertEqual(candidate["cameraIndex"], 0)
        self.assertEqual(candidate["trackBoundingBox"], (100, 200, 300, 700))
        self.assertEqual(candidate["personDepthValidPixelCount"], 48)
        self.assertEqual(candidate["personDepthRoi"], (180, 350, 220, 410))
        self.assertEqual(candidate["personDepthAnchorPixel"], (200, 380))
        self.assertEqual(len(state["visits"][0]["shelfMeasurements"]), 1)
        self.assertEqual(state["visits"][0]["currentShelf"]["shelfId"], 1)
        self.assertEqual(state["visits"][0]["shelfPosition"]["shelfId"], 1)
        self.assertEqual(state["visits"][0]["shelfEngagementState"], "nearest")

    def test_raw_distance_becomes_position_without_proximity_event(self) -> None:
        shelf = _shelf(1, 10)
        now_ms = time.time_ns() // 1_000_000
        self.projector.publish_observer_snapshot(_snapshot(1, _person(2, 7)))
        self.projector.publish_shelf_snapshot(
            _shelf_snapshot(
                0,
                (shelf,),
                (_shelf_observation(
                    shelf,
                    camera_index=0,
                    visit_id=7,
                    distance_mm=700,
                    observed_at_ms=now_ms,
                ),),
                observed_at_ms=now_ms,
            )
        )
        far = ShelfProximityStatus(
            shelf_id=1,
            shelf_label=shelf.label,
            marker_id=10,
            state="far",
            owner_visit_id=7,
            owner_customer_id=None,
            owner_track_id=2,
            source_camera_index=0,
            source_device_id="camera-0",
            distance_mm=700,
            measurement_age_milliseconds=0,
            proximity_session_id=None,
        )
        self.projector.publish_shelf_statuses((far,))
        visit = self.projector.snapshot()["visits"][0]
        self.assertEqual(visit["nearestShelfCandidate"]["shelfId"], 1)
        self.assertEqual(visit["shelfPosition"]["shelfId"], 1)
        self.assertEqual(visit["engagedShelf"], visit["shelfPosition"])

        near = ShelfProximityStatus(
            **{
                **far.__dict__,
                "state": "near",
                "proximity_session_id": "session-1",
            }
        )
        self.projector.publish_shelf_statuses((near,))
        visit = self.projector.snapshot()["visits"][0]
        self.assertEqual(visit["engagedShelf"]["shelfId"], 1)
        self.assertEqual(visit["currentShelf"], visit["engagedShelf"])
        claims = _visit_claims(visit)
        self.assertEqual(claims["shelfPositionId"], 1)
        self.assertEqual(claims["engagedShelfId"], 1)
        self.assertEqual(claims["nearestShelfId"], 1)

    def test_closest_absolute_distance_wins_without_ambiguity(self) -> None:
        shelves = (_shelf(1, 10), _shelf(2, 11))
        now_ms = time.time_ns() // 1_000_000
        self.projector.publish_observer_snapshot(_snapshot(1, _person(2, 7)))
        observations = tuple(
            _shelf_observation(
                shelf,
                camera_index=0,
                visit_id=7,
                distance_mm=distance,
                observed_at_ms=now_ms,
            )
            for shelf, distance in zip(shelves, (650, 600))
        )
        self.projector.publish_shelf_snapshot(
            _shelf_snapshot(0, shelves, observations, observed_at_ms=now_ms)
        )
        statuses = tuple(
            ShelfProximityStatus(
                shelf_id=shelf.shelf_id,
                shelf_label=shelf.label,
                marker_id=shelf.marker_id,
                state="near",
                owner_visit_id=7,
                owner_customer_id=None,
                owner_track_id=2,
                source_camera_index=0,
                source_device_id="camera-0",
                distance_mm=distance,
                measurement_age_milliseconds=0,
                proximity_session_id=f"session-{shelf.shelf_id}",
            )
            for shelf, distance in zip(shelves, (650, 600))
        )
        self.projector.publish_shelf_statuses(statuses)

        visit = self.projector.snapshot()["visits"][0]
        self.assertEqual(visit["shelfEngagementState"], "nearest")
        self.assertEqual(visit["shelfPosition"]["shelfId"], 2)
        self.assertEqual(_visit_claims(visit)["shelfPositionId"], 2)

    def test_shelf_candidates_are_aged_when_camera_updates_stop(self) -> None:
        shelf = _shelf(1, 10)
        old_ms = time.time_ns() // 1_000_000 - 3000
        self.projector.publish_observer_snapshot(_snapshot(1, _person(2, 7)))
        self.projector.publish_shelf_snapshot(
            _shelf_snapshot(
                0,
                (shelf,),
                (_shelf_observation(
                    shelf,
                    camera_index=0,
                    visit_id=7,
                    distance_mm=700,
                    observed_at_ms=old_ms,
                ),),
                observed_at_ms=old_ms,
            )
        )

        visit = self.projector.snapshot()["visits"][0]
        self.assertEqual(visit["nearestShelfCandidate"]["freshness"], "stale")
        claims = _visit_claims(visit)
        self.assertEqual(claims["nearestShelfCandidateId"], 1)
        self.assertEqual(claims["nearestShelfCandidateFreshness"], "stale")

    def test_old_restored_proximity_status_does_not_create_position(self) -> None:
        shelf = _shelf(1, 10)
        self.projector.publish_observer_snapshot(_snapshot(1, _person(2, 7)))
        self.projector.publish_shelf_statuses(
            (
                ShelfProximityStatus(
                    shelf_id=1,
                    shelf_label=shelf.label,
                    marker_id=10,
                    state="near",
                    owner_visit_id=7,
                    owner_customer_id=None,
                    owner_track_id=2,
                    source_camera_index=0,
                    source_device_id="camera-0",
                    distance_mm=700,
                    measurement_age_milliseconds=3000,
                    proximity_session_id="restored-session",
                ),
            )
        )

        visit = self.projector.snapshot()["visits"][0]
        self.assertIsNone(visit["shelfPosition"])
        self.assertEqual(visit["shelfEngagementState"], "none")
        self.assertIsNone(_visit_claims(visit)["shelfPositionId"])


class WorldStateApiTests(unittest.TestCase):
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
        return next(route.endpoint for route in routes if getattr(route, "path", None) == path)

    def _start_run(self) -> str:
        run = self.route("/operator/api/test-runs")(
            {
                "scenario": "world-state",
                "verifier": "Milan",
                "subjects": [{"subjectId": "milan", "displayName": "Milan"}],
            },
            "Bearer secret",
        )
        self.route("/operator/api/test-runs/{run_id}/annotations")(
            run["runId"],
            {"annotationType": "physical_entry", "subjectId": "milan"},
            "Bearer secret",
        )
        return str(run["runId"])

    def test_subject_query_mapping_and_revision_bound_correction(self) -> None:
        run_id = self._start_run()
        self.server.publish_observer_snapshot(0, _snapshot(1, _person(4, 12)))
        subject_route = self.route(
            "/operator/api/test-runs/{run_id}/subjects/{subject_id}/world-state"
        )
        proposed = subject_route(run_id, "milan", False, True, None)
        self.assertEqual(proposed["resolution"]["status"], "single_candidate")
        self.assertEqual(proposed["claims"]["visitId"], 12)

        annotation_route = self.route("/operator/api/test-runs/{run_id}/annotations")
        annotation_route(
            run_id,
            {"annotationType": "subject_visit_mapping", "subjectId": "milan", "visitId": 12},
            "Bearer secret",
        )
        queried = subject_route(run_id, "milan", True, True, None)
        self.assertEqual(queried["resolution"]["status"], "confirmed")

        correction = annotation_route(
            run_id,
            {
                "annotationType": "world_state_claim_incorrect",
                "subjectId": "milan",
                "worldStateRef": queried["worldStateRef"],
                "claim": "nearestShelfId",
                "systemValue": None,
                "physicalValue": 3,
            },
            "Bearer secret",
        )
        self.assertEqual(correction["annotation"]["payload"]["physicalValue"], 3)

        tampered = dict(queried["worldStateRef"])
        tampered["revision"] += 1
        with self.assertRaises(HTTPException) as conflict:
            annotation_route(
                run_id,
                {
                    "annotationType": "world_state_claim_correct",
                    "subjectId": "milan",
                    "worldStateRef": tampered,
                    "claim": "visitId",
                    "systemValue": 12,
                },
                "Bearer secret",
            )
        self.assertEqual(conflict.exception.status_code, 409)

    def test_multiple_visits_are_ambiguous(self) -> None:
        run_id = self._start_run()
        self.server.publish_observer_snapshot(
            0,
            _snapshot(1, _person(4, 12), _person(5, 13)),
        )
        payload = self.route(
            "/operator/api/test-runs/{run_id}/subjects/{subject_id}/world-state"
        )(run_id, "milan", False, True, None)
        self.assertEqual(payload["resolution"]["status"], "ambiguous")
        self.assertEqual(payload["resolution"]["candidateVisitIds"], [12, 13])
        self.assertIsNone(payload["claims"]["visitId"])

    def test_observer_only_visit_is_proposed_and_can_be_mapped_without_entry(self) -> None:
        run = self.route("/operator/api/test-runs")(
            {
                "scenario": "already-inside",
                "verifier": "Milan",
                "subjects": [{"subjectId": "milan", "displayName": "Milan"}],
            },
            "Bearer secret",
        )
        run_id = str(run["runId"])
        self.server.publish_observer_snapshot(
            0,
            _snapshot(1, _person(4, 12, origin="observer_only")),
        )
        subject_route = self.route(
            "/operator/api/test-runs/{run_id}/subjects/{subject_id}/world-state"
        )

        proposed = subject_route(run_id, "milan", False, True, None)
        self.assertEqual(proposed["resolution"]["status"], "single_observer_candidate")
        self.assertEqual(proposed["claims"]["visitId"], 12)
        self.assertIsNone(proposed["claims"]["inside"])
        self.assertFalse(proposed["claims"]["entranceConfirmed"])

        self.route("/operator/api/test-runs/{run_id}/annotations")(
            run_id,
            {
                "annotationType": "subject_visit_mapping",
                "subjectId": "milan",
                "visitId": 12,
            },
            "Bearer secret",
        )
        mapped = subject_route(run_id, "milan", False, True, None)
        self.assertEqual(mapped["resolution"]["status"], "confirmed")

    def test_multiple_observer_only_visits_remain_ambiguous(self) -> None:
        run = self.route("/operator/api/test-runs")(
            {
                "scenario": "already-inside-group",
                "verifier": "Milan",
                "subjects": [{"subjectId": "milan", "displayName": "Milan"}],
            },
            "Bearer secret",
        )
        run_id = str(run["runId"])
        self.server.publish_observer_snapshot(
            0,
            _snapshot(
                1,
                _person(4, 12, origin="observer_only"),
                _person(5, 13, origin="observer_only"),
            ),
        )

        payload = self.route(
            "/operator/api/test-runs/{run_id}/subjects/{subject_id}/world-state"
        )(run_id, "milan", False, True, None)

        self.assertEqual(
            payload["resolution"]["status"],
            "ambiguous_observer_candidates",
        )
        self.assertEqual(payload["resolution"]["candidateVisitIds"], [12, 13])
        self.assertIsNone(payload["claims"]["visitId"])

    def test_world_state_read_api_can_run_without_operator_console(self) -> None:
        self.server.stop()
        root = Path(self.temporary.name)
        self.server = MjpegStreamServer(
            camera_device_ids=["camera-a"],
            camera_roles=["observer"],
            world_state_db=root / "world-only.sqlite",
        )
        self.assertIsNone(self.server.operator_store)
        endpoint = self.route("/world-state")
        payload = endpoint(False, True, None)
        self.assertEqual(payload["cameras"][0]["deviceId"], "camera-a")
        self.assertNotIn("snapshotId", payload)

    def test_visit_shelf_position_endpoint_returns_nearest_marker_distance(self) -> None:
        shelf = _shelf(1, 10)
        now_ms = time.time_ns() // 1_000_000
        self.server.publish_observer_snapshot(0, _snapshot(1, _person(4, 12)))
        self.server.publish_shelf_snapshot(
            0,
            _shelf_snapshot(
                0,
                (shelf,),
                (
                    _shelf_observation(
                        shelf,
                        camera_index=0,
                        visit_id=12,
                        distance_mm=725,
                        observed_at_ms=now_ms,
                    ),
                ),
                observed_at_ms=now_ms,
            ),
        )

        payload = self.route("/world-state/visits/{visit_id}/shelf-position")(12)

        self.assertEqual(payload["visitId"], 12)
        self.assertEqual(payload["position"]["shelfId"], 1)
        self.assertEqual(payload["position"]["markerId"], 10)
        self.assertEqual(payload["position"]["distanceMm"], 725)
        self.assertEqual(payload["candidateCount"], 1)
        self.assertEqual(payload["measurementCount"], 1)
        self.assertEqual(payload["measurements"][0]["cameraIndex"], 0)


if __name__ == "__main__":
    unittest.main()
