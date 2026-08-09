from __future__ import annotations

import copy
import threading
import time
from typing import Any, Mapping, Sequence

from pipeline.observer_api import ObserverCameraSnapshot
from pipeline.shelf_api import ShelfCameraSnapshot
from pipeline.shelf_proximity import ShelfProximityStatus
from pipeline.world_state_store import WorldStateStore


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


class WorldStateProjector:
    """Thread-safe materialized projection of the recognition system's belief."""

    def __init__(
        self,
        *,
        camera_device_ids: Sequence[str],
        camera_roles: Sequence[str],
        camera_timeout_seconds: float,
        store: WorldStateStore,
        track_aging_seconds: float = 1.5,
        track_stale_seconds: float = 5.0,
        shelf_stale_seconds: float = 2.0,
    ) -> None:
        self.store = store
        self.process_instance_id = store.process_instance_id
        self.camera_timeout_ms = int(camera_timeout_seconds * 1000)
        self.track_aging_ms = int(track_aging_seconds * 1000)
        self.track_stale_ms = int(track_stale_seconds * 1000)
        self.shelf_stale_ms = int(shelf_stale_seconds * 1000)
        self._revision, self._revision_ceiling = store.reserve_revision_block()
        self._revision -= 1
        self._source_event_high_watermark: int | None = None
        self._last_persist_enqueue_monotonic = 0.0
        self._lock = threading.RLock()
        self._cameras: dict[int, dict[str, Any]] = {
            index: {
                "cameraIndex": index,
                "deviceId": device_id,
                "role": camera_roles[index],
                "status": "offline",
                "freshness": "unknown",
                "observedAtUnixMilliseconds": None,
                "lastFrameAtUnixMilliseconds": None,
                "lastObservationAtUnixMilliseconds": None,
                "rgbSequenceNumber": None,
                "depthSequenceNumber": None,
                "visibleTrackIds": [],
                "visibleVisitIds": [],
                "tracks": [],
            }
            for index, device_id in enumerate(camera_device_ids)
        }
        self._visits: dict[int, dict[str, Any]] = {}
        self._shelves: dict[int, dict[str, Any]] = {}
        self._shelf_observations: dict[
            tuple[int, int, int, int], dict[str, Any]
        ] = {}
        self._restore(store.load_entities())
        self._persist()

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def _restore(self, entities: Mapping[str, list[dict[str, Any]]]) -> None:
        for persisted in entities.get("visits", []):
            visit = dict(persisted)
            visit["currentTracks"] = []
            visit["shelfPosition"] = None
            visit["currentShelf"] = None
            visit["engagedShelf"] = None
            visit["nearestShelfCandidate"] = None
            visit["shelfCandidates"] = []
            visit["shelfMeasurements"] = []
            visit["shelfEngagementState"] = "stale"
            visit["productRecognition"] = None
            visit["productRecognitionHistory"] = []
            visit["visibility"] = "unknown"
            visit["freshness"] = "stale"
            visit["restoredFromPersistence"] = True
            self._visits[int(visit["visitId"])] = visit
        for persisted in entities.get("shelves", []):
            shelf = dict(persisted)
            shelf["nearbyVisits"] = []
            shelf["closest"] = None
            shelf["freshness"] = "stale"
            shelf["restoredFromPersistence"] = True
            self._shelves[int(shelf["shelfId"])] = shelf

    def _next_revision(self) -> int:
        if self._revision >= self._revision_ceiling:
            start, ceiling = self.store.reserve_revision_block()
            self._revision = start - 1
            self._revision_ceiling = ceiling
        self._revision += 1
        return self._revision

    def mark_camera_frame(
        self,
        camera_index: int,
        *,
        rgb_sequence_number: int | None,
        observed_at_unix_milliseconds: int | None = None,
    ) -> None:
        now_ms = _now_ms() if observed_at_unix_milliseconds is None else observed_at_unix_milliseconds
        with self._lock:
            camera = self._cameras[camera_index]
            before_status = camera.get("status")
            camera.update(
                {
                    "status": "active",
                    "freshness": "current",
                    "observedAtUnixMilliseconds": now_ms,
                    "lastFrameAtUnixMilliseconds": now_ms,
                    "rgbSequenceNumber": rgb_sequence_number,
                }
            )
            self._next_revision()
            changes = []
            if before_status != "active":
                changes.append(
                    self._change(
                        "camera", camera_index, "camera_active", before_status, "active", now_ms
                    )
                )
            self._persist(changes)

    def publish_observer_snapshot(self, snapshot: ObserverCameraSnapshot) -> None:
        now_ms = snapshot.published_at_unix_milliseconds
        with self._lock:
            camera = self._cameras[snapshot.camera_index]
            previous_sequence = camera.get("lastObservationRgbSequenceNumber")
            if previous_sequence is not None and snapshot.rgb_sequence_number < previous_sequence:
                return
            previous_track_refs = {
                (int(track["visitId"]), int(track["trackId"]))
                for visit in self._visits.values()
                for track in visit.get("currentTracks", [])
                if track.get("cameraIndex") == snapshot.camera_index
                and track.get("visitId") is not None
            }
            current_track_refs: set[tuple[int, int]] = set()
            visible_track_ids: list[int] = []
            visible_visit_ids: set[int] = set()
            camera_tracks: list[dict[str, Any]] = []
            for person in snapshot.observations:
                visible_track_ids.append(person.track_id)
                camera_tracks.append(
                    {
                        "trackId": person.track_id,
                        "trackStatus": person.track_status,
                        "visitId": person.visit_id,
                        "visitOrigin": person.visit_origin,
                        "customerId": person.customer_id,
                        "customerBindingStatus": person.customer_binding_status,
                        "boundingBox": {
                            "x1": person.bounding_box[0],
                            "y1": person.bounding_box[1],
                            "x2": person.bounding_box[2],
                            "y2": person.bounding_box[3],
                        },
                        "detectionScore": person.detection_score,
                        "matchScore": person.matched_score,
                        "matchState": person.match_state,
                        "matchDecision": person.match_decision,
                        "matchReason": person.match_reason,
                        "rgbSequenceNumber": snapshot.rgb_sequence_number,
                        "hostSyncedSeconds": snapshot.host_synced_seconds,
                        "observedAtUnixMilliseconds": now_ms,
                        "freshness": "current",
                    }
                )
                if person.visit_id is None:
                    continue
                visit_id = int(person.visit_id)
                visible_visit_ids.add(visit_id)
                current_track_refs.add((visit_id, person.track_id))
                visit = self._visits.setdefault(
                    visit_id,
                    {
                        "visitId": visit_id,
                        "status": "active",
                        "origin": person.visit_origin,
                        "customerId": person.customer_id,
                        "currentTracks": [],
                    },
                )
                tracks = [
                    item
                    for item in visit.get("currentTracks", [])
                    if not (
                        item.get("cameraIndex") == snapshot.camera_index
                        and item.get("trackId") == person.track_id
                    )
                ]
                tracks.append(
                    {
                        "cameraIndex": snapshot.camera_index,
                        "deviceId": snapshot.device_id,
                        "trackId": person.track_id,
                        "visitId": visit_id,
                        "trackStatus": person.track_status,
                        "boundingBox": {
                            "x1": person.bounding_box[0],
                            "y1": person.bounding_box[1],
                            "x2": person.bounding_box[2],
                            "y2": person.bounding_box[3],
                        },
                        "frame": {
                            "width": snapshot.frame_width,
                            "height": snapshot.frame_height,
                            "rgbSequenceNumber": snapshot.rgb_sequence_number,
                        },
                        "detectionScore": person.detection_score,
                        "matchScore": person.matched_score,
                        "matchState": person.match_state,
                        "matchDecision": person.match_decision,
                        "matchReason": person.match_reason,
                        "lastSeenHostSeconds": snapshot.host_synced_seconds,
                        "observedAtUnixMilliseconds": now_ms,
                        "freshness": "current",
                    }
                )
                visit.update(
                    {
                        "origin": person.visit_origin or visit.get("origin"),
                        "customerId": person.customer_id or visit.get("customerId"),
                        "customerBindingStatus": person.customer_binding_status,
                        "currentTracks": tracks,
                        "lastCameraIndex": snapshot.camera_index,
                        "lastDeviceId": snapshot.device_id,
                        "lastTrackId": person.track_id,
                        "lastRgbSequenceNumber": snapshot.rgb_sequence_number,
                        "lastSeenHostSeconds": snapshot.host_synced_seconds,
                        "observedAtUnixMilliseconds": now_ms,
                        "faceIdentityIds": sorted(
                            set(visit.get("faceIdentityIds", []))
                            | set(person.face_identity_ids)
                        ),
                        "visibility": "visible",
                        "freshness": "current",
                        "restoredFromPersistence": False,
                    }
                )
            disappeared = previous_track_refs - current_track_refs
            for visit_id, track_id in disappeared:
                visit = self._visits.get(visit_id)
                if visit is None:
                    continue
                visit["currentTracks"] = [
                    item
                    for item in visit.get("currentTracks", [])
                    if not (
                        item.get("cameraIndex") == snapshot.camera_index
                        and item.get("trackId") == track_id
                    )
                ]
            camera.update(
                {
                    "status": "active",
                    "freshness": "current",
                    "observedAtUnixMilliseconds": now_ms,
                    "lastObservationAtUnixMilliseconds": now_ms,
                    "lastObservationRgbSequenceNumber": snapshot.rgb_sequence_number,
                    "rgbSequenceNumber": snapshot.rgb_sequence_number,
                    "hostSyncedSeconds": snapshot.host_synced_seconds,
                    "visibleTrackIds": sorted(visible_track_ids),
                    "visibleVisitIds": sorted(visible_visit_ids),
                    "tracks": camera_tracks,
                }
            )
            self._next_revision()
            self._persist()

    def publish_shelf_snapshot(self, snapshot: ShelfCameraSnapshot) -> None:
        now_ms = snapshot.published_at_unix_milliseconds
        with self._lock:
            for key in [
                key
                for key in self._shelf_observations
                if key[0] == snapshot.camera_index
            ]:
                self._shelf_observations.pop(key, None)
            for observation in snapshot.observations:
                if observation.visit_id is None:
                    continue
                self._shelf_observations[
                    (
                        snapshot.camera_index,
                        observation.shelf_id,
                        int(observation.visit_id),
                        observation.marker_id,
                    )
                ] = {
                    "shelfId": observation.shelf_id,
                    "visitId": int(observation.visit_id),
                    "trackId": observation.track_id,
                    "customerId": observation.customer_id,
                    "distanceMm": observation.distance_mm,
                    "cameraIndex": snapshot.camera_index,
                    "deviceId": snapshot.device_id,
                    "markerId": observation.marker_id,
                    "rgbSequenceNumber": snapshot.rgb_sequence_number,
                    "depthSequenceNumber": snapshot.depth_sequence_number,
                    "hostSyncedSeconds": snapshot.host_synced_seconds,
                    "observedAtUnixMilliseconds": now_ms,
                    "freshness": "current",
                    "trackBoundingBox": observation.track_bounding_box,
                    "personDepthMm": observation.person_depth_mm,
                    "personDepthValidPixelCount": (
                        observation.person_depth_valid_pixel_count
                    ),
                    "personDepthRoi": observation.person_depth_roi,
                    "personDepthAnchorPixel": observation.person_depth_anchor_px,
                    "personPoint3dMm": observation.person_point_3d_mm,
                    "anchorPoint3dMm": observation.anchor.point_3d_mm,
                }
            for definition in snapshot.shelves:
                shelf = self._shelves.setdefault(definition.shelf_id, {})
                shelf.setdefault("state", "available")
                shelf.update(
                    {
                        "shelfId": definition.shelf_id,
                        "label": definition.label,
                        "markerIds": list(definition.all_marker_ids),
                        "approachDistanceMm": definition.approach_distance_mm,
                        "departureDistanceMm": definition.departure_distance_mm,
                        "observedAtUnixMilliseconds": now_ms,
                        "freshness": "current",
                        "restoredFromPersistence": False,
                    }
                )
            camera = self._cameras[snapshot.camera_index]
            camera["depthSequenceNumber"] = snapshot.depth_sequence_number
            self._refresh_shelf_observations(now_ms)
            self._next_revision()
            self._recompute_visit_shelves()
            self._persist()

    def publish_shelf_statuses(self, statuses: Sequence[ShelfProximityStatus]) -> None:
        now_ms = _now_ms()
        changes: list[dict[str, Any]] = []
        with self._lock:
            for status in statuses:
                measurement_observed_ms = (
                    None
                    if status.measurement_age_milliseconds is None
                    else now_ms - max(0, status.measurement_age_milliseconds)
                )
                shelf = self._shelves.setdefault(
                    status.shelf_id,
                    {
                        "shelfId": status.shelf_id,
                        "label": status.shelf_label,
                        "markerIds": [status.marker_id],
                        "nearbyVisits": [],
                    },
                )
                previous = shelf.get("state")
                engaged = (
                    status.owner_visit_id is not None
                    and status.proximity_session_id is not None
                    and status.state in {"near", "departing"}
                )
                current = "occupied" if engaged else "available"
                shelf.update(
                    {
                        "state": current,
                        "proximityState": status.state,
                        "proximitySessionId": status.proximity_session_id,
                        "ownerVisitId": status.owner_visit_id,
                        "ownerCustomerId": status.owner_customer_id,
                        "ownerTrackId": status.owner_track_id,
                        "sourceCameraIndex": status.source_camera_index,
                        "sourceDeviceId": status.source_device_id,
                        "distanceMm": status.distance_mm,
                        "measurementAgeMilliseconds": status.measurement_age_milliseconds,
                        "measurementObservedAtUnixMilliseconds": measurement_observed_ms,
                        "observedAtUnixMilliseconds": now_ms,
                        "freshness": "current",
                    }
                )
                if previous is not None and previous != current:
                    changes.append(
                        self._change(
                            "shelf", status.shelf_id, "shelf_state_changed", previous, current, now_ms
                        )
                    )
            self._next_revision()
            self._recompute_visit_shelves()
            self._persist(changes)

    def publish_visit_state(
        self,
        visit_id: int | None,
        *,
        status: str | None = None,
        origin: str | None = None,
        customer_id: str | None = None,
        host_synced_seconds: float | None = None,
        camera_index: int | None = None,
        device_id: str | None = None,
        track_id: int | None = None,
        event_type: str | None = None,
    ) -> None:
        if visit_id is None:
            return
        now_ms = _now_ms()
        with self._lock:
            visit = self._visits.setdefault(
                visit_id,
                {"visitId": visit_id, "currentTracks": [], "freshness": "unknown"},
            )
            before = copy.deepcopy(visit)
            if status is not None:
                visit["status"] = status
            if origin is not None:
                visit["origin"] = origin
            if customer_id is not None:
                visit["customerId"] = customer_id
                visit["customerBindingStatus"] = "bound"
            if host_synced_seconds is not None:
                visit["lastSeenHostSeconds"] = host_synced_seconds
            if camera_index is not None:
                visit["lastCameraIndex"] = camera_index
            if device_id is not None:
                visit["lastDeviceId"] = device_id
            if track_id is not None:
                visit["lastTrackId"] = track_id
            visit["updatedAtUnixMilliseconds"] = now_ms
            if status == "left":
                visit["visibility"] = "not_visible"
                visit["currentTracks"] = []
                visit["shelfPosition"] = None
                visit["currentShelf"] = None
                visit["engagedShelf"] = None
                visit["nearestShelfCandidate"] = None
                visit["shelfCandidates"] = []
                visit["shelfMeasurements"] = []
                visit["shelfEngagementState"] = "none"
                visit["productRecognition"] = None
                visit["productRecognitionHistory"] = []
            self._next_revision()
            change_type = event_type or "visit_state_changed"
            self._persist(
                [
                    self._change(
                        "visit",
                        visit_id,
                        change_type,
                        before,
                        copy.deepcopy(visit),
                        now_ms,
                        host_synced_seconds=host_synced_seconds,
                    )
                ]
            )

    def publish_product_recognition(self, payload: Mapping[str, Any]) -> None:
        visit_id = int(payload["visitId"])
        now_ms = int(payload["observedAtUnixMilliseconds"])
        with self._lock:
            visit = self._visits.setdefault(
                visit_id,
                {"visitId": visit_id, "currentTracks": [], "freshness": "unknown"},
            )
            previous = visit.get("productRecognition")
            history = [
                dict(item)
                for item in visit.get("productRecognitionHistory", [])
                if now_ms - int(item["observedAtUnixMilliseconds"])
                <= int(payload["maxAgeMilliseconds"])
            ]
            history.append(dict(payload))
            candidate_evidence: dict[str, dict[str, Any]] = {}
            for observation in history:
                confirmed_labels: set[str] = set()
                for candidate in observation.get("candidates", []):
                    model_label = str(candidate["modelLabel"])
                    evidence = candidate_evidence.setdefault(
                        model_label,
                        {
                            **candidate,
                            "confirmations": 0,
                            "bestScore": 0.0,
                            "latestScore": 0.0,
                        },
                    )
                    if model_label not in confirmed_labels:
                        evidence["confirmations"] += 1
                        confirmed_labels.add(model_label)
                    evidence["bestScore"] = max(
                        float(evidence["bestScore"]), float(candidate["score"])
                    )
                    evidence["latestScore"] = float(candidate["score"])
            candidates = sorted(
                candidate_evidence.values(),
                key=lambda item: (
                    -int(item["confirmations"]),
                    -float(item["bestScore"]),
                    int(item["classId"]),
                ),
            )
            recognition = {
                **dict(payload),
                "status": "recognized" if candidates else "no_product",
                "bestCandidate": None if not candidates else candidates[0],
                "candidates": candidates,
            }
            visit["productRecognitionHistory"] = history
            visit["productRecognition"] = recognition
            self._next_revision()
            self._persist(
                [
                    self._change(
                        "visit",
                        visit_id,
                        "product_recognition_updated",
                        previous,
                        copy.deepcopy(recognition),
                        now_ms,
                        host_synced_seconds=float(payload["hostSyncedSeconds"]),
                    )
                ]
            )

    def publish_transition(
        self,
        *,
        event_type: str,
        entity_type: str,
        entity_id: int | str,
        payload: Mapping[str, Any],
        occurred_at_unix_milliseconds: int | None = None,
        host_synced_seconds: float | None = None,
        source_event_id: int | None = None,
    ) -> None:
        now_ms = _now_ms() if occurred_at_unix_milliseconds is None else occurred_at_unix_milliseconds
        with self._lock:
            if source_event_id is not None:
                self._source_event_high_watermark = max(
                    source_event_id,
                    self._source_event_high_watermark or 0,
                )
            self._next_revision()
            self._persist(
                [
                    self._change(
                        entity_type,
                        entity_id,
                        event_type,
                        None,
                        dict(payload),
                        now_ms,
                        host_synced_seconds=host_synced_seconds,
                        source_event_id=source_event_id,
                        source=payload,
                    )
                ]
            )

    def snapshot(self) -> dict[str, Any]:
        now_ms = _now_ms()
        with self._lock:
            cameras = [self._fresh_camera(copy.deepcopy(item), now_ms) for item in self._cameras.values()]
            visits = [self._fresh_visit(copy.deepcopy(item), now_ms) for item in self._visits.values()]
            shelves = [self._fresh_shelf(copy.deepcopy(item), now_ms) for item in self._shelves.values()]
            inside = [item for item in visits if item.get("status") == "inside"]
            health = "current" if any(item["freshness"] == "current" for item in cameras) else "stale"
            return {
                "schemaVersion": 1,
                "revision": self._revision,
                "processInstanceId": self.process_instance_id,
                "generatedAtUnixMilliseconds": now_ms,
                "sourceEventIdHighWatermark": self._source_event_high_watermark,
                "health": health,
                "persistenceHealth": "degraded" if self.store._writer_error else "healthy",
                "occupancy": {
                    "insideVisitCount": len(inside),
                    "insideVisitIds": sorted(int(item["visitId"]) for item in inside),
                },
                "cameras": sorted(cameras, key=lambda item: item["cameraIndex"]),
                "visits": sorted(visits, key=lambda item: item["visitId"]),
                "shelves": sorted(shelves, key=lambda item: item["shelfId"]),
            }

    def _persist(self, changes: list[Mapping[str, Any]] | None = None) -> None:
        now = time.monotonic()
        if not changes and now - self._last_persist_enqueue_monotonic < 0.25:
            return
        self._last_persist_enqueue_monotonic = now
        self.store.enqueue(self.snapshot(), changes=changes)

    def _recompute_visit_shelves(self) -> None:
        measurements: dict[int, list[dict[str, Any]]] = {}
        for observation in self._shelf_observations.values():
            visit_id = observation.get("visitId")
            shelf_id = observation.get("shelfId")
            shelf = self._shelves.get(int(shelf_id)) if shelf_id is not None else None
            if visit_id is None or shelf is None:
                continue
            measurements.setdefault(int(visit_id), []).append(
                {
                    "shelfId": shelf["shelfId"],
                    "label": shelf.get("label"),
                    "approachDistanceMm": shelf.get("approachDistanceMm"),
                    "departureDistanceMm": shelf.get("departureDistanceMm"),
                    **observation,
                }
            )
        for visit_id, visit in self._visits.items():
            all_measurements = sorted(
                measurements.get(visit_id, []),
                key=lambda item: (
                    item["distanceMm"],
                    item["shelfId"],
                    item.get("markerId", 0),
                    item.get("cameraIndex", 0),
                ),
            )
            best_by_shelf: dict[int, dict[str, Any]] = {}
            for measurement in all_measurements:
                best_by_shelf.setdefault(int(measurement["shelfId"]), measurement)
            options = sorted(
                best_by_shelf.values(),
                key=lambda item: (
                    item["distanceMm"],
                    item["shelfId"],
                    item.get("markerId", 0),
                    item.get("cameraIndex", 0),
                ),
            )
            position = None if not options else options[0]
            visit["shelfMeasurements"] = all_measurements
            visit["shelfCandidates"] = options
            visit["shelfPosition"] = position
            visit["nearestShelfCandidate"] = position
            # Keep legacy names as aliases while consumers move to shelfPosition.
            visit["engagedShelf"] = position
            visit["currentShelf"] = position
            visit["shelfEngagementState"] = "nearest" if position is not None else "none"

    def _refresh_shelf_observations(self, now_ms: int) -> None:
        for key, observation in list(self._shelf_observations.items()):
            if (
                now_ms - int(observation["observedAtUnixMilliseconds"])
                > self.shelf_stale_ms
            ):
                self._shelf_observations.pop(key, None)
        by_shelf: dict[int, dict[int, dict[str, Any]]] = {}
        for (
            _camera_index,
            shelf_id,
            visit_id,
            _marker_id,
        ), observation in self._shelf_observations.items():
            current = by_shelf.setdefault(shelf_id, {}).get(visit_id)
            if current is None or observation["distanceMm"] < current["distanceMm"]:
                by_shelf[shelf_id][visit_id] = observation
        for shelf_id, shelf in self._shelves.items():
            nearby = sorted(
                (dict(item) for item in by_shelf.get(shelf_id, {}).values()),
                key=lambda item: (item["distanceMm"], item["cameraIndex"]),
            )
            shelf["nearbyVisits"] = nearby
            shelf["closest"] = None if not nearby else nearby[0]

    def _fresh_camera(self, camera: dict[str, Any], now_ms: int) -> dict[str, Any]:
        observed = camera.get("lastFrameAtUnixMilliseconds")
        age = None if observed is None else max(0, now_ms - int(observed))
        camera["ageMilliseconds"] = age
        if age is None:
            camera["freshness"] = "unknown"
            camera["status"] = "offline"
        elif age <= self.camera_timeout_ms:
            camera["freshness"] = "current"
            camera["status"] = "active"
        else:
            camera["freshness"] = "stale"
            camera["status"] = "offline"
            camera["visibleTrackIds"] = []
            camera["visibleVisitIds"] = []
            camera["tracks"] = []
        return camera

    def _fresh_visit(self, visit: dict[str, Any], now_ms: int) -> dict[str, Any]:
        tracks = []
        for track in visit.get("currentTracks", []):
            observed = track.get("observedAtUnixMilliseconds")
            age = None if observed is None else max(0, now_ms - int(observed))
            track["ageMilliseconds"] = age
            track["freshness"] = (
                "unknown"
                if age is None
                else "current"
                if age <= self.track_aging_ms
                else "aging"
                if age <= self.track_stale_ms
                else "stale"
            )
            tracks.append(track)
        visit["currentTracks"] = tracks
        recognition = visit.get("productRecognition")
        if recognition is not None:
            recognition = dict(recognition)
            observed_ms = recognition.get("observedAtUnixMilliseconds")
            age_ms = (
                None
                if observed_ms is None
                else max(0, now_ms - int(observed_ms))
            )
            recognition["ageMilliseconds"] = age_ms
            recognition["freshness"] = (
                "unknown"
                if age_ms is None
                else "current"
                if age_ms <= int(recognition.get("maxAgeMilliseconds", 3000))
                else "stale"
            )
            visit["productRecognition"] = recognition
        candidates = [
            self._fresh_shelf_reference(dict(item), now_ms)
            for item in visit.get("shelfCandidates", [])
        ]
        visit["shelfMeasurements"] = [
            self._fresh_shelf_reference(dict(item), now_ms)
            for item in visit.get("shelfMeasurements", [])
        ]
        visit["shelfCandidates"] = candidates
        position = None if not candidates else candidates[0]
        visit["shelfPosition"] = position
        visit["nearestShelfCandidate"] = position
        visit["engagedShelf"] = position
        visit["currentShelf"] = position
        if position is None:
            visit["shelfEngagementState"] = "none"
        elif position["freshness"] == "stale":
            visit["shelfEngagementState"] = "stale"
        else:
            visit["shelfEngagementState"] = "nearest"
        if any(item["freshness"] == "current" for item in tracks):
            visit["visibility"] = "visible"
            visit["freshness"] = "current"
        elif any(item["freshness"] == "aging" for item in tracks):
            visit["visibility"] = "temporarily_lost"
            visit["freshness"] = "aging"
        elif visit.get("status") == "left":
            visit["visibility"] = "not_visible"
            visit["freshness"] = "stale"
        else:
            visit["visibility"] = (
                "not_visible"
                if tracks or visit.get("observedAtUnixMilliseconds") is not None
                else "unknown"
            )
            visit["freshness"] = "stale" if visit.get("observedAtUnixMilliseconds") else "unknown"
        return visit

    def _fresh_shelf_reference(
        self, shelf: dict[str, Any], now_ms: int
    ) -> dict[str, Any]:
        observed = shelf.get("observedAtUnixMilliseconds")
        age = None if observed is None else max(0, now_ms - int(observed))
        shelf["ageMilliseconds"] = age
        shelf["freshness"] = (
            "unknown"
            if age is None
            else "current"
            if age <= self.shelf_stale_ms
            else "stale"
        )
        return shelf

    def _fresh_shelf(self, shelf: dict[str, Any], now_ms: int) -> dict[str, Any]:
        observed = shelf.get("observedAtUnixMilliseconds")
        age = None if observed is None else max(0, now_ms - int(observed))
        shelf["ageMilliseconds"] = age
        if age is None:
            shelf["freshness"] = "unknown"
        elif age <= self.shelf_stale_ms:
            shelf["freshness"] = "current"
        else:
            shelf["freshness"] = "stale"
            shelf["state"] = "stale"
            shelf["nearbyVisits"] = []
            shelf["closest"] = None
        return shelf

    def _change(
        self,
        entity_type: str,
        entity_id: int | str,
        change_type: str,
        before: Any,
        after: Any,
        occurred_at: int,
        *,
        host_synced_seconds: float | None = None,
        source_event_id: int | None = None,
        source: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "revision": self._revision,
            "entityType": entity_type,
            "entityId": str(entity_id),
            "changeType": change_type,
            "occurredAtUnixMilliseconds": occurred_at,
            "hostSyncedSeconds": host_synced_seconds,
            "sourceEventId": source_event_id,
            "source": dict(source or {}),
            "before": before,
            "after": after,
        }
