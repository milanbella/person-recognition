from __future__ import annotations

from typing import Any, Mapping

from fastapi import APIRouter, HTTPException, Query

from pipeline.operator_test_store import OperatorTestStore
from pipeline.world_state import WorldStateProjector
from pipeline.world_state_store import WorldStateStore


def _entity(items: list[dict[str, Any]], key: str, value: int) -> dict[str, Any]:
    for item in items:
        if item.get(key) == value:
            return item
    raise HTTPException(status_code=404, detail=f"Unknown {key}: {value}")


def _visit_claims(visit: Mapping[str, Any] | None) -> dict[str, Any]:
    if visit is None:
        return {
            "visitId": None,
            "inside": None,
            "status": "unknown",
            "entranceConfirmed": None,
            "customerId": None,
            "visibility": "unknown",
            "visibleOnCameraIndexes": [],
            "shelfPositionId": None,
            "shelfPositionDistanceMm": None,
            "shelfPositionFreshness": "unknown",
            "nearestShelfId": None,
            "nearestShelfDistanceMm": None,
            "nearestShelfFreshness": "unknown",
            "nearestShelfCandidateId": None,
            "nearestShelfCandidateDistanceMm": None,
            "nearestShelfCandidateFreshness": "unknown",
            "engagedShelfId": None,
            "engagedShelfDistanceMm": None,
            "engagedShelfFreshness": "unknown",
            "shelfEngagementState": "none",
            "freshness": "unknown",
        }
    tracks = [
        item
        for item in visit.get("currentTracks", [])
        if item.get("freshness") in {"current", "aging"}
    ]
    position = visit.get("shelfPosition") or visit.get("nearestShelfCandidate")
    status = visit.get("status", "unknown")
    inside = True if status == "inside" else False if status == "left" else None
    return {
        "visitId": visit.get("visitId"),
        "inside": inside,
        "status": status,
        "entranceConfirmed": visit.get("origin") == "entrance_confirmed",
        "origin": visit.get("origin"),
        "customerId": visit.get("customerId"),
        "customerBindingStatus": visit.get("customerBindingStatus"),
        "visibility": visit.get("visibility", "unknown"),
        "visibleOnCameraIndexes": sorted(
            {int(item["cameraIndex"]) for item in tracks}
        ),
        "shelfPositionId": None if position is None else position.get("shelfId"),
        "shelfPositionDistanceMm": None if position is None else position.get("distanceMm"),
        "shelfPositionFreshness": "unknown" if position is None else position.get("freshness", "unknown"),
        "nearestShelfId": None if position is None else position.get("shelfId"),
        "nearestShelfDistanceMm": None if position is None else position.get("distanceMm"),
        "nearestShelfFreshness": "unknown" if position is None else position.get("freshness", "unknown"),
        "nearestShelfCandidateId": None if position is None else position.get("shelfId"),
        "nearestShelfCandidateDistanceMm": None if position is None else position.get("distanceMm"),
        "nearestShelfCandidateFreshness": "unknown" if position is None else position.get("freshness", "unknown"),
        "engagedShelfId": None if position is None else position.get("shelfId"),
        "engagedShelfDistanceMm": None if position is None else position.get("distanceMm"),
        "engagedShelfFreshness": "unknown" if position is None else position.get("freshness", "unknown"),
        "shelfEngagementState": visit.get("shelfEngagementState", "none"),
        "freshness": visit.get("freshness", "unknown"),
    }


def subject_state_payload(
    *,
    snapshot: Mapping[str, Any],
    operator_store: OperatorTestStore,
    run_id: str,
    subject_id: str,
) -> dict[str, Any]:
    run = operator_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown test run: {run_id}")
    subjects = operator_store.subjects(run_id)
    if not any(item.get("subjectId") == subject_id for item in subjects):
        raise HTTPException(status_code=404, detail=f"Unknown test subject: {subject_id}")
    visits = {
        int(item["visitId"]): item for item in snapshot.get("visits", [])
    }
    mapped_visit_id = operator_store.subject_visit_mapping(run_id, subject_id)
    if mapped_visit_id is not None:
        candidate_ids = [mapped_visit_id]
        resolution_status = "confirmed"
        selected = visits.get(mapped_visit_id)
    else:
        entrance_candidates = [
            item
            for item in visits.values()
            if item.get("status") != "left"
            and item.get("origin") == "entrance_confirmed"
            and item.get("freshness") in {"current", "aging"}
        ]
        observer_candidates = [
            item
            for item in visits.values()
            if item.get("status") != "left"
            and item.get("origin") == "observer_only"
            and item.get("freshness") in {"current", "aging"}
        ]
        candidates = entrance_candidates or observer_candidates
        candidate_ids = sorted(int(item["visitId"]) for item in candidates)
        if len(candidates) == 1:
            resolution_status = (
                "single_candidate"
                if entrance_candidates
                else "single_observer_candidate"
            )
            selected = candidates[0]
        elif candidates:
            resolution_status = (
                "ambiguous"
                if entrance_candidates
                else "ambiguous_observer_candidates"
            )
            selected = None
        else:
            resolution_status = "unknown"
            selected = None
    return {
        "schemaVersion": snapshot["schemaVersion"],
        "revision": snapshot["revision"],
        "processInstanceId": snapshot["processInstanceId"],
        "generatedAtUnixMilliseconds": snapshot["generatedAtUnixMilliseconds"],
        "runId": run_id,
        "subjectId": subject_id,
        "resolution": {
            "status": resolution_status,
            "visitId": None if selected is None else selected.get("visitId"),
            "candidateVisitIds": candidate_ids,
        },
        "claims": _visit_claims(selected),
        "visit": selected,
        "freshness": "ambiguous"
        if resolution_status in {"ambiguous", "ambiguous_observer_candidates"}
        else "unknown"
        if selected is None
        else selected.get("freshness", "unknown"),
    }


def create_world_state_router(
    *,
    projector: WorldStateProjector,
    store: WorldStateStore,
    operator_store: OperatorTestStore | None = None,
) -> APIRouter:
    router = APIRouter()

    def capture(
        payload: Mapping[str, Any],
        *,
        kind: str,
        run_id: str | None = None,
        subject_id: str | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        return store.capture_query_snapshot(
            payload,
            kind=kind,
            run_id=run_id,
            subject_id=subject_id,
            persist=persist,
        )

    @router.get("/world-state")
    def world_state(
        includeEvidence: bool = Query(default=False),
        includeTracks: bool = Query(default=True),
        maxAgeMilliseconds: int | None = Query(default=None, ge=0),
    ) -> dict[str, Any]:
        payload = projector.snapshot()
        if not includeTracks:
            for visit in payload["visits"]:
                visit.pop("currentTracks", None)
        if not includeEvidence:
            for visit in payload["visits"]:
                visit.pop("faceIdentityIds", None)
        if maxAgeMilliseconds is not None:
            payload["requestedMaxAgeMilliseconds"] = maxAgeMilliseconds
        return payload

    @router.get("/world-state/visits/{visit_id}")
    def world_state_visit(visit_id: int) -> dict[str, Any]:
        snapshot = projector.snapshot()
        visit = _entity(snapshot["visits"], "visitId", visit_id)
        return {
                "schemaVersion": snapshot["schemaVersion"],
                "revision": snapshot["revision"],
                "processInstanceId": snapshot["processInstanceId"],
                "generatedAtUnixMilliseconds": snapshot["generatedAtUnixMilliseconds"],
                "visit": visit,
                "claims": _visit_claims(visit),
            }

    @router.get("/world-state/visits/{visit_id}/shelf-position")
    def world_state_visit_shelf_position(visit_id: int) -> dict[str, Any]:
        snapshot = projector.snapshot()
        visit = _entity(snapshot["visits"], "visitId", visit_id)
        position = visit.get("shelfPosition") or visit.get("nearestShelfCandidate")
        return {
            "schemaVersion": snapshot["schemaVersion"],
            "revision": snapshot["revision"],
            "processInstanceId": snapshot["processInstanceId"],
            "generatedAtUnixMilliseconds": snapshot["generatedAtUnixMilliseconds"],
            "visitId": visit_id,
            "customerId": visit.get("customerId"),
            "position": position,
            "candidateCount": len(visit.get("shelfCandidates", [])),
            "measurementCount": len(visit.get("shelfMeasurements", [])),
            "measurements": visit.get("shelfMeasurements", []),
            "freshness": "unknown"
            if position is None
            else position.get("freshness", "unknown"),
        }

    @router.get("/world-state/shelves/{shelf_id}")
    def world_state_shelf(shelf_id: int) -> dict[str, Any]:
        snapshot = projector.snapshot()
        shelf = _entity(snapshot["shelves"], "shelfId", shelf_id)
        return {
                "schemaVersion": snapshot["schemaVersion"],
                "revision": snapshot["revision"],
                "processInstanceId": snapshot["processInstanceId"],
                "generatedAtUnixMilliseconds": snapshot["generatedAtUnixMilliseconds"],
                "shelf": shelf,
            }

    @router.get("/world-state/cameras/{camera_index}")
    def world_state_camera(camera_index: int) -> dict[str, Any]:
        snapshot = projector.snapshot()
        camera = _entity(snapshot["cameras"], "cameraIndex", camera_index)
        return {
                "schemaVersion": snapshot["schemaVersion"],
                "revision": snapshot["revision"],
                "processInstanceId": snapshot["processInstanceId"],
                "generatedAtUnixMilliseconds": snapshot["generatedAtUnixMilliseconds"],
                "camera": camera,
            }

    @router.get("/world-state/revisions/{revision}")
    def world_state_revision(revision: int) -> dict[str, Any]:
        payload = store.revision_snapshot(revision)
        if payload is None:
            current = projector.snapshot()
            if current["revision"] == revision:
                return capture(current, kind="shop")
            raise HTTPException(status_code=404, detail=f"Unknown world-state revision: {revision}")
        return payload

    if operator_store is not None:
        @router.get("/operator/api/test-runs/{run_id}/world-state")
        def test_run_world_state(run_id: str) -> dict[str, Any]:
            if operator_store.get_run(run_id) is None:
                raise HTTPException(status_code=404, detail=f"Unknown test run: {run_id}")
            return projector.snapshot()

        @router.get("/operator/api/test-runs/{run_id}/subjects/{subject_id}/world-state")
        def test_subject_world_state(
            run_id: str,
            subject_id: str,
            captureQuery: bool = Query(default=False),
            persistQuery: bool = Query(default=True),
            expectedRevision: int | None = Query(default=None, ge=0),
        ) -> dict[str, Any]:
            payload = subject_state_payload(
                snapshot=projector.snapshot(),
                operator_store=operator_store,
                run_id=run_id,
                subject_id=subject_id,
            )
            if expectedRevision is not None and payload["revision"] != expectedRevision:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"World state advanced from revision {expectedRevision} "
                        f"to {payload['revision']}; review the latest state first."
                    ),
                )
            if not captureQuery:
                return payload
            captured = capture(
                payload,
                kind="subject",
                run_id=run_id,
                subject_id=subject_id,
                persist=persistQuery,
            )
            captured["worldStateRef"] = {
                "snapshotId": captured["snapshotId"],
                "revision": captured["revision"],
                "processInstanceId": captured["processInstanceId"],
                "queriedAtUnixMilliseconds": captured["generatedAtUnixMilliseconds"],
            }
            return captured

    return router
