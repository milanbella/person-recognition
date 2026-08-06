from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

from fastapi import APIRouter, Body, Header, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse

from pipeline.operator_analysis import analyze_test_run
from pipeline.operator_models import ObservationReference
from pipeline.operator_state import OperatorState
from pipeline.operator_test_store import OperatorTestStore
from pipeline.world_state_store import WorldStateStore


def create_operator_router(
    *,
    state: OperatorState,
    store: OperatorTestStore,
    api_token: str | None,
    world_state_store: WorldStateStore | None,
    assets_root: Path,
) -> APIRouter:
    router = APIRouter()

    def require_mutation_auth(authorization: str | None) -> None:
        if api_token is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Operator mutations are disabled because no operator API "
                    "token is configured."
                ),
            )
        expected = f"Bearer {api_token}"
        if authorization != expected:
            raise HTTPException(
                status_code=401,
                detail="A valid operator bearer token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def current_state() -> dict[str, Any]:
        return state.state_payload(
            active_run=store.active_run(),
            persisted_visits=store.load_visit_states(),
        )

    def analyze(run_id: str) -> dict[str, Any]:
        store.flush_events()
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown test run: {run_id}")
        annotations, events = store.run_rows(run_id)
        report, mappings = analyze_test_run(
            run=run,
            subjects=store.subjects(run_id),
            physical_visits=store.physical_visits(run_id),
            annotations=annotations,
            events=events,
        )
        store.save_analysis(
            run_id,
            [
                _analysis_result_from_payload(item)
                for item in report["results"]
            ],
            physical_visit_mappings=mappings,
        )
        store.export_run(run_id, report=report)
        return report

    @router.get("/operator/")
    def operator_console() -> FileResponse:
        path = assets_root / "index.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Operator console is not installed.")
        return FileResponse(
            path,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/operator/assets/{asset_name}")
    def operator_asset(asset_name: str) -> FileResponse:
        if asset_name not in {"operator.css", "operator.js"}:
            raise HTTPException(status_code=404, detail="Unknown operator asset.")
        path = assets_root / asset_name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Operator asset is not installed.")
        media_type = "text/css" if asset_name.endswith(".css") else "text/javascript"
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/operator/api/state")
    def operator_state() -> dict[str, Any]:
        return current_state()

    @router.get("/operator/api/events")
    def operator_events(
        afterEventId: int = Query(default=0, ge=0),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        requested_id = afterEventId
        if last_event_id not in (None, ""):
            try:
                requested_id = max(requested_id, int(last_event_id))
            except ValueError as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Last-Event-ID must be an integer.",
                ) from exc

        return StreamingResponse(
            _event_stream(state, requested_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post("/operator/api/test-runs")
    def start_test_run(
        payload: dict[str, Any] = Body(...),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_mutation_auth(authorization)
        try:
            run = store.start_run(payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        state.publish_event(
            event_type="test_run_started",
            source="operator",
            payload={"runId": run["runId"], "scenario": run["scenario"]},
        )
        return run

    @router.get("/operator/api/test-runs/active")
    def active_test_run() -> dict[str, Any]:
        run = store.active_run()
        if run is None:
            raise HTTPException(status_code=404, detail="No test run is active.")
        return run

    @router.get("/operator/api/test-runs/{run_id}")
    def get_test_run(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown test run: {run_id}")
        return run

    @router.get("/operator/api/test-runs/{run_id}/voice-context")
    def test_run_voice_context(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown test run: {run_id}")
        store.flush_events()
        annotations, events = store.run_rows(run_id)
        verdicts = {
            int(annotation["payload"]["systemEventId"]): annotation
            for annotation in annotations
            if annotation["annotationType"] in {
                "system_event_correct",
                "system_event_incorrect",
            }
            and isinstance(annotation.get("payload", {}).get("systemEventId"), int)
        }
        return {
            "run": run,
            "subjects": store.subjects(run_id),
            "physicalVisits": store.physical_visits(run_id),
            "events": events,
            "verdicts": {
                str(event_id): annotation
                for event_id, annotation in verdicts.items()
            },
        }

    @router.post("/operator/api/test-runs/{run_id}/stop")
    def stop_test_run(
        run_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_mutation_auth(authorization)
        try:
            run = store.stop_run(run_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        state.publish_event(
            event_type="test_run_stopped",
            source="operator",
            payload={"runId": run_id},
        )
        return {"run": run, "report": analyze(run_id)}

    @router.post("/operator/api/test-runs/{run_id}/analyze")
    def analyze_test_run_route(
        run_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_mutation_auth(authorization)
        return analyze(run_id)

    @router.get("/operator/api/test-runs/{run_id}/report")
    def test_run_report(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown test run: {run_id}")
        results = store.analysis_results(run_id)
        if not results:
            return analyze(run_id)
        counts = {
            status: sum(1 for item in results if item["status"] == status)
            for status in ("pass", "fail", "pending", "inconclusive")
        }
        return {
            "runId": run_id,
            "scenario": run["scenario"],
            "status": (
                "fail"
                if counts["fail"]
                else "pending"
                if counts["pending"]
                else "pass"
            ),
            "summary": counts,
            "results": results,
        }

    @router.post("/operator/api/test-runs/{run_id}/annotations")
    def create_annotation(
        run_id: str,
        payload: dict[str, Any] = Body(...),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_mutation_auth(authorization)
        payload = dict(payload)
        world_query_snapshot: Mapping[str, Any] | None = None
        annotation_type = str(payload.get("annotationType", ""))
        if annotation_type in {
            "world_state_claim_correct",
            "world_state_claim_incorrect",
        }:
            if world_state_store is None:
                raise HTTPException(status_code=503, detail="World-state persistence is disabled.")
            state_ref = payload.get("worldStateRef")
            if not isinstance(state_ref, Mapping):
                raise HTTPException(status_code=422, detail="worldStateRef must be an object.")
            snapshot_id = state_ref.get("snapshotId")
            if not isinstance(snapshot_id, str) or not snapshot_id:
                raise HTTPException(status_code=422, detail="worldStateRef.snapshotId is required.")
            world_query_snapshot = world_state_store.query_snapshot(snapshot_id)
            if world_query_snapshot is None:
                raise HTTPException(status_code=409, detail="The referenced world-state query is unavailable.")
            if (
                state_ref.get("revision") != world_query_snapshot.get("revision")
                or state_ref.get("processInstanceId")
                != world_query_snapshot.get("processInstanceId")
            ):
                raise HTTPException(status_code=409, detail="The world-state reference does not match its snapshot.")
            claim = payload.get("claim")
            claims = world_query_snapshot.get("claims")
            if not isinstance(claim, str) or not isinstance(claims, Mapping) or claim not in claims:
                raise HTTPException(status_code=422, detail="The requested claim is not present in the referenced snapshot.")
            authoritative_value = claims[claim]
            if "systemValue" in payload and payload["systemValue"] != authoritative_value:
                raise HTTPException(status_code=409, detail="systemValue does not match the referenced world-state claim.")
            payload["systemValue"] = authoritative_value
            payload["worldStateRef"] = {
                "snapshotId": snapshot_id,
                "revision": int(world_query_snapshot["revision"]),
                "processInstanceId": str(world_query_snapshot["processInstanceId"]),
                "queriedAtUnixMilliseconds": int(
                    world_query_snapshot["generatedAtUnixMilliseconds"]
                ),
            }
        reference_payload = payload.get("observationRef")
        reference: ObservationReference | None = None
        observation_snapshot: Mapping[str, Any] | None = None
        if reference_payload is not None:
            if not isinstance(reference_payload, Mapping):
                raise HTTPException(
                    status_code=422,
                    detail="observationRef must be an object.",
                )
            try:
                reference = ObservationReference.from_payload(reference_payload)
                observation_snapshot = state.resolve_observation(reference)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except LookupError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        snapshot = current_state()
        if world_query_snapshot is not None:
            snapshot["worldStateQuery"] = dict(world_query_snapshot)
        if observation_snapshot is not None:
            snapshot["selectedObservation"] = dict(observation_snapshot)
        try:
            annotation = store.create_annotation(
                run_id,
                payload,
                system_snapshot=snapshot,
                observation_reference=reference,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        evidence = (
            None
            if reference is None
            else state.latest_evidence(reference.camera_index)
        )
        if evidence is not None:
            jpeg, stream_sequence, rgb_sequence, captured_at_ms = evidence
            evidence_name = (
                f"annotation-{annotation['annotationId']:06d}-"
                f"camera-{reference.camera_index}.jpg"
            )
            evidence_path = store.evidence_directory(run_id) / evidence_name
            evidence_path.write_bytes(jpeg)
            store.update_annotation_evidence(
                int(annotation["annotationId"]),
                evidence_path=evidence_path,
            )
            annotation["evidence"] = {
                "path": str(evidence_path),
                "streamSequence": stream_sequence,
                "rgbSequenceNumber": rgb_sequence,
                "capturedAtUnixMilliseconds": captured_at_ms,
                "exactFrame": rgb_sequence == reference.rgb_sequence_number,
            }

        state.publish_event(
            event_type="human_annotation_created",
            source="operator",
            payload=annotation,
            visit_id=(
                None if reference is None else reference.observed_visit_id
            ),
            camera_index=None if reference is None else reference.camera_index,
            device_id=None if reference is None else reference.device_id,
            rgb_sequence_number=(
                None if reference is None else reference.rgb_sequence_number
            ),
            track_id=None if reference is None else reference.track_id,
        )
        report = analyze(run_id)
        return {"annotation": annotation, "report": report}

    @router.get(
        "/operator/api/observations/{camera_index}/{rgb_sequence_number}/snapshot"
    )
    def observation_snapshot(
        camera_index: int,
        rgb_sequence_number: int,
    ) -> Response:
        evidence = state.latest_evidence(camera_index)
        if evidence is None:
            raise HTTPException(status_code=404, detail="No camera JPEG is available.")
        jpeg, _stream_sequence, available_rgb_sequence, _captured_at = evidence
        headers = {
            "Cache-Control": "no-store",
            "X-Requested-RGB-Sequence": str(rgb_sequence_number),
            "X-Available-RGB-Sequence": (
                "unknown"
                if available_rgb_sequence is None
                else str(available_rgb_sequence)
            ),
        }
        return Response(jpeg, media_type="image/jpeg", headers=headers)

    return router


def _event_stream(state: OperatorState, after_event_id: int) -> Iterator[str]:
    current_id = after_event_id
    while True:
        events, resync_required = state.wait_for_events(
            current_id,
            timeout_seconds=15.0,
        )
        if resync_required:
            payload = json.dumps(
                {
                    "eventType": "resync_required",
                    "lastEventId": state.last_event_id,
                },
                separators=(",", ":"),
            )
            yield f"event: resync_required\ndata: {payload}\n\n"
            current_id = state.last_event_id
            continue
        if not events:
            yield f": heartbeat {time.time_ns() // 1_000_000}\n\n"
            continue
        for event in events:
            payload = json.dumps(event.as_payload(), separators=(",", ":"))
            yield (
                f"id: {event.event_id}\n"
                f"event: {event.event_type}\n"
                f"data: {payload}\n\n"
            )
            current_id = event.event_id


def _analysis_result_from_payload(payload: Mapping[str, Any]) -> Any:
    from pipeline.operator_models import AnalysisResult

    return AnalysisResult(
        rule_id=str(payload["ruleId"]),
        status=str(payload["status"]),
        summary=str(payload["summary"]),
        expected_annotation_id=payload.get("expectedAnnotationId"),
        matched_event_id=payload.get("matchedEventId"),
        latency_milliseconds=payload.get("latencyMilliseconds"),
        details=dict(payload.get("details", {})),
    )
