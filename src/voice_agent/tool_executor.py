from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from voice_agent.event_bridge import VOICE_EVENT_TYPES, VoiceEventBridge, event_summary
from voice_agent.operator_client import OperatorApiClient, OperatorApiError


class VoiceToolError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def payload(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": self.message}}


@dataclass(frozen=True)
class VoiceToolResult:
    payload: dict[str, Any]
    annotation_id: int | None = None
    resolved_event_id: int | None = None


_CAMERA_FIELD_NAMES = {
    "cameraIndex": "cameraNumber",
    "lastCameraIndex": "lastCameraNumber",
    "sourceCameraIndex": "sourceCameraNumber",
}

_SHELF_STATE_CLAIMS = {
    "shelfPositionId",
}


def _compact_subject_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Keep conversational state bounded while preserving every spoken claim."""
    compact = {
        key: state.get(key)
        for key in (
            "schemaVersion",
            "revision",
            "processInstanceId",
            "generatedAtUnixMilliseconds",
            "runId",
            "subjectId",
            "resolution",
            "claims",
            "freshness",
        )
        if key in state
    }
    visit = state.get("visit")
    if isinstance(visit, Mapping):
        compact["visit"] = {
            key: visit.get(key)
            for key in (
                "visitId",
                "status",
                "origin",
                "customerId",
                "customerBindingStatus",
                "freshness",
                "shelfPosition",
                "observedAtUnixMilliseconds",
                "lastCameraIndex",
                "lastTrackId",
            )
            if key in visit
        }
    return compact


def _camera_numbers_for_voice(value: Any, *, camera_object: bool = False) -> Any:
    """Convert internal zero-based camera indexes to human one-based numbers."""
    if isinstance(value, list):
        return [_camera_numbers_for_voice(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in _CAMERA_FIELD_NAMES and isinstance(item, int) and not isinstance(item, bool):
            result[_CAMERA_FIELD_NAMES[key]] = item + 1
        elif key == "visibleOnCameraIndexes" and isinstance(item, list):
            result["visibleOnCameraNumbers"] = [
                index + 1
                for index in item
                if isinstance(index, int) and not isinstance(index, bool)
            ]
        elif camera_object and key == "id" and isinstance(item, int) and not isinstance(item, bool):
            result["cameraNumber"] = item + 1
        else:
            result[key] = _camera_numbers_for_voice(
                item,
                camera_object=(key == "camera"),
            )
    return result


class VoiceToolExecutor:
    def __init__(
        self,
        client: OperatorApiClient,
        bridge: VoiceEventBridge,
        *,
        expected_run_id: str | None = None,
    ) -> None:
        self.client = client
        self.bridge = bridge
        self.expected_run_id = expected_run_id
        self._last_state_claim: dict[str, Any] | None = None

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> VoiceToolResult:
        handlers = {
            "confirm_system_event": self._confirm_system_event,
            "reject_system_event": self._reject_system_event,
            "report_missing_entry": self._report_missing_entry,
            "report_missing_leave": self._report_missing_leave,
            "get_current_shop_state": self._get_current_shop_state,
            "get_world_state": self._get_world_state,
            "get_subject_state": self._get_subject_state,
            "get_visit_state": self._get_visit_state,
            "get_shelf_state": self._get_shelf_state,
            "get_camera_state": self._get_camera_state,
            "confirm_last_state_claim": self._confirm_last_state_claim,
            "correct_last_state_claim": self._correct_last_state_claim,
            "record_physical_subject_state": self._record_physical_subject_state,
            "confirm_subject_visit_mapping": self._confirm_subject_visit_mapping,
            "repeat_pending_event": self._repeat_pending_event,
            "skip_pending_event": self._skip_pending_event,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            raise VoiceToolError("unsupported_tool", f"Unsupported tool: {tool_name}")
        try:
            return handler(arguments)
        except OperatorApiError as exc:
            raise VoiceToolError("operator_api_error", exc.detail) from exc

    def _active_context(self) -> tuple[str, str, dict[str, Any]]:
        state = self.client.state()
        run = state.get("activeRun")
        if not isinstance(run, Mapping):
            raise VoiceToolError("no_active_run", "No operator test run is active.")
        run_id = str(run["runId"])
        if self.expected_run_id is not None and run_id != self.expected_run_id:
            raise VoiceToolError(
                "run_changed",
                "The test run bound to this voice session is no longer active.",
            )
        context = self.client.voice_context(run_id)
        subjects = context.get("subjects")
        if not isinstance(subjects, list) or not subjects:
            raise VoiceToolError("no_subject", "The active run has no test subject.")
        subject = subjects[0]
        if not isinstance(subject, Mapping) or not subject.get("subjectId"):
            raise VoiceToolError("no_subject", "The active run has no valid test subject.")
        return run_id, str(subject["subjectId"]), context

    def _event_and_context(
        self,
        arguments: Mapping[str, Any],
    ) -> tuple[int, dict[str, Any], str, str, dict[str, Any]]:
        event_id = arguments.get("event_id")
        if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id <= 0:
            raise VoiceToolError("invalid_event_id", "event_id must be a positive integer.")
        run_id, subject_id, context = self._active_context()
        event = next(
            (
                item
                for item in context.get("events", [])
                if isinstance(item, Mapping) and item.get("eventId") == event_id
            ),
            None,
        )
        if event is None:
            raise VoiceToolError("unknown_event", f"Event {event_id} is not in the active run.")
        if event.get("eventType") not in VOICE_EVENT_TYPES:
            raise VoiceToolError("unsupported_event", "That event type cannot receive a voice verdict.")
        return event_id, dict(event), run_id, subject_id, context

    def _write_verdict(
        self,
        arguments: Mapping[str, Any],
        *,
        correct: bool,
    ) -> VoiceToolResult:
        event_id, event, run_id, subject_id, context = self._event_and_context(arguments)
        verdicts = context.get("verdicts", {})
        existing = verdicts.get(str(event_id)) if isinstance(verdicts, Mapping) else None
        expected_type = "system_event_correct" if correct else "system_event_incorrect"
        if isinstance(existing, Mapping):
            if existing.get("annotationType") != expected_type:
                raise VoiceToolError(
                    "conflicting_verdict",
                    f"Event {event_id} already has the opposite verdict.",
                )
            self.bridge.resolve(event_id)
            return VoiceToolResult(
                {
                    "ok": True,
                    "idempotent": True,
                    "event": event,
                    "annotation": dict(existing),
                },
                annotation_id=int(existing["annotationId"]),
                resolved_event_id=event_id,
            )
        payload: dict[str, Any] = {
            "annotationType": expected_type,
            "subjectId": subject_id,
            "systemEventId": event_id,
            "systemEventType": event["eventType"],
            "systemEventVisitId": event.get("visitId"),
            "systemEventPayload": event.get("payload", {}),
        }
        reason = arguments.get("reason")
        if not correct and isinstance(reason, str) and reason.strip():
            payload["reason"] = reason.strip()[:240]
        response = self.client.create_annotation(run_id, payload)
        annotation = response["annotation"]
        self.bridge.resolve(event_id)
        return VoiceToolResult(
            {"ok": True, "event": event, "annotation": annotation},
            annotation_id=int(annotation["annotationId"]),
            resolved_event_id=event_id,
        )

    def _confirm_system_event(self, arguments: Mapping[str, Any]) -> VoiceToolResult:
        return self._write_verdict(arguments, correct=True)

    def _reject_system_event(self, arguments: Mapping[str, Any]) -> VoiceToolResult:
        return self._write_verdict(arguments, correct=False)

    def _create_physical_annotation(
        self,
        annotation_type: str,
        extra: Mapping[str, Any] | None = None,
    ) -> VoiceToolResult:
        run_id, subject_id, _context = self._active_context()
        payload = {"annotationType": annotation_type, "subjectId": subject_id}
        if extra:
            payload.update(extra)
        shelf_id = payload.get("shelfId")
        if isinstance(shelf_id, int) and not isinstance(shelf_id, bool):
            reference, diagnostic = self._shelf_feedback_evidence(
                run_id=run_id,
                subject_id=subject_id,
                shelf_id=shelf_id,
            )
            if reference is not None:
                payload["observationRef"] = reference
            if diagnostic is not None:
                payload["systemShelfDiagnostic"] = diagnostic
        response = self.client.create_annotation(run_id, payload)
        annotation = response["annotation"]
        return VoiceToolResult(
            {"ok": True, "annotation": annotation},
            annotation_id=int(annotation["annotationId"]),
        )

    def _shelf_feedback_evidence(
        self,
        *,
        run_id: str,
        subject_id: str,
        shelf_id: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Resolve the system shelf candidate to an exact current observation."""
        try:
            subject_state = self.client.subject_world_state(run_id, subject_id)
        except OperatorApiError:
            return None, None
        visit = subject_state.get("visit")
        if not isinstance(visit, Mapping):
            return None, None
        candidates = visit.get("shelfCandidates")
        if not isinstance(candidates, list):
            return None, None
        candidate = next(
            (
                item
                for item in candidates
                if isinstance(item, Mapping) and item.get("shelfId") == shelf_id
            ),
            None,
        )
        if candidate is None:
            return None, None
        diagnostic = dict(candidate)
        camera_index = candidate.get("cameraIndex")
        track_id = candidate.get("trackId")
        if (
            isinstance(camera_index, bool)
            or not isinstance(camera_index, int)
            or isinstance(track_id, bool)
            or not isinstance(track_id, int)
        ):
            return None, diagnostic
        try:
            snapshot = self.client.observation(camera_index)
        except OperatorApiError:
            return None, diagnostic
        camera = snapshot.get("camera")
        frame = snapshot.get("frame")
        observations = snapshot.get("observations")
        if (
            not isinstance(camera, Mapping)
            or not isinstance(frame, Mapping)
            or not isinstance(observations, list)
        ):
            return None, diagnostic
        person = next(
            (
                item
                for item in observations
                if isinstance(item, Mapping) and item.get("trackId") == track_id
            ),
            None,
        )
        if person is None:
            return None, diagnostic
        device_id = camera.get("deviceId")
        rgb_sequence = frame.get("rgbSequenceNumber")
        host_seconds = frame.get("hostSyncedSeconds")
        if (
            not isinstance(device_id, str)
            or isinstance(rgb_sequence, bool)
            or not isinstance(rgb_sequence, int)
            or isinstance(host_seconds, bool)
            or not isinstance(host_seconds, (int, float))
        ):
            return None, diagnostic
        return (
            {
                "cameraIndex": camera_index,
                "deviceId": device_id,
                "rgbSequenceNumber": rgb_sequence,
                "hostSyncedSeconds": float(host_seconds),
                "trackId": track_id,
                "observedVisitId": person.get("visitId"),
                "observedCustomerId": person.get("customerId"),
            },
            diagnostic,
        )

    def _report_missing_entry(self, _arguments: Mapping[str, Any]) -> VoiceToolResult:
        return self._create_physical_annotation("physical_entry")

    def _report_missing_leave(self, _arguments: Mapping[str, Any]) -> VoiceToolResult:
        return self._create_physical_annotation("physical_leave")

    def _validated_shelf_id(self, arguments: Mapping[str, Any]) -> int:
        shelf_id = arguments.get("shelf_id")
        if isinstance(shelf_id, bool) or not isinstance(shelf_id, int) or shelf_id <= 0:
            raise VoiceToolError("invalid_shelf_id", "shelf_id must be a positive integer.")
        state = self.client.state()
        configured = {
            int(item["shelfId"])
            for item in state.get("shelves", [])
            if isinstance(item, Mapping) and isinstance(item.get("shelfId"), int)
        }
        if shelf_id not in configured:
            raise VoiceToolError("unknown_shelf", f"Shelf {shelf_id} is not configured.")
        return shelf_id

    def _get_current_shop_state(self, arguments: Mapping[str, Any]) -> VoiceToolResult:
        state = self.client.state()
        camera_number = arguments.get("camera_number")
        observation = None
        if camera_number is not None:
            camera_index = self._camera_index(arguments)
            observation = self.client.observation(camera_index)
        return VoiceToolResult(
            {
                "ok": True,
                "activeRun": state.get("activeRun"),
                "cameras": [
                    _camera_numbers_for_voice(camera, camera_object=True)
                    for camera in state.get("cameras", [])
                ],
                "visits": _camera_numbers_for_voice(state.get("visits", [])),
                "shelves": _camera_numbers_for_voice(state.get("shelves", [])),
                "observation": _camera_numbers_for_voice(observation),
                "pendingEvent": _camera_numbers_for_voice(self.bridge.current_event()),
            }
        )

    def _get_world_state(self, _arguments: Mapping[str, Any]) -> VoiceToolResult:
        return VoiceToolResult(
            {"ok": True, "worldState": _camera_numbers_for_voice(self.client.world_state())}
        )

    def _get_subject_state(self, arguments: Mapping[str, Any]) -> VoiceToolResult:
        run_id, subject_id, _context = self._active_context()
        state = self.client.subject_world_state(run_id, subject_id)
        claim = arguments.get("claim")
        claims = state.get("claims")
        canonical_claim = (
            "visibleOnCameraIndexes"
            if claim == "visibleOnCameraNumbers"
            else claim
        )
        if claim is not None:
            if (
                not isinstance(claim, str)
                or not isinstance(canonical_claim, str)
                or not isinstance(claims, Mapping)
                or canonical_claim not in claims
            ):
                raise VoiceToolError("unknown_claim", f"Unknown subject-state claim: {claim}")
            self._last_state_claim = {
                "runId": run_id,
                "subjectId": subject_id,
                "worldStateRef": state["worldStateRef"],
                "claim": canonical_claim,
                "systemValue": claims[canonical_claim],
            }
        presented_state = _camera_numbers_for_voice(_compact_subject_state(state))
        presented_claims = presented_state.get("claims", {})
        return VoiceToolResult(
            {
                "ok": True,
                "subjectState": presented_state,
                "selectedClaim": None if claim is None else {
                    "claim": claim,
                    "systemValue": presented_claims[claim],
                },
            }
        )

    @staticmethod
    def _positive_id(arguments: Mapping[str, Any], name: str) -> int:
        value = arguments.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise VoiceToolError(f"invalid_{name}", f"{name} must be a non-negative integer.")
        return value

    def _get_visit_state(self, arguments: Mapping[str, Any]) -> VoiceToolResult:
        visit_id = self._positive_id(arguments, "visit_id")
        if visit_id == 0:
            raise VoiceToolError("invalid_visit_id", "visit_id must be positive.")
        return VoiceToolResult(
            {
                "ok": True,
                "visitState": _camera_numbers_for_voice(
                    self.client.visit_world_state(visit_id)
                ),
            }
        )

    def _get_shelf_state(self, arguments: Mapping[str, Any]) -> VoiceToolResult:
        shelf_id = self._positive_id(arguments, "shelf_id")
        if shelf_id == 0:
            raise VoiceToolError("invalid_shelf_id", "shelf_id must be positive.")
        return VoiceToolResult(
            {
                "ok": True,
                "shelfState": _camera_numbers_for_voice(
                    self.client.shelf_world_state(shelf_id)
                ),
            }
        )

    @staticmethod
    def _camera_index(arguments: Mapping[str, Any]) -> int:
        camera_number = arguments.get("camera_number")
        if (
            isinstance(camera_number, bool)
            or not isinstance(camera_number, int)
            or camera_number <= 0
        ):
            raise VoiceToolError(
                "invalid_camera_number",
                "camera_number must be a positive one-based camera number.",
            )
        return camera_number - 1

    def _get_camera_state(self, arguments: Mapping[str, Any]) -> VoiceToolResult:
        camera_index = self._camera_index(arguments)
        return VoiceToolResult(
            {
                "ok": True,
                "cameraState": _camera_numbers_for_voice(
                    self.client.camera_world_state(camera_index)
                ),
            }
        )

    def _state_claim_annotation(
        self,
        *,
        correct: bool,
        physical_value: Any = None,
        reason: str | None = None,
    ) -> VoiceToolResult:
        claim = self._last_state_claim
        if claim is None:
            raise VoiceToolError(
                "no_state_claim",
                "No state claim has been queried in this voice session.",
            )
        payload = {
            "annotationType": (
                "world_state_claim_correct" if correct else "world_state_claim_incorrect"
            ),
            "subjectId": claim["subjectId"],
            "worldStateRef": claim["worldStateRef"],
            "claim": claim["claim"],
            "systemValue": claim["systemValue"],
        }
        if not correct:
            payload["physicalValue"] = physical_value
            if reason:
                payload["reason"] = reason[:240]
        shelf_value = claim["systemValue"] if correct else physical_value
        if (
            claim["claim"] in _SHELF_STATE_CLAIMS
            and isinstance(shelf_value, int)
            and not isinstance(shelf_value, bool)
            and shelf_value > 0
        ):
            payload["shelfId"] = shelf_value
            reference, diagnostic = self._shelf_feedback_evidence(
                run_id=claim["runId"],
                subject_id=claim["subjectId"],
                shelf_id=shelf_value,
            )
            if reference is not None:
                payload["observationRef"] = reference
            if diagnostic is not None:
                payload["systemShelfDiagnostic"] = diagnostic
        response = self.client.create_annotation(claim["runId"], payload)
        annotation = response["annotation"]
        self._last_state_claim = None
        return VoiceToolResult(
            {"ok": True, "annotation": annotation},
            annotation_id=int(annotation["annotationId"]),
        )

    def _confirm_last_state_claim(self, _arguments: Mapping[str, Any]) -> VoiceToolResult:
        return self._state_claim_annotation(correct=True)

    def _correct_last_state_claim(self, arguments: Mapping[str, Any]) -> VoiceToolResult:
        if "physical_value" not in arguments:
            raise VoiceToolError("missing_physical_value", "physical_value is required.")
        return self._state_claim_annotation(
            correct=False,
            physical_value=arguments["physical_value"],
            reason=(
                None
                if not isinstance(arguments.get("reason"), str)
                else str(arguments["reason"]).strip()
            ),
        )

    def _record_physical_subject_state(self, arguments: Mapping[str, Any]) -> VoiceToolResult:
        claim = arguments.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise VoiceToolError("invalid_claim", "claim is required.")
        if "physical_value" not in arguments:
            raise VoiceToolError("missing_physical_value", "physical_value is required.")
        physical_value = arguments["physical_value"]
        extra = {"claim": claim, "physicalValue": physical_value}
        normalized_claim = claim.strip().lower().replace("_", "")
        is_shelf_fact = normalized_claim in {
            "shelf",
            "shelfid",
            "shelfpositionid",
        }
        if is_shelf_fact:
            if (
                isinstance(physical_value, bool)
                or not isinstance(physical_value, int)
                or physical_value <= 0
            ):
                raise VoiceToolError(
                    "invalid_shelf_id",
                    "A physical shelf fact requires a positive integer physical_value.",
                )
            supplied_shelf_id = arguments.get("shelf_id", physical_value)
            shelf_id = self._validated_shelf_id({"shelf_id": supplied_shelf_id})
            if shelf_id != physical_value:
                raise VoiceToolError(
                    "shelf_value_mismatch",
                    "physical_value and shelf_id must identify the same shelf.",
                )
            extra["shelfId"] = shelf_id

            pending_claim = self._last_state_claim
            if pending_claim is not None and pending_claim["claim"] in _SHELF_STATE_CLAIMS:
                return self._state_claim_annotation(
                    correct=pending_claim["systemValue"] == shelf_id,
                    physical_value=shelf_id,
                    reason=f"Operator reported physical shelf {shelf_id}.",
                )
        elif "shelf_id" in arguments:
            extra["shelfId"] = self._validated_shelf_id(arguments)
        return self._create_physical_annotation(
            "physical_subject_state",
            extra,
        )

    def _confirm_subject_visit_mapping(self, arguments: Mapping[str, Any]) -> VoiceToolResult:
        visit_id = self._positive_id(arguments, "visit_id")
        if visit_id == 0:
            raise VoiceToolError("invalid_visit_id", "visit_id must be positive.")
        return self._create_physical_annotation(
            "subject_visit_mapping",
            {"visitId": visit_id},
        )

    def _repeat_pending_event(self, _arguments: Mapping[str, Any]) -> VoiceToolResult:
        event = self.bridge.current_event()
        if event is None:
            event = self.bridge.next_event()
        return VoiceToolResult(
            {
                "ok": True,
                "event": event,
                "summary": "No event is pending." if event is None else event_summary(event),
            }
        )

    def _skip_pending_event(self, _arguments: Mapping[str, Any]) -> VoiceToolResult:
        event = self.bridge.skip_current()
        return VoiceToolResult(
            {"ok": True, "skippedEvent": event},
            resolved_event_id=None if event is None else int(event["eventId"]),
        )


def realtime_tool_definitions() -> list[dict[str, Any]]:
    no_arguments = {"type": "object", "properties": {}, "additionalProperties": False}
    subject_claims = [
        "visitId",
        "inside",
        "status",
        "entranceConfirmed",
        "customerId",
        "customerBindingStatus",
        "visibility",
        "visibleOnCameraNumbers",
        "shelfPositionId",
        "shelfPositionDistanceMm",
        "shelfPositionFreshness",
        "freshness",
    ]
    return [
        {
            "type": "function",
            "name": "confirm_system_event",
            "description": "Confirm that a generated shop event matches physical reality.",
            "parameters": {
                "type": "object",
                "properties": {"event_id": {"type": "integer", "minimum": 1}},
                "required": ["event_id"],
                "additionalProperties": False,
            },
        },
        {"type": "function", "name": "get_world_state", "description": "Read the current persisted system-believed shop state.", "parameters": no_arguments},
        {
            "type": "function",
            "name": "get_subject_state",
            "description": "Read the current state for the human test subject. Set claim when answering a specific question so later feedback is revision-bound.",
            "parameters": {
                "type": "object",
                "properties": {"claim": {"type": "string", "enum": subject_claims}},
                "additionalProperties": False,
            },
        },
        {
            "type": "function", "name": "get_visit_state", "description": "Read one system visit.",
            "parameters": {"type": "object", "properties": {"visit_id": {"type": "integer", "minimum": 1}}, "required": ["visit_id"], "additionalProperties": False},
        },
        {
            "type": "function", "name": "get_shelf_state", "description": "Read one shelf and its current nearby visits.",
            "parameters": {"type": "object", "properties": {"shelf_id": {"type": "integer", "minimum": 1}}, "required": ["shelf_id"], "additionalProperties": False},
        },
        {
            "type": "function", "name": "get_camera_state", "description": "Read one camera's health and visible tracks. Camera numbers are one-based, matching the operator UI.",
            "parameters": {"type": "object", "properties": {"camera_number": {"type": "integer", "minimum": 1}}, "required": ["camera_number"], "additionalProperties": False},
        },
        {"type": "function", "name": "confirm_last_state_claim", "description": "Confirm that the last specifically queried subject-state claim matches physical reality.", "parameters": no_arguments},
        {
            "type": "function", "name": "correct_last_state_claim", "description": "Correct the last specifically queried subject-state claim.",
            "parameters": {"type": "object", "properties": {"physical_value": {"type": ["string", "number", "integer", "boolean", "null"]}, "reason": {"type": "string"}}, "required": ["physical_value"], "additionalProperties": False},
        },
        {
            "type": "function", "name": "record_physical_subject_state", "description": "Record one physical fact. For 'I am at shelf N', set claim='shelf', physical_value=N, and shelf_id=N. This atomically resolves any pending shelf claim; do not call correct_last_state_claim afterward.",
            "parameters": {"type": "object", "properties": {"claim": {"type": "string"}, "physical_value": {"type": ["string", "number", "integer", "boolean", "null"]}, "shelf_id": {"type": "integer", "minimum": 1}}, "required": ["claim", "physical_value"], "additionalProperties": False},
        },
        {
            "type": "function", "name": "confirm_subject_visit_mapping", "description": "Confirm that a proposed system visit represents the physical test subject.",
            "parameters": {"type": "object", "properties": {"visit_id": {"type": "integer", "minimum": 1}}, "required": ["visit_id"], "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "reject_system_event",
            "description": "Mark a generated shop event as physically incorrect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "integer", "minimum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["event_id"],
                "additionalProperties": False,
            },
        },
        {"type": "function", "name": "report_missing_entry", "description": "Record a physical shop entry that the system missed.", "parameters": no_arguments},
        {"type": "function", "name": "report_missing_leave", "description": "Record a physical shop leave that the system missed.", "parameters": no_arguments},
        {
            "type": "function",
            "name": "get_current_shop_state",
            "description": "Read current visits, shelves, cameras, and optionally one camera observation. Camera numbers are one-based, matching the operator UI.",
            "parameters": {
                "type": "object",
                "properties": {"camera_number": {"type": "integer", "minimum": 1}},
                "additionalProperties": False,
            },
        },
        {"type": "function", "name": "repeat_pending_event", "description": "Read the current event awaiting verification.", "parameters": no_arguments},
        {"type": "function", "name": "skip_pending_event", "description": "Skip the current generated event without recording a verdict.", "parameters": no_arguments},
    ]
