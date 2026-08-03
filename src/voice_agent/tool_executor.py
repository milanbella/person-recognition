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
            "report_missing_shelf_approach": self._report_missing_shelf_approach,
            "report_missing_shelf_leave": self._report_missing_shelf_leave,
            "get_current_shop_state": self._get_current_shop_state,
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
        response = self.client.create_annotation(run_id, payload)
        annotation = response["annotation"]
        return VoiceToolResult(
            {"ok": True, "annotation": annotation},
            annotation_id=int(annotation["annotationId"]),
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

    def _report_missing_shelf_approach(self, arguments: Mapping[str, Any]) -> VoiceToolResult:
        return self._create_physical_annotation(
            "shelf_approach",
            {"shelfId": self._validated_shelf_id(arguments)},
        )

    def _report_missing_shelf_leave(self, arguments: Mapping[str, Any]) -> VoiceToolResult:
        return self._create_physical_annotation(
            "shelf_departure",
            {"shelfId": self._validated_shelf_id(arguments)},
        )

    def _get_current_shop_state(self, arguments: Mapping[str, Any]) -> VoiceToolResult:
        state = self.client.state()
        camera_index = arguments.get("camera_index")
        observation = None
        if camera_index is not None:
            if isinstance(camera_index, bool) or not isinstance(camera_index, int) or camera_index < 0:
                raise VoiceToolError("invalid_camera", "camera_index must be a non-negative integer.")
            observation = self.client.observation(camera_index)
        return VoiceToolResult(
            {
                "ok": True,
                "activeRun": state.get("activeRun"),
                "cameras": state.get("cameras", []),
                "visits": state.get("visits", []),
                "shelves": state.get("shelves", []),
                "observation": observation,
                "pendingEvent": self.bridge.current_event(),
            }
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
            "name": "report_missing_shelf_approach",
            "description": "Record a physical shelf approach that the system missed.",
            "parameters": {
                "type": "object",
                "properties": {"shelf_id": {"type": "integer", "minimum": 1}},
                "required": ["shelf_id"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "report_missing_shelf_leave",
            "description": "Record physically leaving a shelf when the system missed it.",
            "parameters": {
                "type": "object",
                "properties": {"shelf_id": {"type": "integer", "minimum": 1}},
                "required": ["shelf_id"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_current_shop_state",
            "description": "Read current visits, shelves, cameras, and optionally one camera observation.",
            "parameters": {
                "type": "object",
                "properties": {"camera_index": {"type": "integer", "minimum": 0}},
                "additionalProperties": False,
            },
        },
        {"type": "function", "name": "repeat_pending_event", "description": "Read the current event awaiting verification.", "parameters": no_arguments},
        {"type": "function", "name": "skip_pending_event", "description": "Skip the current generated event without recording a verdict.", "parameters": no_arguments},
    ]
