from __future__ import annotations

from collections import deque
from typing import Any, Mapping


VOICE_EVENT_TYPES = {
    "entry_accepted",
    "leave_accepted",
    "shelf_approach",
    "shelf_departure",
}


class VoiceEventBridge:
    """Maintains the ordered set of generated events awaiting human verdicts."""

    def __init__(self, *, max_queued_events: int = 100) -> None:
        self.max_queued_events = max_queued_events
        self._events: dict[int, dict[str, Any]] = {}
        self._queued_ids: deque[int] = deque()
        self._resolved_ids: set[int] = set()
        self._skipped_ids: set[int] = set()
        self.current_event_id: int | None = None

    def refresh(self, context: Mapping[str, Any]) -> None:
        verdicts = context.get("verdicts", {})
        if isinstance(verdicts, Mapping):
            self._resolved_ids.update(
                int(event_id)
                for event_id in verdicts
                if str(event_id).isdigit()
            )
        events = context.get("events", [])
        if not isinstance(events, list):
            return
        for event in sorted(
            (item for item in events if isinstance(item, Mapping)),
            key=lambda item: (
                int(item.get("occurredAtUnixMilliseconds", 0)),
                int(item.get("eventId", 0)),
            ),
        ):
            event_id = event.get("eventId")
            if (
                isinstance(event_id, bool)
                or not isinstance(event_id, int)
                or event_id <= 0
                or event.get("eventType") not in VOICE_EVENT_TYPES
            ):
                continue
            self._events[event_id] = dict(event)
            if (
                event_id not in self._resolved_ids
                and event_id not in self._skipped_ids
                and event_id != self.current_event_id
                and event_id not in self._queued_ids
            ):
                self._queued_ids.append(event_id)
        while len(self._queued_ids) > self.max_queued_events:
            self._queued_ids.popleft()

    def next_event(self) -> dict[str, Any] | None:
        if self.current_event_id is not None:
            return self._events.get(self.current_event_id)
        while self._queued_ids:
            event_id = self._queued_ids.popleft()
            if event_id in self._resolved_ids or event_id in self._skipped_ids:
                continue
            self.current_event_id = event_id
            return self._events.get(event_id)
        return None

    def current_event(self) -> dict[str, Any] | None:
        if self.current_event_id is None:
            return None
        return self._events.get(self.current_event_id)

    def resolve(self, event_id: int) -> None:
        self._resolved_ids.add(event_id)
        if self.current_event_id == event_id:
            self.current_event_id = None

    def skip_current(self) -> dict[str, Any] | None:
        event = self.current_event()
        if event is None:
            return None
        event_id = int(event["eventId"])
        self._skipped_ids.add(event_id)
        self.current_event_id = None
        return event

    @property
    def queued_count(self) -> int:
        return len(self._queued_ids) + (1 if self.current_event_id is not None else 0)


def event_summary(event: Mapping[str, Any]) -> str:
    event_type = str(event.get("eventType", "event"))
    visit_id = event.get("visitId")
    camera_index = event.get("cameraIndex")
    camera = "unknown camera" if camera_index is None else f"camera {int(camera_index) + 1}"
    if event_type == "entry_accepted":
        return f"ENTRY detected for visit {visit_id} on {camera}. Is that correct?"
    if event_type == "leave_accepted":
        return f"LEAVE detected for visit {visit_id} on {camera}. Is that correct?"
    payload = event.get("payload")
    details = payload if isinstance(payload, Mapping) else {}
    shelf_id = details.get("shelfId")
    if event_type == "shelf_approach":
        return f"Visit {visit_id} approached shelf {shelf_id}. Is that correct?"
    if event_type == "shelf_departure":
        return f"Visit {visit_id} left shelf {shelf_id}. Is that correct?"
    return f"{event_type.replace('_', ' ')} for visit {visit_id}. Is that correct?"

