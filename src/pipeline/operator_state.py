from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pipeline.observer_api import (
    ObserverCameraSnapshot,
    observer_snapshot_payload,
)
from pipeline.operator_models import ObservationReference, OperatorEvent
from pipeline.shelf_api import ShelfCameraSnapshot, shelf_camera_snapshot_payload
from pipeline.shelf_proximity import ShelfProximityStatus


EventSink = Callable[[OperatorEvent], None]


class OperatorState:
    """Thread-safe latest-state model and bounded transition event feed."""

    def __init__(
        self,
        *,
        camera_device_ids: Sequence[str],
        camera_roles: Sequence[str],
        camera_timeout_seconds: float,
        event_ring_size: int = 5000,
        initial_event_id: int = 0,
        event_sink: EventSink | None = None,
    ) -> None:
        if event_ring_size <= 0:
            raise ValueError("Operator event ring size must be positive.")
        self._camera_timeout_seconds = camera_timeout_seconds
        self._camera_metadata = {
            index: {
                "id": index,
                "deviceId": device_id,
                "role": camera_roles[index],
            }
            for index, device_id in enumerate(camera_device_ids)
        }
        self._observer_snapshots: dict[int, ObserverCameraSnapshot] = {}
        self._observer_history: dict[int, deque[ObserverCameraSnapshot]] = {
            index: deque(maxlen=128) for index in self._camera_metadata
        }
        self._observer_published_monotonic: dict[int, float] = {}
        self._shelf_snapshots: dict[int, ShelfCameraSnapshot] = {}
        self._shelf_published_monotonic: dict[int, float] = {}
        self._camera_frame_monotonic: dict[int, float] = {}
        self._camera_statuses: dict[int, str] = {
            index: "offline" for index in self._camera_metadata
        }
        self._latest_jpegs: dict[int, tuple[bytes, int, int | None, int]] = {}
        self._shelf_statuses: tuple[ShelfProximityStatus, ...] = ()
        self._visit_states: dict[int, dict[str, Any]] = {}
        self._events: deque[OperatorEvent] = deque(maxlen=event_ring_size)
        self._next_event_id = initial_event_id + 1
        self._event_sink = event_sink
        self._condition = threading.Condition()

    @property
    def last_event_id(self) -> int:
        with self._condition:
            return self._next_event_id - 1

    def mark_camera_frame(
        self,
        camera_index: int,
        *,
        jpeg: bytes,
        stream_sequence: int,
        rgb_sequence_number: int | None,
    ) -> None:
        now_monotonic = time.monotonic()
        now_unix_ms = time.time_ns() // 1_000_000
        became_active = False
        with self._condition:
            self._camera_frame_monotonic[camera_index] = now_monotonic
            self._latest_jpegs[camera_index] = (
                jpeg,
                stream_sequence,
                rgb_sequence_number,
                now_unix_ms,
            )
            if self._camera_statuses.get(camera_index) != "active":
                self._camera_statuses[camera_index] = "active"
                became_active = True
        if became_active:
            metadata = self._camera_metadata[camera_index]
            self.publish_event(
                event_type="camera_status_changed",
                occurred_at_unix_milliseconds=now_unix_ms,
                camera_index=camera_index,
                device_id=str(metadata["deviceId"]),
                payload={"previousStatus": "offline", "status": "active"},
            )

    def latest_evidence(
        self,
        camera_index: int,
    ) -> tuple[bytes, int, int | None, int] | None:
        with self._condition:
            evidence = self._latest_jpegs.get(camera_index)
            return None if evidence is None else tuple(evidence)

    def publish_observer_snapshot(self, snapshot: ObserverCameraSnapshot) -> None:
        emitted: list[dict[str, Any]] = []
        with self._condition:
            previous = self._observer_snapshots.get(snapshot.camera_index)
            self._observer_snapshots[snapshot.camera_index] = snapshot
            if snapshot.observations:
                self._observer_history[snapshot.camera_index].append(snapshot)
            self._observer_published_monotonic[snapshot.camera_index] = time.monotonic()

            previous_people = (
                {}
                if previous is None
                else {person.track_id: person for person in previous.observations}
            )
            current_people = {
                person.track_id: person for person in snapshot.observations
            }
            for track_id in sorted(current_people.keys() - previous_people.keys()):
                person = current_people[track_id]
                emitted.append(
                    self._person_event_payload(
                        "track_appeared",
                        snapshot,
                        person,
                    )
                )
                if person.visit_id is not None:
                    self._visit_states.setdefault(person.visit_id, {}).update(
                        {
                            "visitId": person.visit_id,
                            "origin": person.visit_origin,
                            "customerId": person.customer_id,
                            "status": self._visit_states.get(
                                person.visit_id, {}
                            ).get("status", "active"),
                            "lastCameraIndex": snapshot.camera_index,
                            "lastDeviceId": snapshot.device_id,
                            "lastTrackId": person.track_id,
                            "lastSeenHostSeconds": snapshot.host_synced_seconds,
                        }
                    )
            for track_id in sorted(previous_people.keys() - current_people.keys()):
                emitted.append(
                    self._person_event_payload(
                        "track_disappeared",
                        snapshot,
                        previous_people[track_id],
                    )
                )
            for track_id in sorted(current_people.keys() & previous_people.keys()):
                before = previous_people[track_id]
                after = current_people[track_id]
                if before.track_status != after.track_status:
                    payload = self._person_event_payload(
                        "track_status_changed",
                        snapshot,
                        after,
                    )
                    payload["payload"].update(
                        {"previousStatus": before.track_status}
                    )
                    emitted.append(payload)
                if (
                    before.visit_id != after.visit_id
                    or before.visit_origin != after.visit_origin
                ):
                    payload = self._person_event_payload(
                        "visit_assignment_changed",
                        snapshot,
                        after,
                    )
                    payload["payload"].update(
                        {
                            "previousVisitId": before.visit_id,
                            "previousVisitOrigin": before.visit_origin,
                        }
                    )
                    emitted.append(payload)
                known_customer = (
                    None
                    if after.visit_id is None
                    else self._visit_states.get(after.visit_id, {}).get(
                        "customerId"
                    )
                )
                if (
                    before.customer_id != after.customer_id
                    and known_customer != after.customer_id
                ):
                    payload = self._person_event_payload(
                        "customer_binding_changed",
                        snapshot,
                        after,
                    )
                    payload["payload"]["previousCustomerId"] = before.customer_id
                    emitted.append(payload)

                if after.visit_id is not None:
                    self._visit_states.setdefault(after.visit_id, {}).update(
                        {
                            "visitId": after.visit_id,
                            "origin": after.visit_origin,
                            "customerId": after.customer_id,
                            "status": self._visit_states.get(
                                after.visit_id, {}
                            ).get("status", "active"),
                            "lastCameraIndex": snapshot.camera_index,
                            "lastDeviceId": snapshot.device_id,
                            "lastTrackId": after.track_id,
                            "lastSeenHostSeconds": snapshot.host_synced_seconds,
                        }
                    )

        for values in emitted:
            self.publish_event(**values)

    @staticmethod
    def _person_event_payload(
        event_type: str,
        snapshot: ObserverCameraSnapshot,
        person: Any,
    ) -> dict[str, Any]:
        return {
            "event_type": event_type,
            "occurred_at_unix_milliseconds": snapshot.published_at_unix_milliseconds,
            "host_synced_seconds": snapshot.host_synced_seconds,
            "camera_index": snapshot.camera_index,
            "device_id": snapshot.device_id,
            "rgb_sequence_number": snapshot.rgb_sequence_number,
            "track_id": person.track_id,
            "visit_id": person.visit_id,
            "payload": {
                "trackStatus": person.track_status,
                "visitOrigin": person.visit_origin,
                "customerId": person.customer_id,
                "customerBindingStatus": person.customer_binding_status,
                "boundingBox": {
                    "x1": person.bounding_box[0],
                    "y1": person.bounding_box[1],
                    "x2": person.bounding_box[2],
                    "y2": person.bounding_box[3],
                },
                "matchedScore": person.matched_score,
                "matchState": person.match_state,
                "matchDecision": person.match_decision,
                "matchReason": person.match_reason,
            },
        }

    def publish_shelf_snapshot(self, snapshot: ShelfCameraSnapshot) -> None:
        with self._condition:
            self._shelf_snapshots[snapshot.camera_index] = snapshot
            self._shelf_published_monotonic[snapshot.camera_index] = time.monotonic()

    def publish_shelf_statuses(
        self,
        statuses: tuple[ShelfProximityStatus, ...],
    ) -> None:
        with self._condition:
            self._shelf_statuses = tuple(statuses)

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
    ) -> None:
        if visit_id is None:
            return
        with self._condition:
            state = self._visit_states.setdefault(visit_id, {"visitId": visit_id})
            if status is not None:
                state["status"] = status
            if origin is not None:
                state["origin"] = origin
            if customer_id is not None:
                state["customerId"] = customer_id
            if host_synced_seconds is not None:
                state["lastSeenHostSeconds"] = host_synced_seconds
            if camera_index is not None:
                state["lastCameraIndex"] = camera_index
            if device_id is not None:
                state["lastDeviceId"] = device_id
            if track_id is not None:
                state["lastTrackId"] = track_id

    def publish_event(
        self,
        *,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        occurred_at_unix_milliseconds: int | None = None,
        source: str = "recognition_pipeline",
        host_synced_seconds: float | None = None,
        camera_index: int | None = None,
        device_id: str | None = None,
        rgb_sequence_number: int | None = None,
        track_id: int | None = None,
        visit_id: int | None = None,
    ) -> OperatorEvent:
        with self._condition:
            event = OperatorEvent(
                event_id=self._next_event_id,
                event_type=event_type,
                occurred_at_unix_milliseconds=(
                    time.time_ns() // 1_000_000
                    if occurred_at_unix_milliseconds is None
                    else occurred_at_unix_milliseconds
                ),
                source=source,
                payload={} if payload is None else dict(payload),
                host_synced_seconds=host_synced_seconds,
                camera_index=camera_index,
                device_id=device_id,
                rgb_sequence_number=rgb_sequence_number,
                track_id=track_id,
                visit_id=visit_id,
            )
            self._next_event_id += 1
            self._events.append(event)
            self._condition.notify_all()

        if self._event_sink is not None:
            try:
                self._event_sink(event)
            except Exception as exc:
                print(
                    "OPERATOR_EVENT_PERSIST_ERROR "
                    f"event_id={event.event_id} event_type={event.event_type} "
                    f"error={exc}"
                )
        return event

    def events_after(self, event_id: int) -> tuple[list[OperatorEvent], bool]:
        with self._condition:
            if not self._events:
                return [], False
            first_id = self._events[0].event_id
            resync_required = event_id < first_id - 1
            return [
                event for event in self._events if event.event_id > event_id
            ], resync_required

    def wait_for_events(
        self,
        event_id: int,
        *,
        timeout_seconds: float,
    ) -> tuple[list[OperatorEvent], bool]:
        with self._condition:
            self._condition.wait_for(
                lambda: bool(self._events)
                and self._events[-1].event_id > event_id,
                timeout=timeout_seconds,
            )
        return self.events_after(event_id)

    def resolve_observation(
        self,
        reference: ObservationReference,
        *,
        max_age_milliseconds: int = 3000,
    ) -> dict[str, Any]:
        now_unix_ms = time.time_ns() // 1_000_000
        with self._condition:
            current = self._observer_snapshots.get(reference.camera_index)
            snapshots = (
                ([current] if current is not None else [])
                + list(
                    reversed(self._observer_history.get(reference.camera_index, ()))
                )
            )
        snapshot = next(
            (
                candidate
                for candidate in snapshots
                if candidate.camera_index == reference.camera_index
                and candidate.device_id == reference.device_id
                and candidate.rgb_sequence_number == reference.rgb_sequence_number
                and any(
                    person.track_id == reference.track_id
                    for person in candidate.observations
                )
            ),
            None,
        )
        if snapshot is None:
            raise LookupError("Observation frame is no longer available.")
        age_ms = now_unix_ms - snapshot.published_at_unix_milliseconds
        if age_ms > max_age_milliseconds:
            raise LookupError(f"Observation is stale ({age_ms} ms).")
        person = next(
            (
                candidate
                for candidate in snapshot.observations
                if candidate.track_id == reference.track_id
            ),
            None,
        )
        if person is None:
            raise LookupError("Track is no longer visible in the referenced frame.")
        payload = observer_snapshot_payload(
            snapshot,
            age_milliseconds=max(0, age_ms),
            status="active",
            include_observations=True,
        )
        observation = next(
            item
            for item in payload["observations"]
            if item["trackId"] == reference.track_id
        )
        return {
            "camera": payload["camera"],
            "frame": payload["frame"],
            "observation": observation,
        }

    def state_payload(
        self,
        *,
        active_run: Mapping[str, Any] | None,
        persisted_visits: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        now_monotonic = time.monotonic()
        offline_changes: list[tuple[int, str]] = []
        with self._condition:
            for index, status in self._camera_statuses.items():
                frame_time = self._camera_frame_monotonic.get(index)
                if (
                    status == "active"
                    and (
                        frame_time is None
                        or now_monotonic - frame_time > self._camera_timeout_seconds
                    )
                ):
                    self._camera_statuses[index] = "offline"
                    offline_changes.append(
                        (index, str(self._camera_metadata[index]["deviceId"]))
                    )
        for index, device_id in offline_changes:
            self.publish_event(
                event_type="camera_status_changed",
                camera_index=index,
                device_id=device_id,
                payload={"previousStatus": "active", "status": "offline"},
            )

        with self._condition:
            recent_events = [
                event.as_payload() for event in list(self._events)[-100:]
            ]
            cameras = []
            for index, metadata in self._camera_metadata.items():
                status = self._camera_statuses[index]
                snapshot = self._observer_snapshots.get(index)
                cameras.append(
                    {
                        **metadata,
                        "status": status,
                        "visiblePersonCount": (
                            0 if snapshot is None else len(snapshot.observations)
                        ),
                        "streamUrl": f"/stream/{index}",
                        "observationsUrl": (
                            f"/observer-cameras/{index}/observations"
                        ),
                    }
                )
            visits = {int(item["visitId"]): dict(item) for item in persisted_visits}
            for visit_id, state in self._visit_states.items():
                visits.setdefault(visit_id, {}).update(state)

            shelves = [
                {
                    "shelfId": status.shelf_id,
                    "label": status.shelf_label,
                    "markerId": status.marker_id,
                    "state": status.state,
                    "proximitySessionId": status.proximity_session_id,
                    "visitId": status.owner_visit_id,
                    "customerId": status.owner_customer_id,
                    "trackId": status.owner_track_id,
                    "distanceMm": status.distance_mm,
                    "cameraIndex": status.source_camera_index,
                }
                for status in self._shelf_statuses
            ]
            last_event_id = self._next_event_id - 1

        return {
            "serverTimeUnixMilliseconds": time.time_ns() // 1_000_000,
            "activeRun": None if active_run is None else dict(active_run),
            "cameras": cameras,
            "visits": sorted(visits.values(), key=lambda item: item["visitId"]),
            "shelves": shelves,
            "lastEventId": last_event_id,
            "recentEvents": recent_events,
        }

    def shelf_snapshot_payload(self, camera_index: int) -> dict[str, Any] | None:
        now = time.monotonic()
        with self._condition:
            snapshot = self._shelf_snapshots.get(camera_index)
            published = self._shelf_published_monotonic.get(camera_index)
        if snapshot is None or published is None:
            return None
        age_ms = max(0, int(round((now - published) * 1000)))
        fresh = now - published <= self._camera_timeout_seconds
        return shelf_camera_snapshot_payload(
            snapshot,
            age_milliseconds=age_ms,
            status="active" if fresh else "offline",
            include_observations=fresh,
        )
