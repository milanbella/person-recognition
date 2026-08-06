from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

from pipeline.shelf_anchors import ShelfAnchor
from pipeline.shelf_config import ShelfDefinition, ShelfWatchingConfig


@dataclass(frozen=True)
class ShelfCameraObservation:
    shelf_id: int
    shelf_label: str
    marker_id: int
    camera_index: int
    device_id: str
    track_id: int
    visit_id: int | None
    visit_origin: str | None
    customer_id: str | None
    distance_mm: float
    person_point_3d_mm: tuple[float, float, float]
    anchor: ShelfAnchor
    host_synced_seconds: float
    observed_at_unix_milliseconds: int
    rgb_sequence_number: int
    depth_sequence_number: int
    track_bounding_box: tuple[int, int, int, int] | None = None
    person_depth_mm: float | None = None
    person_depth_valid_pixel_count: int | None = None
    person_depth_roi: tuple[int, int, int, int] | None = None
    person_depth_anchor_px: tuple[int, int] | None = None


@dataclass(frozen=True)
class ShelfProximityEvent:
    event_type: str
    proximity_session_id: str
    shelf_id: int
    shelf_label: str
    marker_id: int
    visit_id: int
    visit_origin: str | None
    customer_id: str | None
    camera_index: int
    device_id: str
    track_id: int
    distance_mm: float
    threshold_mm: float
    host_synced_seconds: float
    occurred_at_unix_milliseconds: int
    rgb_sequence_number: int
    depth_sequence_number: int
    person_point_3d_mm: tuple[float, float, float]
    anchor: ShelfAnchor
    reason: str
    event_id: int | None = None

    def with_event_id(self, event_id: int) -> ShelfProximityEvent:
        return replace(self, event_id=event_id)


@dataclass(frozen=True)
class ShelfProximityStatus:
    shelf_id: int
    shelf_label: str
    marker_id: int
    state: str
    owner_visit_id: int | None
    owner_customer_id: str | None
    owner_track_id: int | None
    source_camera_index: int | None
    source_device_id: str | None
    distance_mm: float | None
    measurement_age_milliseconds: int | None
    proximity_session_id: str | None


@dataclass
class _ShelfRuntimeState:
    owner_visit_id: int | None = None
    phase: str = "far"
    transition_started_unix_milliseconds: int | None = None
    challenger_visit_id: int | None = None
    challenger_started_unix_milliseconds: int | None = None
    last_observation: ShelfCameraObservation | None = None
    proximity_session_id: str | None = None
    minimum_distance_mm: float | None = None


def person_to_shelf_distance_mm(
    person_point_3d_mm: tuple[float, float, float],
    shelf_anchor_point_3d_mm: tuple[float, float, float],
) -> float:
    return math.sqrt(
        sum(
            (person_coordinate - shelf_coordinate) ** 2
            for person_coordinate, shelf_coordinate in zip(
                person_point_3d_mm,
                shelf_anchor_point_3d_mm,
            )
        )
    )


class ShelfProximityCoordinator:
    def __init__(self, config: ShelfWatchingConfig) -> None:
        self._shelves = config.shelf_by_id()
        self._states = {
            shelf_id: _ShelfRuntimeState() for shelf_id in self._shelves
        }
        self._camera_observations: dict[
            tuple[int, int, int], ShelfCameraObservation
        ] = {}
        self._closed_visit_ids: set[int] = set()
        self._last_now_unix_milliseconds = 0

    def update_camera(
        self,
        *,
        camera_index: int,
        observations: Sequence[ShelfCameraObservation],
        host_synced_seconds: float,
        now_unix_milliseconds: int,
    ) -> tuple[ShelfProximityEvent, ...]:
        self._last_now_unix_milliseconds = max(
            self._last_now_unix_milliseconds,
            now_unix_milliseconds,
        )
        for key in [
            key for key in self._camera_observations if key[0] == camera_index
        ]:
            self._camera_observations.pop(key, None)
        for observation in observations:
            if (
                observation.visit_id is None
                or observation.visit_id in self._closed_visit_ids
                or observation.shelf_id not in self._shelves
            ):
                continue
            self._camera_observations[
                (camera_index, observation.shelf_id, observation.visit_id)
            ] = observation

        preferred_shelves = self._preferred_shelves_by_visit(
            now_unix_milliseconds=now_unix_milliseconds,
        )
        events: list[ShelfProximityEvent] = []
        for shelf in self._shelves.values():
            events.extend(
                self._advance_shelf(
                    shelf,
                    preferred_shelves=preferred_shelves,
                    host_synced_seconds=host_synced_seconds,
                    now_unix_milliseconds=now_unix_milliseconds,
                )
            )
        return tuple(events)

    def close_visit(
        self,
        visit_id: int | None,
        *,
        host_synced_seconds: float,
        now_unix_milliseconds: int,
    ) -> tuple[ShelfProximityEvent, ...]:
        if visit_id is None:
            return ()
        self._closed_visit_ids.add(visit_id)
        for key in [
            key for key in self._camera_observations if key[2] == visit_id
        ]:
            self._camera_observations.pop(key, None)

        events: list[ShelfProximityEvent] = []
        for shelf_id, state in self._states.items():
            if state.owner_visit_id != visit_id:
                continue
            if state.phase in {"near", "departing"} and state.last_observation is not None:
                events.append(
                    self._departure_event(
                        self._shelves[shelf_id],
                        state,
                        state.last_observation,
                        host_synced_seconds=host_synced_seconds,
                        now_unix_milliseconds=now_unix_milliseconds,
                        reason="visit_closed",
                    )
                )
            self._reset_state(state)
        return tuple(events)

    def restore_near_session(
        self,
        *,
        shelf_id: int,
        visit_id: int,
        proximity_session_id: str,
        observation: ShelfCameraObservation,
        minimum_distance_mm: float | None = None,
    ) -> None:
        shelf = self._shelves.get(shelf_id)
        if shelf is None or observation.marker_id not in shelf.all_marker_ids:
            return
        if any(
            other_shelf_id != shelf_id
            and other_state.owner_visit_id == visit_id
            and other_state.phase in {"near", "departing"}
            for other_shelf_id, other_state in self._states.items()
        ):
            return
        state = self._states[shelf_id]
        state.owner_visit_id = visit_id
        state.phase = "near"
        state.proximity_session_id = proximity_session_id
        state.last_observation = observation
        state.minimum_distance_mm = minimum_distance_mm

    def statuses(
        self,
        *,
        now_unix_milliseconds: int | None = None,
    ) -> tuple[ShelfProximityStatus, ...]:
        now_ms = (
            self._last_now_unix_milliseconds
            if now_unix_milliseconds is None
            else now_unix_milliseconds
        )
        statuses: list[ShelfProximityStatus] = []
        for shelf_id, shelf in sorted(self._shelves.items()):
            state = self._states[shelf_id]
            observation = state.last_observation
            statuses.append(
                ShelfProximityStatus(
                    shelf_id=shelf.shelf_id,
                    shelf_label=shelf.label,
                    marker_id=(
                        shelf.marker_id
                        if observation is None
                        else observation.marker_id
                    ),
                    state=state.phase,
                    owner_visit_id=state.owner_visit_id,
                    owner_customer_id=(
                        None if observation is None else observation.customer_id
                    ),
                    owner_track_id=(
                        None if observation is None else observation.track_id
                    ),
                    source_camera_index=(
                        None if observation is None else observation.camera_index
                    ),
                    source_device_id=(
                        None if observation is None else observation.device_id
                    ),
                    distance_mm=(
                        None if observation is None else observation.distance_mm
                    ),
                    measurement_age_milliseconds=(
                        None
                        if observation is None
                        else max(
                            0,
                            now_ms - observation.observed_at_unix_milliseconds,
                        )
                    ),
                    proximity_session_id=state.proximity_session_id,
                )
            )
        return tuple(statuses)

    def _fresh_observations(
        self,
        shelf: ShelfDefinition,
        *,
        now_unix_milliseconds: int,
    ) -> dict[int, ShelfCameraObservation]:
        best_by_visit: dict[int, ShelfCameraObservation] = {}
        for (_camera_index, shelf_id, visit_id), observation in self._camera_observations.items():
            if shelf_id != shelf.shelf_id or visit_id in self._closed_visit_ids:
                continue
            age_ms = now_unix_milliseconds - observation.observed_at_unix_milliseconds
            if age_ms > shelf.lost_visit_grace_milliseconds:
                continue
            current = best_by_visit.get(visit_id)
            if current is None or observation.distance_mm < current.distance_mm:
                best_by_visit[visit_id] = observation
        return best_by_visit

    def _preferred_shelves_by_visit(
        self,
        *,
        now_unix_milliseconds: int,
    ) -> dict[int, int]:
        """Choose at most one shelf allowed to begin approach for each visit."""
        active: dict[int, int] = {}
        for shelf_id, state in self._states.items():
            if (
                state.owner_visit_id is not None
                and state.phase in {"near", "departing"}
                and state.proximity_session_id is not None
            ):
                active.setdefault(state.owner_visit_id, shelf_id)

        candidates: dict[int, list[tuple[ShelfDefinition, ShelfCameraObservation]]] = {}
        for shelf in self._shelves.values():
            for visit_id, observation in self._fresh_observations(
                shelf,
                now_unix_milliseconds=now_unix_milliseconds,
            ).items():
                if observation.distance_mm <= shelf.approach_distance_mm:
                    candidates.setdefault(visit_id, []).append((shelf, observation))

        preferred = dict(active)
        for visit_id, options in candidates.items():
            if visit_id in active:
                continue
            options.sort(
                key=lambda item: (
                    item[1].distance_mm / item[0].approach_distance_mm,
                    item[1].distance_mm,
                    item[0].shelf_id,
                )
            )
            winner_shelf, winner_observation = options[0]
            if len(options) > 1:
                runner_shelf, runner_observation = options[1]
                distance_margin = (
                    runner_observation.distance_mm - winner_observation.distance_mm
                )
                required_margin = max(
                    winner_shelf.owner_switch_margin_mm,
                    runner_shelf.owner_switch_margin_mm,
                )
                if distance_margin < required_margin:
                    continue
            preferred[visit_id] = winner_shelf.shelf_id
        return preferred

    def _advance_shelf(
        self,
        shelf: ShelfDefinition,
        *,
        preferred_shelves: Mapping[int, int],
        host_synced_seconds: float,
        now_unix_milliseconds: int,
    ) -> list[ShelfProximityEvent]:
        state = self._states[shelf.shelf_id]
        best_by_visit = self._fresh_observations(
            shelf,
            now_unix_milliseconds=now_unix_milliseconds,
        )
        closest = min(
            best_by_visit.values(),
            key=lambda observation: observation.distance_mm,
            default=None,
        )
        events: list[ShelfProximityEvent] = []

        owner_observation = (
            None
            if state.owner_visit_id is None
            else best_by_visit.get(state.owner_visit_id)
        )
        if state.owner_visit_id is not None and owner_observation is None:
            last_seen_ms = (
                None
                if state.last_observation is None
                else state.last_observation.observed_at_unix_milliseconds
            )
            if (
                last_seen_ms is not None
                and now_unix_milliseconds - last_seen_ms
                <= shelf.lost_visit_grace_milliseconds
            ):
                return events
            if state.phase in {"near", "departing"} and state.last_observation is not None:
                events.append(
                    self._departure_event(
                        shelf,
                        state,
                        state.last_observation,
                        host_synced_seconds=host_synced_seconds,
                        now_unix_milliseconds=now_unix_milliseconds,
                        reason="observation_lost",
                    )
                )
            self._reset_state(state)
            owner_observation = None

        if state.owner_visit_id is None and closest is not None:
            state.owner_visit_id = closest.visit_id
            owner_observation = closest

        if (
            owner_observation is not None
            and closest is not None
            and closest.visit_id != state.owner_visit_id
            and closest.distance_mm
            <= owner_observation.distance_mm - shelf.owner_switch_margin_mm
        ):
            if state.challenger_visit_id != closest.visit_id:
                state.challenger_visit_id = closest.visit_id
                state.challenger_started_unix_milliseconds = now_unix_milliseconds
            elif (
                state.challenger_started_unix_milliseconds is not None
                and now_unix_milliseconds
                - state.challenger_started_unix_milliseconds
                >= shelf.owner_switch_dwell_milliseconds
            ):
                if state.phase in {"near", "departing"} and state.last_observation is not None:
                    events.append(
                        self._departure_event(
                            shelf,
                            state,
                            state.last_observation,
                            host_synced_seconds=host_synced_seconds,
                            now_unix_milliseconds=now_unix_milliseconds,
                            reason="owner_changed",
                        )
                    )
                self._reset_state(state)
                state.owner_visit_id = closest.visit_id
                owner_observation = closest
        else:
            state.challenger_visit_id = None
            state.challenger_started_unix_milliseconds = None

        if owner_observation is None:
            return events
        state.last_observation = owner_observation
        if (
            state.minimum_distance_mm is None
            or owner_observation.distance_mm < state.minimum_distance_mm
        ):
            state.minimum_distance_mm = owner_observation.distance_mm
        events.extend(
            self._advance_owner_distance(
                shelf,
                state,
                owner_observation,
                approach_allowed=(
                    preferred_shelves.get(owner_observation.visit_id)
                    == shelf.shelf_id
                ),
                host_synced_seconds=host_synced_seconds,
                now_unix_milliseconds=now_unix_milliseconds,
            )
        )
        return events

    def _advance_owner_distance(
        self,
        shelf: ShelfDefinition,
        state: _ShelfRuntimeState,
        observation: ShelfCameraObservation,
        *,
        approach_allowed: bool,
        host_synced_seconds: float,
        now_unix_milliseconds: int,
    ) -> list[ShelfProximityEvent]:
        events: list[ShelfProximityEvent] = []
        if state.phase in {"far", "approaching"} and not approach_allowed:
            state.phase = "far"
            state.transition_started_unix_milliseconds = None
            return events
        if state.phase == "far":
            if observation.distance_mm <= shelf.approach_distance_mm:
                state.phase = "approaching"
                state.transition_started_unix_milliseconds = now_unix_milliseconds
                if shelf.approach_dwell_milliseconds == 0:
                    events.append(
                        self._approach_event(
                            shelf,
                            state,
                            observation,
                            host_synced_seconds=host_synced_seconds,
                            now_unix_milliseconds=now_unix_milliseconds,
                        )
                    )
        elif state.phase == "approaching":
            if observation.distance_mm > shelf.approach_distance_mm:
                state.phase = "far"
                state.transition_started_unix_milliseconds = None
            elif (
                state.transition_started_unix_milliseconds is not None
                and now_unix_milliseconds
                - state.transition_started_unix_milliseconds
                >= shelf.approach_dwell_milliseconds
            ):
                events.append(
                    self._approach_event(
                        shelf,
                        state,
                        observation,
                        host_synced_seconds=host_synced_seconds,
                        now_unix_milliseconds=now_unix_milliseconds,
                    )
                )
        elif state.phase == "near":
            if observation.distance_mm >= shelf.departure_distance_mm:
                state.phase = "departing"
                state.transition_started_unix_milliseconds = now_unix_milliseconds
                if shelf.departure_dwell_milliseconds == 0:
                    events.append(
                        self._departure_event(
                            shelf,
                            state,
                            observation,
                            host_synced_seconds=host_synced_seconds,
                            now_unix_milliseconds=now_unix_milliseconds,
                            reason="distance_hysteresis",
                        )
                    )
        elif state.phase == "departing":
            if observation.distance_mm < shelf.departure_distance_mm:
                state.phase = "near"
                state.transition_started_unix_milliseconds = None
            elif (
                state.transition_started_unix_milliseconds is not None
                and now_unix_milliseconds
                - state.transition_started_unix_milliseconds
                >= shelf.departure_dwell_milliseconds
            ):
                events.append(
                    self._departure_event(
                        shelf,
                        state,
                        observation,
                        host_synced_seconds=host_synced_seconds,
                        now_unix_milliseconds=now_unix_milliseconds,
                        reason="distance_hysteresis",
                    )
                )
        return events

    def _approach_event(
        self,
        shelf: ShelfDefinition,
        state: _ShelfRuntimeState,
        observation: ShelfCameraObservation,
        *,
        host_synced_seconds: float,
        now_unix_milliseconds: int,
    ) -> ShelfProximityEvent:
        assert observation.visit_id is not None
        session_id = (
            f"{shelf.shelf_id}:visit-{observation.visit_id}:"
            f"{now_unix_milliseconds}"
        )
        state.phase = "near"
        state.transition_started_unix_milliseconds = None
        state.proximity_session_id = session_id
        return self._event(
            event_type="shelf_approach",
            shelf=shelf,
            state=state,
            observation=observation,
            threshold_mm=shelf.approach_distance_mm,
            host_synced_seconds=host_synced_seconds,
            now_unix_milliseconds=now_unix_milliseconds,
            reason="distance_dwell",
        )

    def _departure_event(
        self,
        shelf: ShelfDefinition,
        state: _ShelfRuntimeState,
        observation: ShelfCameraObservation,
        *,
        host_synced_seconds: float,
        now_unix_milliseconds: int,
        reason: str,
    ) -> ShelfProximityEvent:
        event = self._event(
            event_type="shelf_departure",
            shelf=shelf,
            state=state,
            observation=observation,
            threshold_mm=shelf.departure_distance_mm,
            host_synced_seconds=host_synced_seconds,
            now_unix_milliseconds=now_unix_milliseconds,
            reason=reason,
        )
        state.phase = "far"
        state.transition_started_unix_milliseconds = None
        state.proximity_session_id = None
        state.minimum_distance_mm = None
        return event

    @staticmethod
    def _event(
        *,
        event_type: str,
        shelf: ShelfDefinition,
        state: _ShelfRuntimeState,
        observation: ShelfCameraObservation,
        threshold_mm: float,
        host_synced_seconds: float,
        now_unix_milliseconds: int,
        reason: str,
    ) -> ShelfProximityEvent:
        assert observation.visit_id is not None
        assert state.proximity_session_id is not None
        return ShelfProximityEvent(
            event_type=event_type,
            proximity_session_id=state.proximity_session_id,
            shelf_id=shelf.shelf_id,
            shelf_label=shelf.label,
            marker_id=observation.marker_id,
            visit_id=observation.visit_id,
            visit_origin=observation.visit_origin,
            customer_id=observation.customer_id,
            camera_index=observation.camera_index,
            device_id=observation.device_id,
            track_id=observation.track_id,
            distance_mm=observation.distance_mm,
            threshold_mm=threshold_mm,
            host_synced_seconds=host_synced_seconds,
            occurred_at_unix_milliseconds=now_unix_milliseconds,
            rgb_sequence_number=observation.rgb_sequence_number,
            depth_sequence_number=observation.depth_sequence_number,
            person_point_3d_mm=observation.person_point_3d_mm,
            anchor=observation.anchor,
            reason=reason,
        )

    @staticmethod
    def _reset_state(state: _ShelfRuntimeState) -> None:
        state.owner_visit_id = None
        state.phase = "far"
        state.transition_started_unix_milliseconds = None
        state.challenger_visit_id = None
        state.challenger_started_unix_milliseconds = None
        state.last_observation = None
        state.proximity_session_id = None
        state.minimum_distance_mm = None


def best_observations_by_shelf(
    observations: Iterable[ShelfCameraObservation],
) -> Mapping[int, ShelfCameraObservation]:
    result: dict[int, ShelfCameraObservation] = {}
    for observation in observations:
        current = result.get(observation.shelf_id)
        if current is None or observation.distance_mm < current.distance_mm:
            result[observation.shelf_id] = observation
    return result
