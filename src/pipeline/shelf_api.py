from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from pipeline.shelf_anchors import ShelfAnchor
from pipeline.shelf_config import ShelfDefinition
from pipeline.shelf_proximity import (
    ShelfCameraObservation,
    ShelfProximityEvent,
    ShelfProximityStatus,
)


@dataclass(frozen=True)
class ShelfCameraSnapshot:
    camera_index: int
    device_id: str
    camera_role: str
    rgb_sequence_number: int
    depth_sequence_number: int
    host_synced_seconds: float
    published_at_unix_milliseconds: int
    shelves: tuple[ShelfDefinition, ...]
    anchors_by_shelf: Mapping[int, tuple[ShelfAnchor, ...]]
    observations: tuple[ShelfCameraObservation, ...]
    states_by_shelf: Mapping[int, str]


def build_shelf_camera_snapshot(
    *,
    camera_index: int,
    device_id: str,
    camera_role: str,
    rgb_sequence_number: int,
    depth_sequence_number: int,
    host_synced_seconds: float,
    shelves: Sequence[ShelfDefinition],
    anchors_by_shelf: Mapping[int, Sequence[ShelfAnchor]],
    observations: Sequence[ShelfCameraObservation],
    states_by_shelf: Mapping[int, str],
) -> ShelfCameraSnapshot:
    return ShelfCameraSnapshot(
        camera_index=camera_index,
        device_id=device_id,
        camera_role=camera_role,
        rgb_sequence_number=rgb_sequence_number,
        depth_sequence_number=depth_sequence_number,
        host_synced_seconds=host_synced_seconds,
        published_at_unix_milliseconds=time.time_ns() // 1_000_000,
        shelves=tuple(shelves),
        anchors_by_shelf={
            shelf_id: tuple(anchors)
            for shelf_id, anchors in anchors_by_shelf.items()
        },
        observations=tuple(observations),
        states_by_shelf=dict(states_by_shelf),
    )


def _point_payload(point: tuple[float, float, float]) -> dict[str, float]:
    return {"x": point[0], "y": point[1], "z": point[2]}


def shelf_camera_snapshot_payload(
    snapshot: ShelfCameraSnapshot,
    *,
    age_milliseconds: int,
    status: str,
    include_observations: bool,
) -> dict[str, object]:
    observations_by_shelf: dict[int, list[ShelfCameraObservation]] = {}
    if include_observations:
        for observation in snapshot.observations:
            observations_by_shelf.setdefault(observation.shelf_id, []).append(
                observation
            )

    shelves_payload: list[dict[str, object]] = []
    for shelf in snapshot.shelves:
        observations = sorted(
            observations_by_shelf.get(shelf.shelf_id, []),
            key=lambda item: item.distance_mm,
        )
        closest = observations[0] if observations else None
        anchors = snapshot.anchors_by_shelf.get(shelf.shelf_id, ())
        anchor = anchors[0] if anchors else None
        shelves_payload.append(
            {
                "shelfId": shelf.shelf_id,
                "label": shelf.label,
                "markerId": shelf.marker_id,
                "markerIds": list(shelf.all_marker_ids),
                "anchorStatus": "valid" if anchors else "calibrating",
                "anchor": None
                if anchor is None
                else {
                    "markerId": anchor.marker_id,
                    "source": anchor.source,
                    "point3dMm": _point_payload(anchor.point_3d_mm),
                    "sampleCount": anchor.sample_count,
                    "rmsSpreadMm": anchor.rms_spread_mm,
                    "updatedAtUnixMilliseconds": anchor.updated_at_unix_milliseconds,
                },
                "anchors": [
                    {
                        "markerId": item.marker_id,
                        "source": item.source,
                        "point3dMm": _point_payload(item.point_3d_mm),
                        "sampleCount": item.sample_count,
                        "rmsSpreadMm": item.rms_spread_mm,
                        "updatedAtUnixMilliseconds": item.updated_at_unix_milliseconds,
                    }
                    for item in anchors
                ],
                "closest": None
                if closest is None
                else {
                    "markerId": closest.marker_id,
                    "trackId": closest.track_id,
                    "visitId": closest.visit_id,
                    "visitOrigin": closest.visit_origin,
                    "customerId": closest.customer_id,
                    "distanceMm": closest.distance_mm,
                    "personDepthAnchor": "torso",
                    "state": snapshot.states_by_shelf.get(
                        shelf.shelf_id,
                        "far",
                    ),
                },
                "people": [
                    {
                        "trackId": observation.track_id,
                        "visitId": observation.visit_id,
                        "visitOrigin": observation.visit_origin,
                        "customerId": observation.customer_id,
                        "markerId": observation.marker_id,
                        "distanceMm": observation.distance_mm,
                        "personPoint3dMm": _point_payload(
                            observation.person_point_3d_mm
                        ),
                    }
                    for observation in observations
                ],
            }
        )

    return {
        "camera": {
            "id": snapshot.camera_index,
            "deviceId": snapshot.device_id,
            "role": snapshot.camera_role,
            "status": status,
        },
        "frame": {
            "rgbSequenceNumber": snapshot.rgb_sequence_number,
            "depthSequenceNumber": snapshot.depth_sequence_number,
            "hostSyncedSeconds": snapshot.host_synced_seconds,
            "publishedAtUnixMilliseconds": snapshot.published_at_unix_milliseconds,
            "ageMilliseconds": age_milliseconds,
        },
        "shelves": shelves_payload,
    }


def shelf_status_payload(
    statuses: Sequence[ShelfProximityStatus],
) -> dict[str, object]:
    return {
        "shelves": [
            {
                "shelfId": status.shelf_id,
                "label": status.shelf_label,
                "markerId": status.marker_id,
                "state": status.state,
                "proximitySessionId": status.proximity_session_id,
                "closest": None
                if status.owner_visit_id is None
                else {
                    "visitId": status.owner_visit_id,
                    "customerId": status.owner_customer_id,
                    "trackId": status.owner_track_id,
                    "distanceMm": status.distance_mm,
                    "measurementAgeMilliseconds": status.measurement_age_milliseconds,
                    "camera": {
                        "id": status.source_camera_index,
                        "deviceId": status.source_device_id,
                    },
                },
            }
            for status in statuses
        ]
    }


def shelf_event_payload(event: ShelfProximityEvent) -> dict[str, object]:
    return {
        "eventId": event.event_id,
        "eventType": event.event_type,
        "proximitySessionId": event.proximity_session_id,
        "shelfId": event.shelf_id,
        "shelfLabel": event.shelf_label,
        "markerId": event.marker_id,
        "visitId": event.visit_id,
        "visitOrigin": event.visit_origin,
        "customerId": event.customer_id,
        "camera": {
            "id": event.camera_index,
            "deviceId": event.device_id,
            "trackId": event.track_id,
        },
        "distanceMm": event.distance_mm,
        "thresholdMm": event.threshold_mm,
        "hostSyncedSeconds": event.host_synced_seconds,
        "occurredAtUnixMilliseconds": event.occurred_at_unix_milliseconds,
        "rgbSequenceNumber": event.rgb_sequence_number,
        "depthSequenceNumber": event.depth_sequence_number,
        "personPoint3dMm": _point_payload(event.person_point_3d_mm),
        "reason": event.reason,
        "anchor": {
            "source": event.anchor.source,
            "point3dMm": _point_payload(event.anchor.point_3d_mm),
            "sampleCount": event.anchor.sample_count,
            "rmsSpreadMm": event.anchor.rms_spread_mm,
            "updatedAtUnixMilliseconds": event.anchor.updated_at_unix_milliseconds,
        },
    }
