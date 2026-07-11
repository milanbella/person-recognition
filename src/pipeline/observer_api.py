from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from pipeline.depth import DepthSample
from pipeline.tracking import Track
from pipeline.visit_identity import VisitAssignment
from pipeline.visit_registry import TrackVisitEvidence, VISIT_ORIGIN_ENTRANCE


VISIBLE_TRACK_STATUSES = {"NEW", "TRACKED"}


@dataclass(frozen=True)
class ObservedDepth:
    depth_mm: float
    anchor_pixel: tuple[int, int]
    point_3d_mm: tuple[float, float, float]
    valid_pixel_count: int


@dataclass(frozen=True)
class ObservedBody:
    has_appearance: bool
    aspect_ratio: float | None
    height_pixels: int | None


@dataclass(frozen=True)
class ObservedPerson:
    track_id: int
    track_status: str
    detection_score: float
    visit_id: int | None
    visit_origin: str | None
    customer_id: str | None
    customer_binding_status: str
    bounding_box: tuple[int, int, int, int]
    centroid: tuple[float, float]
    depth: ObservedDepth | None
    face_identity_ids: tuple[str, ...]
    body: ObservedBody
    matched_score: float | None


@dataclass(frozen=True)
class ObserverCameraSnapshot:
    camera_index: int
    device_id: str
    camera_role: str
    rgb_sequence_number: int
    host_synced_seconds: float
    published_at_unix_milliseconds: int
    frame_width: int
    frame_height: int
    observations: tuple[ObservedPerson, ...]


def _observed_depth(sample: DepthSample | None) -> ObservedDepth | None:
    if sample is None:
        return None
    return ObservedDepth(
        depth_mm=float(sample.depth_mm),
        anchor_pixel=(int(sample.anchor_px[0]), int(sample.anchor_px[1])),
        point_3d_mm=tuple(float(value) for value in sample.point_3d_mm),
        valid_pixel_count=int(sample.valid_pixel_count),
    )


def _customer_binding(
    assignment: VisitAssignment | None,
    customer_ids_by_visit: Mapping[int, str],
) -> tuple[str | None, str]:
    if assignment is None:
        return None, "not_available"
    customer_id = customer_ids_by_visit.get(assignment.visit_id)
    if customer_id is not None:
        return customer_id, "bound"
    if assignment.origin == VISIT_ORIGIN_ENTRANCE:
        return None, "pending"
    return None, "not_available"


def build_observer_camera_snapshot(
    *,
    camera_index: int,
    device_id: str,
    camera_role: str,
    rgb_frame: np.ndarray,
    rgb_sequence_number: int,
    host_synced_seconds: float,
    tracks: Sequence[Track],
    track_visit_evidence_by_id: Mapping[int, TrackVisitEvidence],
    visit_assignments: Mapping[int, VisitAssignment],
    depth_samples: Mapping[int, DepthSample],
    customer_ids_by_visit: Mapping[int, str],
) -> ObserverCameraSnapshot:
    frame_height, frame_width = rgb_frame.shape[:2]
    observations: list[ObservedPerson] = []
    for track in tracks:
        if track.status not in VISIBLE_TRACK_STATUSES:
            continue

        assignment = visit_assignments.get(track.track_id)
        evidence = track_visit_evidence_by_id.get(track.track_id)
        appearance = None if evidence is None else evidence.body_appearance
        customer_id, binding_status = _customer_binding(assignment, customer_ids_by_visit)
        observations.append(
            ObservedPerson(
                track_id=int(track.track_id),
                track_status=str(track.status),
                detection_score=float(track.score),
                visit_id=None if assignment is None else int(assignment.visit_id),
                visit_origin=None if assignment is None else str(assignment.origin),
                customer_id=customer_id,
                customer_binding_status=binding_status,
                bounding_box=(int(track.x1), int(track.y1), int(track.x2), int(track.y2)),
                centroid=tuple(float(value) for value in track.centroid()),
                depth=_observed_depth(depth_samples.get(track.track_id)),
                face_identity_ids=() if evidence is None else tuple(evidence.face_identity_ids),
                body=ObservedBody(
                    has_appearance=appearance is not None,
                    aspect_ratio=None if appearance is None else float(appearance.aspect_ratio),
                    height_pixels=None if appearance is None else int(appearance.height_px),
                ),
                matched_score=None if assignment is None else assignment.matched_score,
            )
        )

    return ObserverCameraSnapshot(
        camera_index=int(camera_index),
        device_id=str(device_id),
        camera_role=str(camera_role),
        rgb_sequence_number=int(rgb_sequence_number),
        host_synced_seconds=float(host_synced_seconds),
        published_at_unix_milliseconds=time.time_ns() // 1_000_000,
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        observations=tuple(observations),
    )


def observer_snapshot_payload(
    snapshot: ObserverCameraSnapshot,
    *,
    age_milliseconds: int,
    status: str,
    include_observations: bool,
) -> dict[str, object]:
    return {
        "camera": {
            "id": snapshot.camera_index,
            "deviceId": snapshot.device_id,
            "role": snapshot.camera_role,
            "status": status,
        },
        "frame": {
            "rgbSequenceNumber": snapshot.rgb_sequence_number,
            "hostSyncedSeconds": snapshot.host_synced_seconds,
            "publishedAtUnixMilliseconds": snapshot.published_at_unix_milliseconds,
            "ageMilliseconds": age_milliseconds,
            "width": snapshot.frame_width,
            "height": snapshot.frame_height,
        },
        "observations": [
            {
                "trackId": person.track_id,
                "trackStatus": person.track_status,
                "detectionScore": person.detection_score,
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
                "centroid": {"x": person.centroid[0], "y": person.centroid[1]},
                "depth": None
                if person.depth is None
                else {
                    "depthMm": person.depth.depth_mm,
                    "anchorPixel": {
                        "x": person.depth.anchor_pixel[0],
                        "y": person.depth.anchor_pixel[1],
                    },
                    "point3dMm": {
                        "x": person.depth.point_3d_mm[0],
                        "y": person.depth.point_3d_mm[1],
                        "z": person.depth.point_3d_mm[2],
                    },
                    "validPixelCount": person.depth.valid_pixel_count,
                },
                "faceIdentityIds": list(person.face_identity_ids),
                "body": {
                    "hasAppearance": person.body.has_appearance,
                    "aspectRatio": person.body.aspect_ratio,
                    "heightPixels": person.body.height_pixels,
                },
                "visitMatch": {"matchedScore": person.matched_score},
            }
            for person in snapshot.observations
        ]
        if include_observations
        else [],
    }
