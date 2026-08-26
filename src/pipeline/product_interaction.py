from __future__ import annotations

import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from pipeline.depth import CameraIntrinsics, pixel_to_camera_point_mm
from pipeline.pose import PoseObservation
from pipeline.product_detection import ProductDetection
from pipeline.shelf_regions import (
    ShelfDepthCell,
    ShelfRegion,
    box_center_in_shelf_region,
    shelf_depth_cell_at,
)


INTERACTION_STATE_UNKNOWN = "UNKNOWN"
INTERACTION_STATE_ON_SHELF = "ON_SHELF"
INTERACTION_STATE_HAND_CONTACT = "HAND_PRODUCT_CONTACT"
INTERACTION_STATE_HELD = "HELD_BY_VISIT"
INTERACTION_STATE_RETURN_PLACING = "RETURN_PLACING"
EVENT_PRODUCT_PICKED = "PRODUCT_PICKED"
EVENT_PRODUCT_RETURNED = "PRODUCT_RETURNED"
SHELF_OCCUPANCY_ON = "ON_SHELF_3D"
SHELF_OCCUPANCY_OFF = "OFF_SHELF_3D"
SHELF_OCCUPANCY_OUTSIDE = "OUTSIDE_SHELF_VOLUME"
SHELF_OCCUPANCY_UNKNOWN = "DEPTH_UNKNOWN"


@dataclass(frozen=True)
class ProductDepthObservation:
    rgb_sequence_number: int
    depth_sequence_number: int
    timestamp_delta_milliseconds: float
    roi: tuple[int, int, int, int]
    valid_pixel_count: int
    valid_fraction: float
    median_depth_mm: float
    mad_depth_mm: float
    point_3d_mm: tuple[float, float, float]


@dataclass(frozen=True)
class ShelfDepthReference:
    shelf_id: int | None
    occupancy: str
    expected_depth_mm: float | None = None
    expected_mad_depth_mm: float | None = None
    depth_residual_mm: float | None = None
    on_shelf_tolerance_mm: float | None = None
    valid_fraction: float | None = None
    grid_column: int | None = None
    grid_row: int | None = None
    calibration_id: str | None = None
    suppression_reason: str | None = None


@dataclass(frozen=True)
class TrackedProduct:
    product_track_id: int
    detection: ProductDetection
    missed_frames: int = 0


@dataclass(frozen=True)
class ProductInteractionSample:
    camera_index: int
    device_id: str
    person_track_id: int
    product_track_id: int
    visit_id: int
    customer_id: str | None
    product_class_id: int
    product_label: str
    product_score: float
    product_box: tuple[int, int, int, int]
    shelf_id: int | None
    hand: str | None
    hand_score: float
    host_synced_seconds: float
    rgb_sequence_number: int
    pose_sequence_number: int | None = None
    pose_delta_milliseconds: int | None = None
    product_depth: ProductDepthObservation | None = None
    shelf_depth_reference: ShelfDepthReference = ShelfDepthReference(
        shelf_id=None,
        occupancy=SHELF_OCCUPANCY_UNKNOWN,
        suppression_reason="missing_depth_evidence",
    )


@dataclass(frozen=True)
class ProductInteractionEvent:
    event_id: str
    event_type: str
    visit_id: int
    customer_id: str | None
    product_class_id: int
    product_label: str
    shelf_id: int | None
    occurred_host_synced_seconds: float
    confidence: float
    camera_index: int
    device_id: str
    person_track_id: int
    product_track_id: int
    rgb_sequence_number: int
    hand: str | None = None
    hand_score: float = 0.0
    pose_sequence_number: int | None = None
    pose_delta_milliseconds: int | None = None
    product_depth: ProductDepthObservation | None = None
    shelf_depth_reference: ShelfDepthReference | None = None


def product_interaction_event_payload(
    event: ProductInteractionEvent,
    *,
    status: str = "candidate",
) -> dict[str, Any]:
    depth = event.product_depth
    reference = event.shelf_depth_reference
    return {
        "eventId": event.event_id,
        "eventType": event.event_type,
        "status": status,
        "visitId": event.visit_id,
        "customerId": event.customer_id,
        "productClassId": event.product_class_id,
        "productLabel": event.product_label,
        "shelfId": event.shelf_id,
        "occurredHostSyncedSeconds": event.occurred_host_synced_seconds,
        "confidence": event.confidence,
        "cameraIndex": event.camera_index,
        "cameraNumber": event.camera_index + 1,
        "deviceId": event.device_id,
        "personTrackId": event.person_track_id,
        "productTrackId": event.product_track_id,
        "rgbSequenceNumber": event.rgb_sequence_number,
        "evidence": {
            "hand": event.hand,
            "handScore": event.hand_score,
            "poseSequenceNumber": event.pose_sequence_number,
            "poseDeltaMilliseconds": event.pose_delta_milliseconds,
            "depthConfirmed": (
                depth is not None
                and reference is not None
                and reference.occupancy != SHELF_OCCUPANCY_UNKNOWN
            ),
            "depthSequenceNumber": (
                None if depth is None else depth.depth_sequence_number
            ),
            "depthDeltaMilliseconds": (
                None if depth is None else depth.timestamp_delta_milliseconds
            ),
            "productDepthMm": None if depth is None else depth.median_depth_mm,
            "productDepthMadMm": None if depth is None else depth.mad_depth_mm,
            "productDepthValidFraction": (
                None if depth is None else depth.valid_fraction
            ),
            "productPoint3dMm": None if depth is None else depth.point_3d_mm,
            "shelfOccupancy": None if reference is None else reference.occupancy,
            "expectedShelfDepthMm": (
                None if reference is None else reference.expected_depth_mm
            ),
            "shelfDepthResidualMm": (
                None if reference is None else reference.depth_residual_mm
            ),
            "shelfCalibrationId": (
                None if reference is None else reference.calibration_id
            ),
            "depthSuppressionReason": (
                None if reference is None else reference.suppression_reason
            ),
        },
    }


@dataclass
class _TrackedProductState:
    product_track_id: int
    detection: ProductDetection
    missed_frames: int = 0


def _box_iou(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    intersection_x1 = max(left[0], right[0])
    intersection_y1 = max(left[1], right[1])
    intersection_x2 = min(left[2], right[2])
    intersection_y2 = min(left[3], right[3])
    intersection = max(0, intersection_x2 - intersection_x1) * max(
        0, intersection_y2 - intersection_y1
    )
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _center_distance_fraction(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    left_center = ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0)
    right_center = ((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0)
    scale = max(
        1.0,
        math.hypot(right[2] - right[0], right[3] - right[1]),
        math.hypot(left[2] - left[0], left[3] - left[1]),
    )
    return math.dist(left_center, right_center) / scale


class ClassAwareProductTracker:
    def __init__(
        self,
        *,
        iou_threshold: float = 0.20,
        max_center_distance_fraction: float = 1.25,
        max_missed: int = 2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_center_distance_fraction = max_center_distance_fraction
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, _TrackedProductState] = {}

    def update(self, detections: Sequence[ProductDetection]) -> tuple[TrackedProduct, ...]:
        available_tracks = set(self._tracks)
        assignments: list[tuple[int, ProductDetection]] = []
        unmatched: list[ProductDetection] = []
        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            detection_box = (detection.x1, detection.y1, detection.x2, detection.y2)
            candidates: list[tuple[float, int]] = []
            for track_id in available_tracks:
                state = self._tracks[track_id]
                if state.detection.class_id != detection.class_id:
                    continue
                track_box = (
                    state.detection.x1,
                    state.detection.y1,
                    state.detection.x2,
                    state.detection.y2,
                )
                iou = _box_iou(track_box, detection_box)
                distance = _center_distance_fraction(track_box, detection_box)
                if iou >= self.iou_threshold or distance <= self.max_center_distance_fraction:
                    candidates.append((iou - distance * 0.1, track_id))
            if not candidates:
                unmatched.append(detection)
                continue
            _score, selected = max(candidates)
            available_tracks.remove(selected)
            assignments.append((selected, detection))

        for track_id, detection in assignments:
            state = self._tracks[track_id]
            state.detection = detection
            state.missed_frames = 0
        for track_id in available_tracks:
            self._tracks[track_id].missed_frames += 1
        for track_id in tuple(self._tracks):
            if self._tracks[track_id].missed_frames > self.max_missed:
                del self._tracks[track_id]
        for detection in unmatched:
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _TrackedProductState(track_id, detection)

        return tuple(
            TrackedProduct(track_id, state.detection, state.missed_frames)
            for track_id, state in sorted(self._tracks.items())
            if state.missed_frames == 0
        )


def product_shelf_id(
    detection: ProductDetection,
    *,
    frame_width: int,
    frame_height: int,
    shelf_regions: Mapping[int, ShelfRegion],
) -> int | None:
    box = (detection.x1, detection.y1, detection.x2, detection.y2)
    containing = [
        shelf_id
        for shelf_id, region in shelf_regions.items()
        if box_center_in_shelf_region(
            box,
            frame_width=frame_width,
            frame_height=frame_height,
            region=region,
        )
    ]
    return min(containing) if containing else None


def sample_product_depth(
    depth_frame_mm: np.ndarray,
    detection: ProductDetection,
    *,
    source_frame_width: int,
    source_frame_height: int,
    rgb_sequence_number: int,
    depth_sequence_number: int,
    timestamp_delta_milliseconds: float,
    intrinsics: CameraIntrinsics,
    roi_fraction: float = 0.5,
    minimum_valid_pixels: int = 16,
    minimum_valid_fraction: float = 0.20,
) -> ProductDepthObservation | None:
    if depth_frame_mm.ndim != 2:
        raise ValueError("Depth frame must be a single-channel millimeter image.")
    if source_frame_width <= 0 or source_frame_height <= 0:
        raise ValueError("Source frame dimensions must be positive.")
    if not 0.0 < roi_fraction <= 1.0:
        raise ValueError("Product depth ROI fraction must be between zero and one.")
    depth_height, depth_width = depth_frame_mm.shape
    source_center_x = (detection.x1 + detection.x2) / 2.0
    source_center_y = (detection.y1 + detection.y2) / 2.0
    source_width = max(1.0, detection.x2 - detection.x1)
    source_height = max(1.0, detection.y2 - detection.y1)
    center_x = source_center_x / source_frame_width * depth_width
    center_y = source_center_y / source_frame_height * depth_height
    roi_width = max(4, int(round(source_width / source_frame_width * depth_width * roi_fraction)))
    roi_height = max(4, int(round(source_height / source_frame_height * depth_height * roi_fraction)))
    x1 = max(0, int(round(center_x - roi_width / 2.0)))
    y1 = max(0, int(round(center_y - roi_height / 2.0)))
    x2 = min(depth_width, x1 + roi_width)
    y2 = min(depth_height, y1 + roi_height)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth_frame_mm[y1:y2, x1:x2]
    valid = roi[(roi > 0) & np.isfinite(roi)]
    valid_fraction = valid.size / roi.size
    if valid.size < minimum_valid_pixels or valid_fraction < minimum_valid_fraction:
        return None
    values = valid.astype(np.float32, copy=False)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    anchor_x = int(round((x1 + x2 - 1) / 2.0))
    anchor_y = int(round((y1 + y2 - 1) / 2.0))
    return ProductDepthObservation(
        rgb_sequence_number=rgb_sequence_number,
        depth_sequence_number=depth_sequence_number,
        timestamp_delta_milliseconds=float(timestamp_delta_milliseconds),
        roi=(x1, y1, x2, y2),
        valid_pixel_count=int(valid.size),
        valid_fraction=float(valid_fraction),
        median_depth_mm=median,
        mad_depth_mm=mad,
        point_3d_mm=pixel_to_camera_point_mm(
            pixel_x=anchor_x,
            pixel_y=anchor_y,
            depth_mm=median,
            intrinsics=intrinsics,
        ),
    )


def _reference_from_cell(
    *,
    shelf_id: int,
    region: ShelfRegion,
    cell: ShelfDepthCell,
    product_depth: ProductDepthObservation,
) -> ShelfDepthReference:
    base_tolerance = region.depth_tolerance_mm or 250.0
    adaptive_tolerance = min(
        600.0,
        max(
            base_tolerance,
            3.0 * cell.mad_depth_mm + 2.0 * product_depth.mad_depth_mm,
        ),
    )
    residual = product_depth.median_depth_mm - cell.median_depth_mm
    absolute_residual = abs(residual)
    if absolute_residual <= adaptive_tolerance:
        occupancy = SHELF_OCCUPANCY_ON
        reason = None
    elif absolute_residual >= adaptive_tolerance + region.off_shelf_hysteresis_mm:
        occupancy = SHELF_OCCUPANCY_OFF
        reason = None
    else:
        occupancy = SHELF_OCCUPANCY_UNKNOWN
        reason = "shelf_depth_hysteresis_band"
    model = region.depth_model
    return ShelfDepthReference(
        shelf_id=shelf_id,
        occupancy=occupancy,
        expected_depth_mm=cell.median_depth_mm,
        expected_mad_depth_mm=cell.mad_depth_mm,
        depth_residual_mm=residual,
        on_shelf_tolerance_mm=adaptive_tolerance,
        valid_fraction=cell.valid_fraction,
        grid_column=cell.column,
        grid_row=cell.row,
        calibration_id=None if model is None else model.camera_calibration_id,
        suppression_reason=reason,
    )


def product_shelf_depth_reference(
    detection: ProductDetection,
    *,
    source_frame_width: int,
    source_frame_height: int,
    shelf_regions: Mapping[int, ShelfRegion],
    product_depth: ProductDepthObservation | None,
) -> ShelfDepthReference:
    normalized_x = (detection.x1 + detection.x2) / 2.0 / source_frame_width
    normalized_y = (detection.y1 + detection.y2) / 2.0 / source_frame_height
    containing = [
        (shelf_id, region)
        for shelf_id, region in shelf_regions.items()
        if box_center_in_shelf_region(
            (detection.x1, detection.y1, detection.x2, detection.y2),
            frame_width=source_frame_width,
            frame_height=source_frame_height,
            region=region,
        )
    ]
    shelf_id = min((item[0] for item in containing), default=None)
    if product_depth is None:
        return ShelfDepthReference(
            shelf_id=shelf_id,
            occupancy=SHELF_OCCUPANCY_UNKNOWN,
            suppression_reason="invalid_product_depth",
        )
    if not containing:
        return ShelfDepthReference(
            shelf_id=None,
            occupancy=SHELF_OCCUPANCY_OUTSIDE,
            suppression_reason=None,
        )
    references = []
    for candidate_shelf_id, region in containing:
        cell = shelf_depth_cell_at(
            region,
            normalized_x=normalized_x,
            normalized_y=normalized_y,
        )
        if cell is not None:
            references.append(
                _reference_from_cell(
                    shelf_id=candidate_shelf_id,
                    region=region,
                    cell=cell,
                    product_depth=product_depth,
                )
            )
    if not references:
        return ShelfDepthReference(
            shelf_id=shelf_id,
            occupancy=SHELF_OCCUPANCY_UNKNOWN,
            suppression_reason="missing_local_shelf_depth_reference",
        )
    return min(
        references,
        key=lambda reference: abs(reference.depth_residual_mm or 0.0),
    )


def hand_product_association(
    detection: ProductDetection,
    pose: PoseObservation | None,
    *,
    frame_width: int,
    frame_height: int,
    keypoint_threshold: float,
) -> tuple[str | None, float]:
    if pose is None:
        return None, 0.0
    if pose.source_frame_width > 0 and pose.source_frame_height > 0:
        person_width = max(
            1.0,
            (pose.person_box[2] - pose.person_box[0])
            / pose.source_frame_width
            * frame_width,
        )
        person_height = max(
            1.0,
            (pose.person_box[3] - pose.person_box[1])
            / pose.source_frame_height
            * frame_height,
        )
    else:
        person_width = max(1, pose.person_box[2] - pose.person_box[0])
        person_height = max(1, pose.person_box[3] - pose.person_box[1])
    person_diagonal = math.hypot(person_width, person_height)
    product_center = (
        (detection.x1 + detection.x2) / 2.0,
        (detection.y1 + detection.y2) / 2.0,
    )
    candidates: list[tuple[float, str]] = []
    by_name = {landmark.name: landmark for landmark in pose.landmarks}
    for hand in ("left", "right"):
        wrist = by_name.get(f"{hand}_wrist")
        if wrist is None or wrist.score < keypoint_threshold:
            continue
        wrist_point = (wrist.x * frame_width, wrist.y * frame_height)
        distance_fraction = math.dist(product_center, wrist_point) / max(
            1.0, person_diagonal
        )
        proximity = max(0.0, 1.0 - distance_fraction / 0.35)
        candidates.append((proximity * wrist.score, hand))
    if not candidates:
        return None, 0.0
    score, hand = max(candidates)
    return hand, float(score)


@dataclass
class _InteractionHypothesis:
    product_class_id: int | None = None
    state: str = INTERACTION_STATE_UNKNOWN
    shelf_id: int | None = None
    state_since: float = 0.0
    stable_since: float = 0.0
    shelf_point_3d_mm: tuple[float, float, float] | None = None
    contact_score_observed: float = 0.0


@dataclass
class _ShelfContactCandidate:
    first_seen_seconds: float
    last_sample: ProductInteractionSample


@dataclass
class _ReturnPlacementCandidate:
    shelf_id: int
    first_seen_seconds: float
    last_seen_seconds: float
    carry_contact_score: float
    last_sample: ProductInteractionSample
    weak_contact_since_by_context: dict[tuple[int, int, int], float] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _ActiveProductPickup:
    event: ProductInteractionEvent
    expires_at_seconds: float


class ProductInteractionStateMachine:
    def __init__(
        self,
        *,
        shelf_stable_seconds: float = 0.4,
        contact_score: float = 0.35,
        release_score: float = 0.15,
        outside_confirm_seconds: float = 0.25,
        missing_confirm_seconds: float = 0.75,
        return_confirm_seconds: float = 0.4,
        minimum_off_shelf_displacement_mm: float = 150.0,
    ) -> None:
        self.shelf_stable_seconds = shelf_stable_seconds
        self.contact_score = contact_score
        self.release_score = release_score
        self.outside_confirm_seconds = outside_confirm_seconds
        self.missing_confirm_seconds = missing_confirm_seconds
        self.return_confirm_seconds = return_confirm_seconds
        self.minimum_off_shelf_displacement_mm = minimum_off_shelf_displacement_mm
        self._hypotheses: dict[
            tuple[int, int, int, int], _InteractionHypothesis
        ] = {}

    @staticmethod
    def _key(sample: ProductInteractionSample) -> tuple[int, int, int, int]:
        return (
            sample.camera_index,
            sample.person_track_id,
            sample.visit_id,
            sample.product_track_id,
        )

    def _hypothesis(self, sample: ProductInteractionSample) -> _InteractionHypothesis:
        hypothesis = self._hypotheses.setdefault(
            self._key(sample), _InteractionHypothesis()
        )
        hypothesis.product_class_id = sample.product_class_id
        return hypothesis

    def reset_product_class(self, product_class_id: int) -> None:
        """Discard competing track hypotheses after one physical event wins."""
        for key, hypothesis in tuple(self._hypotheses.items()):
            if hypothesis.product_class_id == product_class_id:
                del self._hypotheses[key]

    def prime_return(self, sample: ProductInteractionSample) -> None:
        """Transfer recent carry evidence into a new track/visit hypothesis."""
        hypothesis = self._hypothesis(sample)
        if hypothesis.state != INTERACTION_STATE_UNKNOWN:
            return
        hypothesis.state = INTERACTION_STATE_HELD
        hypothesis.state_since = sample.host_synced_seconds
        hypothesis.contact_score_observed = sample.hand_score

    def prime_pick_departure(
        self,
        sample: ProductInteractionSample,
        shelf_sample: ProductInteractionSample,
    ) -> None:
        """Transfer shelf contact across a product-track split in one camera."""
        hypothesis = self._hypothesis(sample)
        if hypothesis.state not in {
            INTERACTION_STATE_UNKNOWN,
            INTERACTION_STATE_ON_SHELF,
        }:
            return
        hypothesis.state = INTERACTION_STATE_HAND_CONTACT
        hypothesis.shelf_id = shelf_sample.shelf_id
        hypothesis.state_since = sample.host_synced_seconds
        hypothesis.stable_since = shelf_sample.host_synced_seconds
        hypothesis.shelf_point_3d_mm = (
            None
            if shelf_sample.product_depth is None
            else shelf_sample.product_depth.point_3d_mm
        )
        hypothesis.contact_score_observed = sample.hand_score

    def prime_return_placing(
        self,
        sample: ProductInteractionSample,
        *,
        contact_score: float,
        stable_since: float,
    ) -> None:
        """Transfer shelf-placement evidence across product-track boundaries."""
        hypothesis = self._hypothesis(sample)
        hypothesis.state = INTERACTION_STATE_RETURN_PLACING
        hypothesis.shelf_id = sample.shelf_id
        hypothesis.state_since = sample.host_synced_seconds
        hypothesis.stable_since = stable_since
        hypothesis.contact_score_observed = max(
            hypothesis.contact_score_observed,
            contact_score,
        )

    def update(self, sample: ProductInteractionSample) -> ProductInteractionEvent | None:
        hypothesis = self._hypothesis(sample)
        now = sample.host_synced_seconds
        if hypothesis.state == INTERACTION_STATE_UNKNOWN:
            if sample.shelf_depth_reference.occupancy == SHELF_OCCUPANCY_ON:
                hypothesis.state = INTERACTION_STATE_ON_SHELF
                hypothesis.shelf_id = sample.shelf_id
                hypothesis.state_since = now
                hypothesis.stable_since = now
                hypothesis.shelf_point_3d_mm = (
                    None
                    if sample.product_depth is None
                    else sample.product_depth.point_3d_mm
                )
            elif (
                sample.shelf_depth_reference.occupancy
                in {SHELF_OCCUPANCY_OFF, SHELF_OCCUPANCY_OUTSIDE}
                and sample.product_depth is not None
                and sample.hand_score >= self.contact_score
            ):
                # A return is independently observable; no earlier pickup is required.
                hypothesis.state = INTERACTION_STATE_HELD
                hypothesis.state_since = now
                hypothesis.contact_score_observed = sample.hand_score
            return None

        if hypothesis.state == INTERACTION_STATE_ON_SHELF:
            occupancy = sample.shelf_depth_reference.occupancy
            if occupancy == SHELF_OCCUPANCY_UNKNOWN:
                return None
            if occupancy != SHELF_OCCUPANCY_ON or sample.shelf_id != hypothesis.shelf_id:
                hypothesis.state = INTERACTION_STATE_UNKNOWN
                hypothesis.shelf_id = sample.shelf_id
                hypothesis.state_since = now
                return None
            if sample.product_depth is not None:
                hypothesis.shelf_point_3d_mm = sample.product_depth.point_3d_mm
            if (
                now - hypothesis.stable_since >= self.shelf_stable_seconds
                and sample.hand_score >= self.contact_score
            ):
                hypothesis.state = INTERACTION_STATE_HAND_CONTACT
                hypothesis.state_since = now
                hypothesis.contact_score_observed = sample.hand_score
            return None

        if hypothesis.state == INTERACTION_STATE_HAND_CONTACT:
            occupancy = sample.shelf_depth_reference.occupancy
            if occupancy == SHELF_OCCUPANCY_UNKNOWN:
                return None
            if occupancy == SHELF_OCCUPANCY_ON:
                if sample.hand_score < self.release_score:
                    hypothesis.state = INTERACTION_STATE_ON_SHELF
                    hypothesis.stable_since = now
                return None
            if sample.hand_score < self.contact_score:
                hypothesis.state = INTERACTION_STATE_UNKNOWN
                return None
            if not self._has_required_3d_departure(sample, hypothesis):
                return None
            if now - hypothesis.state_since < self.outside_confirm_seconds:
                return None
            hypothesis.state = INTERACTION_STATE_HELD
            hypothesis.state_since = now
            hypothesis.contact_score_observed = max(
                hypothesis.contact_score_observed,
                sample.hand_score,
            )
            return self._event(EVENT_PRODUCT_PICKED, sample, hypothesis.shelf_id)

        if hypothesis.state == INTERACTION_STATE_HELD:
            if (
                sample.shelf_depth_reference.occupancy == SHELF_OCCUPANCY_ON
                and sample.shelf_id is not None
                and sample.hand_score >= self.contact_score
            ):
                hypothesis.state = INTERACTION_STATE_RETURN_PLACING
                hypothesis.shelf_id = sample.shelf_id
                hypothesis.state_since = now
                hypothesis.stable_since = now
                hypothesis.contact_score_observed = max(
                    hypothesis.contact_score_observed,
                    sample.hand_score,
                )
            return None

        if hypothesis.state == INTERACTION_STATE_RETURN_PLACING:
            occupancy = sample.shelf_depth_reference.occupancy
            if occupancy == SHELF_OCCUPANCY_UNKNOWN:
                return None
            if occupancy != SHELF_OCCUPANCY_ON or sample.shelf_id != hypothesis.shelf_id:
                hypothesis.state = INTERACTION_STATE_HELD
                hypothesis.shelf_id = None
                return None
            # A product can be stably back on the shelf while the hand remains
            # nearby. Require loss of positive contact, not complete pose-box
            # separation, otherwise ordinary releases around 0.3 never close.
            if sample.hand_score >= self.contact_score:
                hypothesis.stable_since = now
                return None
            if now - hypothesis.stable_since < self.return_confirm_seconds:
                return None
            hypothesis.state = INTERACTION_STATE_ON_SHELF
            hypothesis.state_since = now
            hypothesis.stable_since = now
            return self._event(
                EVENT_PRODUCT_RETURNED,
                sample,
                sample.shelf_id,
                confidence_hand_score=hypothesis.contact_score_observed,
            )
        return None

    def update_missing(
        self,
        sample: ProductInteractionSample,
        now: float,
    ) -> ProductInteractionEvent | None:
        """Confirm a pickup when a contacted shelf product disappears from view."""
        hypothesis = self._hypotheses.get(self._key(sample))
        if hypothesis is None or hypothesis.state != INTERACTION_STATE_HAND_CONTACT:
            return None
        if now - sample.host_synced_seconds < self.missing_confirm_seconds:
            return None
        hypothesis.state = INTERACTION_STATE_HELD
        hypothesis.state_since = now
        return self._event(
            EVENT_PRODUCT_PICKED,
            replace(sample, host_synced_seconds=now),
            hypothesis.shelf_id,
            confidence_hand_score=hypothesis.contact_score_observed,
        )

    def confirm_return(
        self,
        sample: ProductInteractionSample,
        *,
        carry_contact_score: float,
    ) -> ProductInteractionEvent:
        """Create a return confirmed by camera-scoped placement evidence."""
        return self._event(
            EVENT_PRODUCT_RETURNED,
            sample,
            sample.shelf_id,
            confidence_hand_score=carry_contact_score,
        )

    def _has_required_3d_departure(
        self,
        sample: ProductInteractionSample,
        hypothesis: _InteractionHypothesis,
    ) -> bool:
        occupancy = sample.shelf_depth_reference.occupancy
        if occupancy == SHELF_OCCUPANCY_OFF:
            return sample.product_depth is not None
        if occupancy != SHELF_OCCUPANCY_OUTSIDE or sample.product_depth is None:
            return False
        if hypothesis.shelf_point_3d_mm is None:
            return False
        return (
            math.dist(
                hypothesis.shelf_point_3d_mm,
                sample.product_depth.point_3d_mm,
            )
            >= self.minimum_off_shelf_displacement_mm
        )

    @staticmethod
    def _event(
        event_type: str,
        sample: ProductInteractionSample,
        shelf_id: int | None,
        *,
        confidence_hand_score: float | None = None,
    ) -> ProductInteractionEvent:
        hand_confidence = (
            sample.hand_score
            if confidence_hand_score is None
            else confidence_hand_score
        )
        confidence = min(
            1.0,
            max(0.0, hand_confidence)
            * sample.product_score
            * (0.8 + 0.2 * (sample.product_depth.valid_fraction if sample.product_depth else 0.0)),
        )
        return ProductInteractionEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            visit_id=sample.visit_id,
            customer_id=sample.customer_id,
            product_class_id=sample.product_class_id,
            product_label=sample.product_label,
            shelf_id=shelf_id,
            occurred_host_synced_seconds=sample.host_synced_seconds,
            confidence=confidence,
            camera_index=sample.camera_index,
            device_id=sample.device_id,
            person_track_id=sample.person_track_id,
            product_track_id=sample.product_track_id,
            rgb_sequence_number=sample.rgb_sequence_number,
            hand=sample.hand,
            hand_score=sample.hand_score,
            pose_sequence_number=sample.pose_sequence_number,
            pose_delta_milliseconds=sample.pose_delta_milliseconds,
            product_depth=sample.product_depth,
            shelf_depth_reference=sample.shelf_depth_reference,
        )


@dataclass
class ProductInteractionCoordinator:
    state_machine: ProductInteractionStateMachine = field(
        default_factory=ProductInteractionStateMachine
    )
    pose_max_delta_seconds: float = 0.5
    keypoint_threshold: float = 0.35
    _pose_by_camera_track: dict[tuple[int, int], deque[PoseObservation]] = field(
        default_factory=dict
    )
    _trackers: dict[tuple[int, int], ClassAwareProductTracker] = field(
        default_factory=dict
    )
    duplicate_event_window_seconds: float = 1.0
    carry_transfer_window_seconds: float = 10.0
    placement_transfer_window_seconds: float = 2.0
    shelf_contact_transfer_window_seconds: float = 3.0
    shelf_presence_transfer_window_seconds: float = 5.0
    carry_end_confirm_seconds: float = 0.5
    active_pick_timeout_seconds: float = 300.0
    cross_camera_return_min_seconds: float = 0.5
    post_return_pick_cooldown_seconds: float = 5.0
    _last_event_time_by_signature: dict[tuple[int, str, int | None], float] = field(
        default_factory=dict
    )
    _recent_carried_by_context: dict[tuple[int, int], ProductInteractionSample] = field(
        default_factory=dict
    )
    _recent_shelf_contact_by_context: dict[
        tuple[int, int], ProductInteractionSample
    ] = field(default_factory=dict)
    _recent_shelf_presence_by_context: dict[
        tuple[int, int], ProductInteractionSample
    ] = field(default_factory=dict)
    _shelf_contact_candidates: dict[
        tuple[int, int, int], _ShelfContactCandidate
    ] = field(default_factory=dict)
    _shelf_presence_candidates: dict[
        tuple[int, int, int], _ShelfContactCandidate
    ] = field(default_factory=dict)
    _recent_placement_by_context: dict[
        tuple[int, int], _ReturnPlacementCandidate
    ] = field(default_factory=dict)
    _recent_carried_by_product_class: dict[int, ProductInteractionSample] = field(
        default_factory=dict
    )
    _recent_placement_by_product_shelf: dict[
        tuple[int, int], _ReturnPlacementCandidate
    ] = field(default_factory=dict)
    _current_samples: dict[tuple[int, int, int], ProductInteractionSample] = field(
        default_factory=dict
    )
    _active_pick_by_product_shelf: dict[
        tuple[int, int], _ActiveProductPickup
    ] = field(default_factory=dict)
    _last_return_time_by_product_shelf: dict[tuple[int, int], float] = field(
        default_factory=dict
    )
    last_tracker_milliseconds: float = 0.0
    last_coordinator_milliseconds: float = 0.0

    def publish_pose(self, observation: PoseObservation) -> None:
        key = (observation.camera_index, observation.track_id)
        history = self._pose_by_camera_track.setdefault(key, deque(maxlen=64))
        history.append(observation)

    def nearest_pose(
        self,
        camera_index: int,
        track_id: int,
        host_synced_seconds: float,
    ) -> PoseObservation | None:
        history = self._pose_by_camera_track.get((camera_index, track_id))
        if not history:
            return None
        pose = min(
            history,
            key=lambda item: abs(host_synced_seconds - item.host_synced_seconds),
        )
        if abs(host_synced_seconds - pose.host_synced_seconds) > self.pose_max_delta_seconds:
            return None
        return pose

    def process(
        self,
        *,
        camera_index: int,
        device_id: str,
        person_track_id: int,
        visit_id: int,
        customer_id: str | None,
        rgb_sequence_number: int,
        host_synced_seconds: float,
        detections: Sequence[ProductDetection],
        frame_width: int,
        frame_height: int,
        shelf_regions: Mapping[int, ShelfRegion],
        depth_frame_mm: np.ndarray | None,
        depth_sequence_number: int | None,
        depth_delta_milliseconds: float | None,
        intrinsics: CameraIntrinsics,
    ) -> tuple[ProductInteractionEvent, ...]:
        coordinator_started = time.perf_counter()
        for key, previous in tuple(self._current_samples.items()):
            if host_synced_seconds - previous.host_synced_seconds > 3.0:
                del self._current_samples[key]
        tracker_key = (camera_index, person_track_id)
        tracker = self._trackers.setdefault(tracker_key, ClassAwareProductTracker())
        tracker_started = time.perf_counter()
        tracked_products = tracker.update(detections)
        self.last_tracker_milliseconds = (
            time.perf_counter() - tracker_started
        ) * 1000.0
        pose = self.nearest_pose(camera_index, person_track_id, host_synced_seconds)
        pose_delta_seconds = (
            None
            if pose is None
            else abs(host_synced_seconds - pose.host_synced_seconds)
        )
        events: list[ProductInteractionEvent] = []
        visible_product_track_ids = {
            tracked.product_track_id for tracked in tracked_products
        }
        for key, previous_sample in tuple(self._current_samples.items()):
            previous_camera, previous_person_track, previous_product_track = key
            if (
                previous_camera != camera_index
                or previous_person_track != person_track_id
                or previous_product_track in visible_product_track_ids
            ):
                continue
            missing_event = self.state_machine.update_missing(
                previous_sample,
                host_synced_seconds,
            )
            if missing_event is not None:
                inferred_carry = replace(
                    previous_sample,
                    host_synced_seconds=host_synced_seconds,
                    shelf_depth_reference=replace(
                        previous_sample.shelf_depth_reference,
                        occupancy=SHELF_OCCUPANCY_OFF,
                    ),
                )
                self._recent_carried_by_context[
                    (camera_index, previous_sample.product_class_id)
                ] = inferred_carry
                self._accept_event(missing_event, events)
        for tracked in tracked_products:
            detection = tracked.detection
            product_depth = (
                None
                if depth_frame_mm is None
                or depth_sequence_number is None
                or depth_delta_milliseconds is None
                else sample_product_depth(
                    depth_frame_mm,
                    detection,
                    source_frame_width=frame_width,
                    source_frame_height=frame_height,
                    rgb_sequence_number=rgb_sequence_number,
                    depth_sequence_number=depth_sequence_number,
                    timestamp_delta_milliseconds=depth_delta_milliseconds,
                    intrinsics=intrinsics,
                )
            )
            shelf_reference = product_shelf_depth_reference(
                detection,
                source_frame_width=frame_width,
                source_frame_height=frame_height,
                shelf_regions=shelf_regions,
                product_depth=product_depth,
            )
            hand, hand_score = hand_product_association(
                detection,
                pose,
                frame_width=frame_width,
                frame_height=frame_height,
                keypoint_threshold=self.keypoint_threshold,
            )
            sample = ProductInteractionSample(
                    camera_index=camera_index,
                    device_id=device_id,
                    person_track_id=person_track_id,
                    product_track_id=tracked.product_track_id,
                    visit_id=visit_id,
                    customer_id=customer_id,
                    product_class_id=detection.class_id,
                    product_label=detection.label,
                    product_score=detection.score,
                    product_box=(detection.x1, detection.y1, detection.x2, detection.y2),
                    shelf_id=shelf_reference.shelf_id,
                    hand=hand,
                    hand_score=hand_score,
                    host_synced_seconds=host_synced_seconds,
                    rgb_sequence_number=rgb_sequence_number,
                    pose_sequence_number=(
                        None if pose is None else pose.rgb_sequence_number
                    ),
                    pose_delta_milliseconds=(
                        None
                        if pose is None or pose_delta_seconds is None
                        else int(round(pose_delta_seconds * 1000.0))
                    ),
                    product_depth=product_depth,
                    shelf_depth_reference=shelf_reference,
                )
            self._current_samples[
                (camera_index, person_track_id, tracked.product_track_id)
            ] = sample
            context_event = self._apply_carry_context(sample)
            if context_event is not None:
                self._accept_event(context_event, events)
            event = self.state_machine.update(sample)
            if event is not None:
                self._accept_event(event, events)
        self.last_coordinator_milliseconds = (
            time.perf_counter() - coordinator_started
        ) * 1000.0
        return tuple(events)

    def _apply_carry_context(
        self,
        sample: ProductInteractionSample,
    ) -> ProductInteractionEvent | None:
        occupancy = sample.shelf_depth_reference.occupancy
        now = sample.host_synced_seconds
        self._expire_active_picks(now)
        context = (sample.camera_index, sample.product_class_id)
        contact_key = (
            sample.camera_index,
            sample.person_track_id,
            sample.product_track_id,
        )
        previous = self._recent_carried_by_context.get(context)
        shelf_contact = self._recent_shelf_contact_by_context.get(context)
        shelf_presence = self._recent_shelf_presence_by_context.get(context)
        global_previous = self._recent_carried_by_product_class.get(
            sample.product_class_id
        )
        placement_key = (
            None
            if sample.shelf_id is None
            else (sample.product_class_id, sample.shelf_id)
        )
        placement = (
            self._recent_placement_by_context.get(context)
            if placement_key is None
            else self._recent_placement_by_product_shelf.get(placement_key)
        )
        if (
            previous is not None
            and now - previous.host_synced_seconds
            > self.carry_transfer_window_seconds
        ):
            self._recent_carried_by_context.pop(context, None)
            previous = None
        if (
            shelf_contact is not None
            and now - shelf_contact.host_synced_seconds
            > self.shelf_contact_transfer_window_seconds
        ):
            self._recent_shelf_contact_by_context.pop(context, None)
            shelf_contact = None
        if (
            shelf_presence is not None
            and now - shelf_presence.host_synced_seconds
            > self.shelf_presence_transfer_window_seconds
        ):
            self._recent_shelf_presence_by_context.pop(context, None)
            shelf_presence = None
        if (
            global_previous is not None
            and now - global_previous.host_synced_seconds
            > self.carry_transfer_window_seconds
        ):
            self._recent_carried_by_product_class.pop(
                sample.product_class_id,
                None,
            )
            global_previous = None
        if (
            placement is not None
            and now - placement.last_seen_seconds
            > self.placement_transfer_window_seconds
        ):
            self._recent_placement_by_context.pop(context, None)
            if placement_key is not None:
                self._recent_placement_by_product_shelf.pop(placement_key, None)
            placement = None

        if occupancy == SHELF_OCCUPANCY_ON:
            presence_candidate = self._shelf_presence_candidates.get(contact_key)
            if (
                presence_candidate is None
                or presence_candidate.last_sample.shelf_id != sample.shelf_id
            ):
                presence_candidate = _ShelfContactCandidate(
                    first_seen_seconds=now,
                    last_sample=sample,
                )
                self._shelf_presence_candidates[contact_key] = presence_candidate
            else:
                presence_candidate.last_sample = sample
            if (
                now - presence_candidate.first_seen_seconds
                >= self.state_machine.shelf_stable_seconds
            ):
                self._recent_shelf_presence_by_context[context] = sample
        else:
            self._shelf_presence_candidates.pop(contact_key, None)

        if (
            occupancy in {SHELF_OCCUPANCY_OFF, SHELF_OCCUPANCY_OUTSIDE}
            and sample.product_depth is not None
            and sample.hand_score >= self.state_machine.contact_score
        ):
            self._shelf_contact_candidates.pop(contact_key, None)
            self._recent_placement_by_context.pop(context, None)
            if (
                shelf_contact is not None
                and 0.0 <= now - shelf_contact.host_synced_seconds
                <= self.shelf_contact_transfer_window_seconds
            ):
                self.state_machine.prime_pick_departure(sample, shelf_contact)
            elif (
                shelf_presence is not None
                and 0.0 <= now - shelf_presence.host_synced_seconds
                <= self.shelf_presence_transfer_window_seconds
            ):
                # Product detectors commonly assign a new ID as the hand lifts
                # an item. Stable shelf presence immediately followed by a
                # hand-associated departure is sufficient continuity evidence.
                self.state_machine.prime_pick_departure(sample, shelf_presence)
            self._recent_carried_by_context[context] = sample
            self._recent_carried_by_product_class[sample.product_class_id] = sample
            return None

        if (
            occupancy == SHELF_OCCUPANCY_ON
            and sample.hand_score >= self.state_machine.contact_score
        ):
            candidate = self._shelf_contact_candidates.get(contact_key)
            if candidate is None or candidate.last_sample.shelf_id != sample.shelf_id:
                candidate = _ShelfContactCandidate(
                    first_seen_seconds=now,
                    last_sample=sample,
                )
                self._shelf_contact_candidates[contact_key] = candidate
            else:
                candidate.last_sample = sample
            if (
                now - candidate.first_seen_seconds
                >= self.state_machine.shelf_stable_seconds
            ):
                self._recent_shelf_contact_by_context[context] = sample
        elif (
            occupancy != SHELF_OCCUPANCY_ON
            or sample.hand_score < self.state_machine.release_score
        ):
            self._shelf_contact_candidates.pop(contact_key, None)

        return_carry = previous if previous is not None else global_previous
        carry_has_ended_near_shelf = (
            return_carry is not None
            and return_carry.shelf_depth_reference.occupancy
            in {SHELF_OCCUPANCY_OFF, SHELF_OCCUPANCY_OUTSIDE}
            and self.carry_end_confirm_seconds
            <= now - return_carry.host_synced_seconds
            <= self.carry_transfer_window_seconds
            and (
                previous is not None
                or sample.hand_score >= self.state_machine.contact_score
            )
        )
        active_pick = max(
            (
                active
                for (product_class_id, _shelf_id), active
                in self._active_pick_by_product_shelf.items()
                if product_class_id == sample.product_class_id
            ),
            key=lambda active: active.event.occurred_host_synced_seconds,
            default=None,
        )
        active_pick_can_seed_return = (
            active_pick is not None
            and now - active_pick.event.occurred_host_synced_seconds
            >= self.cross_camera_return_min_seconds
            and sample.hand_score >= self.state_machine.contact_score
        )
        if (
            occupancy == SHELF_OCCUPANCY_ON
            and (carry_has_ended_near_shelf or active_pick_can_seed_return)
            and placement is None
        ):
            carry_contact_score = (
                return_carry.hand_score
                if carry_has_ended_near_shelf and return_carry is not None
                else active_pick.event.hand_score
            )
            placement = _ReturnPlacementCandidate(
                shelf_id=sample.shelf_id,
                first_seen_seconds=now,
                last_seen_seconds=now,
                carry_contact_score=carry_contact_score,
                last_sample=sample,
            )
            self._recent_placement_by_context[context] = placement
            if placement_key is not None:
                self._recent_placement_by_product_shelf[placement_key] = placement
            self.state_machine.prime_return_placing(
                sample,
                contact_score=placement.carry_contact_score,
                stable_since=placement.first_seen_seconds,
            )

        if (
            occupancy == SHELF_OCCUPANCY_ON
            and placement is not None
        ):
            if placement.shelf_id != sample.shelf_id:
                self._recent_placement_by_context.pop(context, None)
                return None
            placement.last_seen_seconds = now
            placement.last_sample = sample
            release_context = (
                sample.camera_index,
                sample.person_track_id,
                sample.product_track_id,
            )
            if sample.hand_score >= self.state_machine.contact_score:
                placement.weak_contact_since_by_context.pop(
                    release_context,
                    None,
                )
                return None
            weak_contact_since = placement.weak_contact_since_by_context.get(
                release_context
            )
            if weak_contact_since is None:
                matching_release_times = (
                    started_at
                    for (camera_index, person_track_id, _product_track_id), started_at
                    in placement.weak_contact_since_by_context.items()
                    if camera_index == sample.camera_index
                    and person_track_id == sample.person_track_id
                )
                weak_contact_since = min(matching_release_times, default=None)
            if weak_contact_since is None:
                weak_contact_since = now
                placement.weak_contact_since_by_context[release_context] = (
                    weak_contact_since
                )
                self.state_machine.prime_return_placing(
                    sample,
                    contact_score=placement.carry_contact_score,
                    stable_since=now,
                )
                return None
            placement.weak_contact_since_by_context[release_context] = (
                weak_contact_since
            )
            self.state_machine.prime_return_placing(
                sample,
                contact_score=placement.carry_contact_score,
                stable_since=weak_contact_since,
            )
            if (
                now - weak_contact_since
                < self.state_machine.return_confirm_seconds
            ):
                return None
            return self.state_machine.confirm_return(
                sample,
                carry_contact_score=placement.carry_contact_score,
            )
        return None

    def _accept_event(
        self,
        event: ProductInteractionEvent,
        events: list[ProductInteractionEvent],
    ) -> None:
        self._expire_active_picks(event.occurred_host_synced_seconds)
        cycle_key = (
            None
            if event.shelf_id is None
            else (event.product_class_id, event.shelf_id)
        )
        if event.event_type == EVENT_PRODUCT_PICKED and cycle_key is not None:
            active_pick = self._active_pick_by_product_shelf.get(cycle_key)
            last_return = self._last_return_time_by_product_shelf.get(cycle_key)
            in_return_cooldown = (
                last_return is not None
                and event.occurred_host_synced_seconds - last_return
                < self.post_return_pick_cooldown_seconds
            )
            if active_pick is not None or in_return_cooldown:
                self.state_machine.reset_product_class(event.product_class_id)
                if in_return_cooldown:
                    self._clear_interaction_context(event.product_class_id)
                else:
                    self._clear_pick_contact_context(event.product_class_id)
                return

        # Multiple cameras and unstable product tracks can complete the same
        # physical transition. Once one event wins, stale sibling hypotheses
        # must not emit the transition again later.
        self.state_machine.reset_product_class(event.product_class_id)
        if event.event_type == EVENT_PRODUCT_RETURNED:
            self._clear_interaction_context(event.product_class_id)
        elif event.event_type == EVENT_PRODUCT_PICKED:
            self._clear_pick_contact_context(event.product_class_id)
        signature = (
            event.product_class_id,
            event.event_type,
            event.shelf_id,
        )
        previous_time = self._last_event_time_by_signature.get(signature)
        if (
            previous_time is None
            or event.occurred_host_synced_seconds - previous_time
            >= self.duplicate_event_window_seconds
        ):
            self._last_event_time_by_signature[signature] = (
                event.occurred_host_synced_seconds
            )
            events.append(event)
            if event.event_type == EVENT_PRODUCT_PICKED and cycle_key is not None:
                self._active_pick_by_product_shelf[cycle_key] = _ActiveProductPickup(
                    event=event,
                    expires_at_seconds=(
                        event.occurred_host_synced_seconds
                        + self.active_pick_timeout_seconds
                    ),
                )
            elif event.event_type == EVENT_PRODUCT_RETURNED and cycle_key is not None:
                for active_key in tuple(self._active_pick_by_product_shelf):
                    if active_key[0] == event.product_class_id:
                        del self._active_pick_by_product_shelf[active_key]
                self._last_return_time_by_product_shelf[cycle_key] = (
                    event.occurred_host_synced_seconds
                )

    def _expire_active_picks(self, now: float) -> None:
        for key, active_pick in tuple(self._active_pick_by_product_shelf.items()):
            if now > active_pick.expires_at_seconds:
                del self._active_pick_by_product_shelf[key]

    def _clear_pick_contact_context(self, product_class_id: int) -> None:
        self._clear_context_map_for_class(
            self._recent_shelf_contact_by_context,
            product_class_id,
        )
        self._clear_context_map_for_class(
            self._recent_shelf_presence_by_context,
            product_class_id,
        )
        for key, candidate in tuple(self._shelf_contact_candidates.items()):
            if candidate.last_sample.product_class_id == product_class_id:
                del self._shelf_contact_candidates[key]
        for key, candidate in tuple(self._shelf_presence_candidates.items()):
            if candidate.last_sample.product_class_id == product_class_id:
                del self._shelf_presence_candidates[key]

    @staticmethod
    def _clear_context_map_for_class(
        context_map: dict[tuple[int, int], Any],
        product_class_id: int,
    ) -> None:
        for key in tuple(context_map):
            if key[1] == product_class_id:
                del context_map[key]

    def _clear_interaction_context(self, product_class_id: int) -> None:
        self._clear_context_map_for_class(
            self._recent_carried_by_context,
            product_class_id,
        )
        self._clear_context_map_for_class(
            self._recent_shelf_contact_by_context,
            product_class_id,
        )
        self._clear_context_map_for_class(
            self._recent_placement_by_context,
            product_class_id,
        )
        self._recent_carried_by_product_class.pop(product_class_id, None)
        for key in tuple(self._recent_placement_by_product_shelf):
            if key[0] == product_class_id:
                del self._recent_placement_by_product_shelf[key]

    def current_payloads(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "cameraIndex": sample.camera_index,
                "cameraNumber": sample.camera_index + 1,
                "deviceId": sample.device_id,
                "personTrackId": sample.person_track_id,
                "productTrackId": sample.product_track_id,
                "visitId": sample.visit_id,
                "customerId": sample.customer_id,
                "productClassId": sample.product_class_id,
                "productLabel": sample.product_label,
                "productScore": sample.product_score,
                "productBox": {
                    "x1": sample.product_box[0],
                    "y1": sample.product_box[1],
                    "x2": sample.product_box[2],
                    "y2": sample.product_box[3],
                },
                "shelfId": sample.shelf_id,
                "shelfOccupancy": sample.shelf_depth_reference.occupancy,
                "expectedShelfDepthMm": sample.shelf_depth_reference.expected_depth_mm,
                "shelfDepthResidualMm": sample.shelf_depth_reference.depth_residual_mm,
                "shelfDepthToleranceMm": sample.shelf_depth_reference.on_shelf_tolerance_mm,
                "shelfDepthGridCell": (
                    None
                    if sample.shelf_depth_reference.grid_column is None
                    else {
                        "column": sample.shelf_depth_reference.grid_column,
                        "row": sample.shelf_depth_reference.grid_row,
                    }
                ),
                "depthSuppressionReason": sample.shelf_depth_reference.suppression_reason,
                "depthSequenceNumber": (
                    None
                    if sample.product_depth is None
                    else sample.product_depth.depth_sequence_number
                ),
                "depthDeltaMilliseconds": (
                    None
                    if sample.product_depth is None
                    else sample.product_depth.timestamp_delta_milliseconds
                ),
                "productDepthMm": (
                    None
                    if sample.product_depth is None
                    else sample.product_depth.median_depth_mm
                ),
                "productDepthMadMm": (
                    None
                    if sample.product_depth is None
                    else sample.product_depth.mad_depth_mm
                ),
                "productDepthValidFraction": (
                    None
                    if sample.product_depth is None
                    else sample.product_depth.valid_fraction
                ),
                "hand": sample.hand,
                "handScore": sample.hand_score,
                "rgbSequenceNumber": sample.rgb_sequence_number,
                "poseSequenceNumber": sample.pose_sequence_number,
                "poseDeltaMilliseconds": sample.pose_delta_milliseconds,
                "hostSyncedSeconds": sample.host_synced_seconds,
            }
            for sample in sorted(
                self._current_samples.values(),
                key=lambda item: (
                    item.camera_index,
                    item.person_track_id,
                    item.product_track_id,
                ),
            )
        )
