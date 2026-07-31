from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


PHYSICAL_ANNOTATION_TYPES = {
    "physical_entry",
    "physical_leave",
    "subject_visible_but_not_detected",
    "subject_occluded",
    "subject_reappeared",
    "subject_stationary",
    "shelf_approach",
    "shelf_departure",
    "no_entrance_crossing",
    "note",
    "system_event_correct",
    "system_event_incorrect",
}

OBSERVATION_ANNOTATION_TYPES = {
    "observation_is_subject",
    "observation_is_not_subject",
    "bounding_box_correct",
    "bounding_box_incorrect",
    "customer_identity_correct",
    "customer_identity_incorrect",
    "duplicate_track",
    "different_physical_person",
}

ANNOTATION_TYPES = PHYSICAL_ANNOTATION_TYPES | OBSERVATION_ANNOTATION_TYPES


@dataclass(frozen=True)
class ObservationReference:
    camera_index: int
    device_id: str
    rgb_sequence_number: int
    host_synced_seconds: float
    track_id: int
    observed_visit_id: int | None = None
    observed_customer_id: str | None = None

    def payload(self) -> dict[str, object]:
        return {
            "cameraIndex": self.camera_index,
            "deviceId": self.device_id,
            "rgbSequenceNumber": self.rgb_sequence_number,
            "hostSyncedSeconds": self.host_synced_seconds,
            "trackId": self.track_id,
            "observedVisitId": self.observed_visit_id,
            "observedCustomerId": self.observed_customer_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ObservationReference:
        required = (
            "cameraIndex",
            "deviceId",
            "rgbSequenceNumber",
            "hostSyncedSeconds",
            "trackId",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(
                "Observation reference is missing: " + ", ".join(missing)
            )
        observed_visit_id = payload.get("observedVisitId")
        observed_customer_id = payload.get("observedCustomerId")
        return cls(
            camera_index=int(payload["cameraIndex"]),
            device_id=str(payload["deviceId"]),
            rgb_sequence_number=int(payload["rgbSequenceNumber"]),
            host_synced_seconds=float(payload["hostSyncedSeconds"]),
            track_id=int(payload["trackId"]),
            observed_visit_id=(
                None if observed_visit_id is None else int(observed_visit_id)
            ),
            observed_customer_id=(
                None
                if observed_customer_id is None
                else str(observed_customer_id)
            ),
        )


@dataclass(frozen=True)
class OperatorEvent:
    event_id: int
    event_type: str
    occurred_at_unix_milliseconds: int
    source: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    host_synced_seconds: float | None = None
    camera_index: int | None = None
    device_id: str | None = None
    rgb_sequence_number: int | None = None
    track_id: int | None = None
    visit_id: int | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "eventId": self.event_id,
            "eventType": self.event_type,
            "occurredAtUnixMilliseconds": self.occurred_at_unix_milliseconds,
            "source": self.source,
            "hostSyncedSeconds": self.host_synced_seconds,
            "cameraIndex": self.camera_index,
            "deviceId": self.device_id,
            "rgbSequenceNumber": self.rgb_sequence_number,
            "trackId": self.track_id,
            "visitId": self.visit_id,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class AnalysisResult:
    rule_id: str
    status: str
    summary: str
    expected_annotation_id: int | None = None
    matched_event_id: int | None = None
    latency_milliseconds: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            "ruleId": payload["rule_id"],
            "status": payload["status"],
            "summary": payload["summary"],
            "expectedAnnotationId": payload["expected_annotation_id"],
            "matchedEventId": payload["matched_event_id"],
            "latencyMilliseconds": payload["latency_milliseconds"],
            "details": payload["details"],
        }
