from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


SCHEMA_VERSION = "2026-05-02"

AssociationState = Literal["unassigned", "assigned_shopping_customer_id", "ambiguous"]
ObservationDecision = Literal["confirmed", "ambiguous", "unresolved"]


@dataclass
class EvidenceImageRef:
    path: str
    kind: Literal["pre", "event", "post", "best"]
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceImageRef":
        return cls(
            path=str(data["path"]),
            kind=data["kind"],
            score=data.get("score"),
        )


@dataclass
class EmbeddingRef:
    model_name: str
    vector_path: str
    dimension: int
    source_image_path: Optional[str] = None
    mean_vector_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "vector_path": self.vector_path,
            "dimension": self.dimension,
            "source_image_path": self.source_image_path,
            "mean_vector_path": self.mean_vector_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingRef":
        return cls(
            model_name=str(data["model_name"]),
            vector_path=str(data["vector_path"]),
            dimension=int(data["dimension"]),
            source_image_path=data.get("source_image_path"),
            mean_vector_path=data.get("mean_vector_path"),
        )


@dataclass
class EntryEventQuality:
    face_count: int
    best_face_det_score: Optional[float]
    embedding_consistency: Optional[float]
    quality_score: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face_count": self.face_count,
            "best_face_det_score": self.best_face_det_score,
            "embedding_consistency": self.embedding_consistency,
            "quality_score": self.quality_score,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntryEventQuality":
        return cls(
            face_count=int(data["face_count"]),
            best_face_det_score=data.get("best_face_det_score"),
            embedding_consistency=data.get("embedding_consistency"),
            quality_score=float(data["quality_score"]),
            notes=list(data.get("notes", [])),
        )


@dataclass
class EntryEvent:
    schema_version: str
    entry_event_id: str
    shop_id: str
    camera_id: str
    timestamp_utc: str
    track_id: int
    entry_direction: Literal["outside_to_inside"]
    line_axis: Literal["x", "y"]
    line_position: float
    evidence_images: List[EvidenceImageRef]
    face_embedding: Optional[EmbeddingRef]
    body_embedding: Optional[EmbeddingRef]
    quality: EntryEventQuality
    raw_evidence_dir: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entry_event_id": self.entry_event_id,
            "shop_id": self.shop_id,
            "camera_id": self.camera_id,
            "timestamp_utc": self.timestamp_utc,
            "track_id": self.track_id,
            "entry_direction": self.entry_direction,
            "line_axis": self.line_axis,
            "line_position": self.line_position,
            "evidence_images": [item.to_dict() for item in self.evidence_images],
            "face_embedding": None if self.face_embedding is None else self.face_embedding.to_dict(),
            "body_embedding": None if self.body_embedding is None else self.body_embedding.to_dict(),
            "quality": self.quality.to_dict(),
            "raw_evidence_dir": self.raw_evidence_dir,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntryEvent":
        return cls(
            schema_version=str(data["schema_version"]),
            entry_event_id=str(data["entry_event_id"]),
            shop_id=str(data["shop_id"]),
            camera_id=str(data["camera_id"]),
            timestamp_utc=str(data["timestamp_utc"]),
            track_id=int(data["track_id"]),
            entry_direction=data["entry_direction"],
            line_axis=data["line_axis"],
            line_position=float(data["line_position"]),
            evidence_images=[
                EvidenceImageRef.from_dict(item) for item in data.get("evidence_images", [])
            ],
            face_embedding=None
            if data.get("face_embedding") is None
            else EmbeddingRef.from_dict(data["face_embedding"]),
            body_embedding=None
            if data.get("body_embedding") is None
            else EmbeddingRef.from_dict(data["body_embedding"]),
            quality=EntryEventQuality.from_dict(data["quality"]),
            raw_evidence_dir=data.get("raw_evidence_dir"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EntrySessionPacket:
    schema_version: str
    entry_session_id: str
    shop_id: str
    started_at_utc: str
    ended_at_utc: str
    contributing_event_ids: List[str]
    contributing_camera_ids: List[str]
    primary_entry_event_id: str
    representative_images: List[EvidenceImageRef]
    merged_face_embedding: Optional[EmbeddingRef]
    merged_body_embedding: Optional[EmbeddingRef]
    aggregate_quality_score: float
    local_person_id: Optional[str]
    shopping_customer_id: Optional[str]
    association_state: AssociationState
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entry_session_id": self.entry_session_id,
            "shop_id": self.shop_id,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "contributing_event_ids": list(self.contributing_event_ids),
            "contributing_camera_ids": list(self.contributing_camera_ids),
            "primary_entry_event_id": self.primary_entry_event_id,
            "representative_images": [item.to_dict() for item in self.representative_images],
            "merged_face_embedding": None
            if self.merged_face_embedding is None
            else self.merged_face_embedding.to_dict(),
            "merged_body_embedding": None
            if self.merged_body_embedding is None
            else self.merged_body_embedding.to_dict(),
            "aggregate_quality_score": self.aggregate_quality_score,
            "local_person_id": self.local_person_id,
            "shopping_customer_id": self.shopping_customer_id,
            "association_state": self.association_state,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntrySessionPacket":
        return cls(
            schema_version=str(data["schema_version"]),
            entry_session_id=str(data["entry_session_id"]),
            shop_id=str(data["shop_id"]),
            started_at_utc=str(data["started_at_utc"]),
            ended_at_utc=str(data["ended_at_utc"]),
            contributing_event_ids=list(data.get("contributing_event_ids", [])),
            contributing_camera_ids=list(data.get("contributing_camera_ids", [])),
            primary_entry_event_id=str(data["primary_entry_event_id"]),
            representative_images=[
                EvidenceImageRef.from_dict(item) for item in data.get("representative_images", [])
            ],
            merged_face_embedding=None
            if data.get("merged_face_embedding") is None
            else EmbeddingRef.from_dict(data["merged_face_embedding"]),
            merged_body_embedding=None
            if data.get("merged_body_embedding") is None
            else EmbeddingRef.from_dict(data["merged_body_embedding"]),
            aggregate_quality_score=float(data["aggregate_quality_score"]),
            local_person_id=data.get("local_person_id"),
            shopping_customer_id=data.get("shopping_customer_id"),
            association_state=data["association_state"],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class BackendSessionCandidate:
    schema_version: str
    shop_id: str
    shopping_customer_id: str
    opened_at_utc: str
    expires_at_utc: Optional[str]
    status: Literal["pending_entry", "active", "expired", "cancelled"]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shop_id": self.shop_id,
            "shopping_customer_id": self.shopping_customer_id,
            "opened_at_utc": self.opened_at_utc,
            "expires_at_utc": self.expires_at_utc,
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackendSessionCandidate":
        return cls(
            schema_version=str(data["schema_version"]),
            shop_id=str(data["shop_id"]),
            shopping_customer_id=str(data["shopping_customer_id"]),
            opened_at_utc=str(data["opened_at_utc"]),
            expires_at_utc=data.get("expires_at_utc"),
            status=data["status"],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ObserverObservation:
    schema_version: str
    observer_observation_id: str
    shop_id: str
    camera_id: str
    timestamp_utc: str
    track_id: int
    best_image: Optional[EvidenceImageRef]
    face_embedding: Optional[EmbeddingRef]
    body_embedding: Optional[EmbeddingRef]
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observer_observation_id": self.observer_observation_id,
            "shop_id": self.shop_id,
            "camera_id": self.camera_id,
            "timestamp_utc": self.timestamp_utc,
            "track_id": self.track_id,
            "best_image": None if self.best_image is None else self.best_image.to_dict(),
            "face_embedding": None if self.face_embedding is None else self.face_embedding.to_dict(),
            "body_embedding": None if self.body_embedding is None else self.body_embedding.to_dict(),
            "quality_score": self.quality_score,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObserverObservation":
        return cls(
            schema_version=str(data["schema_version"]),
            observer_observation_id=str(data["observer_observation_id"]),
            shop_id=str(data["shop_id"]),
            camera_id=str(data["camera_id"]),
            timestamp_utc=str(data["timestamp_utc"]),
            track_id=int(data["track_id"]),
            best_image=None
            if data.get("best_image") is None
            else EvidenceImageRef.from_dict(data["best_image"]),
            face_embedding=None
            if data.get("face_embedding") is None
            else EmbeddingRef.from_dict(data["face_embedding"]),
            body_embedding=None
            if data.get("body_embedding") is None
            else EmbeddingRef.from_dict(data["body_embedding"]),
            quality_score=float(data["quality_score"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ObserverAssociationResult:
    schema_version: str
    observer_observation_id: str
    matched_entry_session_id: Optional[str]
    matched_shopping_customer_id: Optional[str]
    decision: ObservationDecision
    best_score: Optional[float]
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observer_observation_id": self.observer_observation_id,
            "matched_entry_session_id": self.matched_entry_session_id,
            "matched_shopping_customer_id": self.matched_shopping_customer_id,
            "decision": self.decision,
            "best_score": self.best_score,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObserverAssociationResult":
        return cls(
            schema_version=str(data["schema_version"]),
            observer_observation_id=str(data["observer_observation_id"]),
            matched_entry_session_id=data.get("matched_entry_session_id"),
            matched_shopping_customer_id=data.get("matched_shopping_customer_id"),
            decision=data["decision"],
            best_score=data.get("best_score"),
            reason=str(data["reason"]),
            metadata=dict(data.get("metadata", {})),
        )


def new_entry_event(
    *,
    entry_event_id: str,
    shop_id: str,
    camera_id: str,
    timestamp_utc: str,
    track_id: int,
    line_axis: Literal["x", "y"],
    line_position: float,
    evidence_images: List[EvidenceImageRef],
    face_embedding: Optional[EmbeddingRef],
    body_embedding: Optional[EmbeddingRef],
    quality: EntryEventQuality,
    raw_evidence_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EntryEvent:
    return EntryEvent(
        schema_version=SCHEMA_VERSION,
        entry_event_id=entry_event_id,
        shop_id=shop_id,
        camera_id=camera_id,
        timestamp_utc=timestamp_utc,
        track_id=track_id,
        entry_direction="outside_to_inside",
        line_axis=line_axis,
        line_position=line_position,
        evidence_images=evidence_images,
        face_embedding=face_embedding,
        body_embedding=body_embedding,
        quality=quality,
        raw_evidence_dir=raw_evidence_dir,
        metadata={} if metadata is None else dict(metadata),
    )
