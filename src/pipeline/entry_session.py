from __future__ import annotations

import json
import re
import argparse
import html
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from contracts import (
    SCHEMA_VERSION,
    EmbeddingRef,
    EntryEvent,
    EntryEventQuality,
    EntrySessionPacket,
    EvidenceImageRef,
    new_entry_event,
)
from pipeline.config import (
    DEFAULT_AMBIGUITY_FACE_SIMILARITY,
    DEFAULT_CAMERA_ID,
    DEFAULT_ENTRY_SESSION_LINE_AXIS,
    DEFAULT_ENTRY_SESSION_LINE_POSITION,
    DEFAULT_ENTRY_SESSION_RUNS_DIR,
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_EMBEDDING_RUNS_DIR,
    DEFAULT_MERGE_WINDOW_SECONDS,
    DEFAULT_MIN_FACE_SIMILARITY,
    DEFAULT_MIN_SAME_CAMERA_SIMILARITY,
    DEFAULT_SHOP_ID,
)


EVENT_NAME_RE = re.compile(r"track_(\d+)_event_(\d+)_(\d{8})_(\d{6})$")


@dataclass
class SessionCandidate:
    packet_id: str
    events: List[EntryEvent]
    merge_reasons: List[Dict[str, object]]


@dataclass
class MergeDecision:
    incoming_event_id: str
    candidate_session_id: str | None
    candidate_event_id: str | None
    time_delta_seconds: float | None
    face_similarity: float | None
    same_camera: bool
    decision: str
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "incoming_event_id": self.incoming_event_id,
            "candidate_session_id": self.candidate_session_id,
            "candidate_event_id": self.candidate_event_id,
            "time_delta_seconds": self.time_delta_seconds,
            "face_similarity": self.face_similarity,
            "same_camera": self.same_camera,
            "decision": self.decision,
            "reason": self.reason,
        }


def build_entry_session_argparser(
    description: str = "Build typed EntryEvent and EntrySessionPacket artifacts from saved evidence.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--shop-id",
        type=str,
        default=DEFAULT_SHOP_ID,
        help="Shop identifier to stamp into the generated contracts.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default=DEFAULT_CAMERA_ID,
        help="Default entrance camera identifier for generated entry events.",
    )
    parser.add_argument(
        "--camera-map-json",
        type=Path,
        default=None,
        help="Optional JSON mapping of event_name to camera_id for multi-camera offline experiments.",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help="Directory containing event evidence folders.",
    )
    parser.add_argument(
        "--embedding-runs-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_RUNS_DIR,
        help="Directory containing embedding output folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ENTRY_SESSION_RUNS_DIR,
        help="Root directory where EntryEvent and EntrySessionPacket outputs will be written.",
    )
    parser.add_argument(
        "--merge-window-seconds",
        type=float,
        default=DEFAULT_MERGE_WINDOW_SECONDS,
        help="Time window for merging nearby entrance events into one entry session.",
    )
    parser.add_argument(
        "--line-axis",
        type=str,
        choices=["x", "y"],
        default=DEFAULT_ENTRY_SESSION_LINE_AXIS,
        help="Line axis to stamp into generated entry events.",
    )
    parser.add_argument(
        "--line-position",
        type=float,
        default=DEFAULT_ENTRY_SESSION_LINE_POSITION,
        help="Normalized entrance line position to stamp into generated entry events.",
    )
    parser.add_argument(
        "--min-face-similarity",
        type=float,
        default=DEFAULT_MIN_FACE_SIMILARITY,
        help="Minimum face-embedding cosine similarity for merging different-camera events.",
    )
    parser.add_argument(
        "--ambiguity-face-similarity",
        type=float,
        default=DEFAULT_AMBIGUITY_FACE_SIMILARITY,
        help="Below merge threshold but above this value is marked ambiguous instead of decisively separate.",
    )
    parser.add_argument(
        "--min-same-camera-similarity",
        type=float,
        default=DEFAULT_MIN_SAME_CAMERA_SIMILARITY,
        help="Higher similarity required before same-camera events are merged into one session.",
    )
    return parser


def build_entry_session_html_argparser(
    description: str = "Generate an HTML reviewer for EntryEvent and EntrySessionPacket artifacts.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ENTRY_SESSION_RUNS_DIR,
        help="Directory containing entry_events, entry_sessions, and review outputs.",
    )
    return parser


def parse_event_timestamp(event_name: str) -> datetime:
    match = EVENT_NAME_RE.match(event_name)
    if match is None:
        raise ValueError(f"Unrecognized event folder name: {event_name}")
    stamp = f"{match.group(3)}_{match.group(4)}"
    return datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)


def format_timestamp_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_camera_map(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def list_embedding_event_dirs(embedding_runs_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in embedding_runs_dir.iterdir()
        if path.is_dir() and (path / "summary.json").exists() and (path / "mean_embedding.npy").exists()
    )


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32, copy=True)
    return (vector / norm).astype(np.float32, copy=False)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_n = l2_normalize(left)
    right_n = l2_normalize(right)
    return float(np.dot(left_n, right_n))


def mean_cosine_to_centroid(vectors: np.ndarray) -> float | None:
    if vectors.size == 0:
        return None
    centroid = l2_normalize(np.mean(vectors, axis=0))
    normalized = np.stack([l2_normalize(vector) for vector in vectors], axis=0)
    scores = normalized @ centroid
    return float(np.mean(scores))


def compute_quality(summary: Dict[str, object], event_dir: Path) -> EntryEventQuality:
    images = list(summary.get("images", []))
    det_scores = [float(item["det_score"]) for item in images if item.get("det_score") is not None]
    best_face_det_score = max(det_scores) if det_scores else None

    embeddings_path = event_dir / "embeddings.npy"
    embedding_consistency = None
    if embeddings_path.exists():
        vectors = np.load(embeddings_path).astype(np.float32)
        embedding_consistency = mean_cosine_to_centroid(vectors)

    face_count = int(summary.get("face_count", 0))
    face_component = min(face_count / 6.0, 1.0)
    det_component = 0.0 if best_face_det_score is None else best_face_det_score
    consistency_component = 0.0 if embedding_consistency is None else embedding_consistency
    quality_score = (0.35 * face_component) + (0.30 * det_component) + (0.35 * consistency_component)

    notes: List[str] = []
    if face_count < 3:
        notes.append("low_face_count")
    if best_face_det_score is not None and best_face_det_score < 0.70:
        notes.append("low_best_face_score")
    if embedding_consistency is not None and embedding_consistency < 0.60:
        notes.append("low_embedding_consistency")

    return EntryEventQuality(
        face_count=face_count,
        best_face_det_score=best_face_det_score,
        embedding_consistency=embedding_consistency,
        quality_score=float(quality_score),
        notes=notes,
    )


def evidence_refs_from_summary(summary: Dict[str, object], evidence_dir: Path) -> List[EvidenceImageRef]:
    refs: List[EvidenceImageRef] = []
    for item in summary.get("images", []):
        image_name = item["image_name"]
        if image_name.startswith("pre_"):
            kind = "pre"
        elif image_name.startswith("post_"):
            kind = "post"
        elif image_name.startswith("event_frame_"):
            kind = "event"
        else:
            kind = "best"
        refs.append(
            EvidenceImageRef(
                path=str((evidence_dir / image_name).resolve()),
                kind=kind,
                score=float(item["det_score"]) if item.get("det_score") is not None else None,
            )
        )

    best_image = summary.get("best_image")
    if best_image:
        refs.append(
            EvidenceImageRef(
                path=str((evidence_dir / best_image).resolve()),
                kind="best",
                score=None,
            )
        )
    return refs


def build_entry_event(
    *,
    event_name: str,
    summary: Dict[str, object],
    evidence_dir: Path,
    embedding_dir: Path,
    shop_id: str,
    camera_id: str,
    line_axis: str,
    line_position: float,
) -> EntryEvent:
    timestamp = parse_event_timestamp(event_name)
    track_id_match = EVENT_NAME_RE.match(event_name)
    assert track_id_match is not None
    track_id = int(track_id_match.group(1))

    face_embedding = EmbeddingRef(
        model_name="arcface",
        vector_path=str((embedding_dir / "mean_embedding.npy").resolve()),
        dimension=int(summary.get("embedding_dim", 0)),
        source_image_path=str((evidence_dir / summary["best_image"]).resolve())
        if summary.get("best_image")
        else None,
        mean_vector_path=str((embedding_dir / "mean_embedding.npy").resolve()),
    )

    quality = compute_quality(summary, embedding_dir)
    metadata = {
        "embedding_run_dir": str(embedding_dir.resolve()),
        "evidence_dir": str(evidence_dir.resolve()),
        "best_image": summary.get("best_image"),
    }

    return new_entry_event(
        entry_event_id=f"entry_evt_{event_name}",
        shop_id=shop_id,
        camera_id=camera_id,
        timestamp_utc=format_timestamp_utc(timestamp),
        track_id=track_id,
        line_axis=line_axis,  # type: ignore[arg-type]
        line_position=line_position,
        evidence_images=evidence_refs_from_summary(summary, evidence_dir),
        face_embedding=face_embedding,
        body_embedding=None,
        quality=quality,
        raw_evidence_dir=str(evidence_dir.resolve()),
        metadata=metadata,
    )


def save_entry_event(event: EntryEvent, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{event.entry_event_id}.json"
    path.write_text(json.dumps(event.to_dict(), indent=2), encoding="utf-8")
    return path


def load_embedding_vector(embedding: EmbeddingRef | None) -> np.ndarray | None:
    if embedding is None:
        return None
    return np.load(embedding.vector_path).astype(np.float32)


def parse_timestamp_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def merge_events_to_packet(
    session_index: int,
    events: Sequence[EntryEvent],
    merge_reasons: Sequence[Dict[str, object]],
    output_dir: Path,
) -> EntrySessionPacket:
    sorted_events = sorted(events, key=lambda event: event.timestamp_utc)
    primary = max(sorted_events, key=lambda event: event.quality.quality_score)
    session_id = f"entry_session_{session_index:03d}"
    session_dir = output_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    vectors: List[np.ndarray] = []
    for event in sorted_events:
        vector = load_embedding_vector(event.face_embedding)
        if vector is not None:
            vectors.append(vector)

    merged_face_embedding = None
    if vectors:
        merged_vector = l2_normalize(np.mean(np.stack(vectors, axis=0), axis=0))
        merged_vector_path = session_dir / "merged_face_embedding.npy"
        np.save(merged_vector_path, merged_vector)
        merged_face_embedding = EmbeddingRef(
            model_name="arcface",
            vector_path=str(merged_vector_path.resolve()),
            dimension=int(merged_vector.shape[0]),
            source_image_path=primary.face_embedding.source_image_path if primary.face_embedding else None,
            mean_vector_path=str(merged_vector_path.resolve()),
        )

    representative_images: List[EvidenceImageRef] = []
    seen_paths = set()
    for event in sorted_events:
        best_refs = [ref for ref in event.evidence_images if ref.kind == "best"]
        chosen = best_refs[0] if best_refs else (event.evidence_images[0] if event.evidence_images else None)
        if chosen is not None and chosen.path not in seen_paths:
            representative_images.append(chosen)
            seen_paths.add(chosen.path)

    return EntrySessionPacket(
        schema_version=SCHEMA_VERSION,
        entry_session_id=session_id,
        shop_id=primary.shop_id,
        started_at_utc=sorted_events[0].timestamp_utc,
        ended_at_utc=sorted_events[-1].timestamp_utc,
        contributing_event_ids=[event.entry_event_id for event in sorted_events],
        contributing_camera_ids=sorted({event.camera_id for event in sorted_events}),
        primary_entry_event_id=primary.entry_event_id,
        representative_images=representative_images,
        merged_face_embedding=merged_face_embedding,
        merged_body_embedding=None,
        aggregate_quality_score=float(
            np.mean([event.quality.quality_score for event in sorted_events])
        ),
        local_person_id=None,
        shopping_customer_id=None,
        association_state="unassigned",
        metadata={
            "event_count": len(sorted_events),
            "merge_strategy": "time_window_plus_similarity",
            "quality_scores": [event.quality.quality_score for event in sorted_events],
            "merge_reasons": list(merge_reasons),
        },
    )


def save_entry_session(packet: EntrySessionPacket, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{packet.entry_session_id}.json"
    path.write_text(json.dumps(packet.to_dict(), indent=2), encoding="utf-8")
    return path


def run_entry_session_pipeline(
    *,
    shop_id: str,
    camera_id: str,
    camera_map_json: Path | None,
    evidence_dir: Path,
    embedding_runs_dir: Path,
    output_root: Path,
    merge_window_seconds: float,
    line_axis: str,
    line_position: float,
    min_face_similarity: float,
    ambiguity_face_similarity: float,
    min_same_camera_similarity: float,
) -> Dict[str, object]:
    camera_map = load_camera_map(camera_map_json)
    event_output_dir = output_root / "entry_events"
    session_output_dir = output_root / "entry_sessions"
    review_output_dir = output_root / "review"
    event_output_dir.mkdir(parents=True, exist_ok=True)
    session_output_dir.mkdir(parents=True, exist_ok=True)
    review_output_dir.mkdir(parents=True, exist_ok=True)

    embedding_dirs = list_embedding_event_dirs(embedding_runs_dir)
    built_events: List[EntryEvent] = []
    event_json_paths: List[str] = []

    for embedding_dir in embedding_dirs:
        event_name = embedding_dir.name
        event_evidence_dir = evidence_dir / event_name
        summary = json.loads((embedding_dir / "summary.json").read_text(encoding="utf-8"))
        mapped_camera_id = camera_map.get(event_name, camera_id)

        event = build_entry_event(
            event_name=event_name,
            summary=summary,
            evidence_dir=event_evidence_dir,
            embedding_dir=embedding_dir,
            shop_id=shop_id,
            camera_id=mapped_camera_id,
            line_axis=line_axis,
            line_position=line_position,
        )
        built_events.append(event)
        event_json_paths.append(str(save_entry_event(event, event_output_dir).resolve()))
        print(
            f"ENTRY_EVENT_JSON event={event.entry_event_id} camera={event.camera_id} "
            f"quality={event.quality.quality_score:.3f}"
        )

    sessions, decisions = group_events_with_correlation(
        built_events,
        merge_window_seconds=merge_window_seconds,
        min_face_similarity=min_face_similarity,
        ambiguity_face_similarity=ambiguity_face_similarity,
        min_same_camera_similarity=min_same_camera_similarity,
    )
    session_json_paths: List[str] = []
    for index, session in enumerate(sessions, start=1):
        packet = merge_events_to_packet(index, session.events, session.merge_reasons, session_output_dir)
        session_json_paths.append(str(save_entry_session(packet, session_output_dir).resolve()))
        print(
            f"ENTRY_SESSION_JSON session={packet.entry_session_id} "
            f"events={len(packet.contributing_event_ids)} cameras={len(packet.contributing_camera_ids)}"
        )

    decisions_path = review_output_dir / "merge_decisions.json"
    decisions_path.write_text(
        json.dumps([decision.to_dict() for decision in decisions], indent=2),
        encoding="utf-8",
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "shop_id": shop_id,
        "camera_id_default": camera_id,
        "merge_window_seconds": merge_window_seconds,
        "min_face_similarity": min_face_similarity,
        "ambiguity_face_similarity": ambiguity_face_similarity,
        "min_same_camera_similarity": min_same_camera_similarity,
        "entry_event_count": len(built_events),
        "entry_session_count": len(sessions),
        "entry_event_files": event_json_paths,
        "entry_session_files": session_json_paths,
        "merge_decisions_file": str(decisions_path.resolve()),
    }
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote run summary to {summary_path}")
    return summary


def classify_candidate(
    incoming_event: EntryEvent,
    candidate_session: SessionCandidate,
    merge_window_seconds: float,
    min_face_similarity: float,
    ambiguity_face_similarity: float,
    min_same_camera_similarity: float,
) -> MergeDecision:
    candidate_event = max(
        candidate_session.events,
        key=lambda event: (
            parse_timestamp_utc(event.timestamp_utc),
            event.quality.quality_score,
        ),
    )
    incoming_time = parse_timestamp_utc(incoming_event.timestamp_utc)
    candidate_time = parse_timestamp_utc(candidate_event.timestamp_utc)
    delta = abs((incoming_time - candidate_time).total_seconds())
    same_camera = incoming_event.camera_id == candidate_event.camera_id

    if delta > merge_window_seconds:
        return MergeDecision(
            incoming_event_id=incoming_event.entry_event_id,
            candidate_session_id=candidate_session.packet_id,
            candidate_event_id=candidate_event.entry_event_id,
            time_delta_seconds=delta,
            face_similarity=None,
            same_camera=same_camera,
            decision="kept_separate",
            reason="outside_merge_window",
        )

    incoming_vector = load_embedding_vector(incoming_event.face_embedding)
    candidate_vector = load_embedding_vector(candidate_event.face_embedding)
    similarity = None
    if incoming_vector is not None and candidate_vector is not None:
        similarity = cosine_similarity(incoming_vector, candidate_vector)

    if same_camera:
        if similarity is not None and similarity >= min_same_camera_similarity:
            return MergeDecision(
                incoming_event_id=incoming_event.entry_event_id,
                candidate_session_id=candidate_session.packet_id,
                candidate_event_id=candidate_event.entry_event_id,
                time_delta_seconds=delta,
                face_similarity=similarity,
                same_camera=True,
                decision="merged_same_entry",
                reason="same_camera_high_similarity",
            )
        return MergeDecision(
            incoming_event_id=incoming_event.entry_event_id,
            candidate_session_id=candidate_session.packet_id,
            candidate_event_id=candidate_event.entry_event_id,
            time_delta_seconds=delta,
            face_similarity=similarity,
            same_camera=True,
            decision="kept_separate",
            reason="same_camera_without_high_similarity",
        )

    if similarity is not None and similarity >= min_face_similarity:
        return MergeDecision(
            incoming_event_id=incoming_event.entry_event_id,
            candidate_session_id=candidate_session.packet_id,
            candidate_event_id=candidate_event.entry_event_id,
            time_delta_seconds=delta,
            face_similarity=similarity,
            same_camera=False,
            decision="merged_same_entry",
            reason="cross_camera_similarity_above_threshold",
        )

    if similarity is not None and similarity >= ambiguity_face_similarity:
        return MergeDecision(
            incoming_event_id=incoming_event.entry_event_id,
            candidate_session_id=candidate_session.packet_id,
            candidate_event_id=candidate_event.entry_event_id,
            time_delta_seconds=delta,
            face_similarity=similarity,
            same_camera=False,
            decision="ambiguous",
            reason="cross_camera_similarity_in_ambiguity_band",
        )

    return MergeDecision(
        incoming_event_id=incoming_event.entry_event_id,
        candidate_session_id=candidate_session.packet_id,
        candidate_event_id=candidate_event.entry_event_id,
        time_delta_seconds=delta,
        face_similarity=similarity,
        same_camera=False,
        decision="kept_separate",
        reason="cross_camera_similarity_below_threshold",
    )


def group_events_with_correlation(
    events: Sequence[EntryEvent],
    merge_window_seconds: float,
    min_face_similarity: float,
    ambiguity_face_similarity: float,
    min_same_camera_similarity: float,
) -> tuple[List[SessionCandidate], List[MergeDecision]]:
    sorted_events = sorted(events, key=lambda event: event.timestamp_utc)
    sessions: List[SessionCandidate] = []
    decisions: List[MergeDecision] = []

    for event in sorted_events:
        best_merge: MergeDecision | None = None
        best_ambiguous: MergeDecision | None = None

        for session in sessions:
            decision = classify_candidate(
                incoming_event=event,
                candidate_session=session,
                merge_window_seconds=merge_window_seconds,
                min_face_similarity=min_face_similarity,
                ambiguity_face_similarity=ambiguity_face_similarity,
                min_same_camera_similarity=min_same_camera_similarity,
            )
            decisions.append(decision)

            if decision.decision == "merged_same_entry":
                if best_merge is None:
                    best_merge = decision
                else:
                    current_score = decision.face_similarity if decision.face_similarity is not None else -1.0
                    best_score = best_merge.face_similarity if best_merge.face_similarity is not None else -1.0
                    if current_score > best_score:
                        best_merge = decision
            elif decision.decision == "ambiguous" and best_ambiguous is None:
                best_ambiguous = decision

        if best_merge is not None and best_merge.candidate_session_id is not None:
            target_session = next(
                session for session in sessions if session.packet_id == best_merge.candidate_session_id
            )
            target_session.events.append(event)
            target_session.merge_reasons.append(best_merge.to_dict())
            continue

        if best_ambiguous is not None:
            decisions.append(
                MergeDecision(
                    incoming_event_id=event.entry_event_id,
                    candidate_session_id=best_ambiguous.candidate_session_id,
                    candidate_event_id=best_ambiguous.candidate_event_id,
                    time_delta_seconds=best_ambiguous.time_delta_seconds,
                    face_similarity=best_ambiguous.face_similarity,
                    same_camera=best_ambiguous.same_camera,
                    decision="created_new_session",
                    reason="ambiguous_candidate_not_auto_merged",
                )
            )

        session_id = f"entry_session_{len(sessions) + 1:03d}"
        sessions.append(SessionCandidate(packet_id=session_id, events=[event], merge_reasons=[]))

    return sessions, decisions


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def image_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def render_image(ref: Dict[str, object]) -> str:
    path = ref.get("path")
    if not isinstance(path, str):
        return '<div class="missing">No image</div>'
    return f'<img src="{image_uri(path)}" alt="{html.escape(Path(path).name)}">'


def load_entry_events(entry_events_dir: Path) -> Dict[str, Dict[str, object]]:
    events: Dict[str, Dict[str, object]] = {}
    for path in sorted(entry_events_dir.glob("*.json")):
        payload = load_json(path)
        events[str(payload["entry_event_id"])] = payload
    return events


def load_entry_sessions(entry_sessions_dir: Path) -> List[Dict[str, object]]:
    return [load_json(path) for path in sorted(entry_sessions_dir.glob("*.json"))]


def load_merge_decisions(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def representative_image(event: Dict[str, object]) -> Dict[str, object] | None:
    images = event.get("evidence_images", [])
    if not isinstance(images, list):
        return None
    best = next((item for item in images if isinstance(item, dict) and item.get("kind") == "best"), None)
    if isinstance(best, dict):
        return best
    first = images[0] if images else None
    return first if isinstance(first, dict) else None


def render_event_card(event: Dict[str, object]) -> str:
    quality = event.get("quality", {})
    if not isinstance(quality, dict):
        quality = {}
    image_ref = representative_image(event)
    image_html = render_image(image_ref) if image_ref is not None else '<div class="missing">No image</div>'
    return f"""
    <div class="event-card">
      <div class="event-image">{image_html}</div>
      <div class="event-meta">
        <div class="event-title">{html.escape(str(event.get('entry_event_id', 'unknown')))}</div>
        <div>camera={html.escape(str(event.get('camera_id', 'n/a')))}</div>
        <div>time={html.escape(str(event.get('timestamp_utc', 'n/a')))}</div>
        <div>track_id={html.escape(str(event.get('track_id', 'n/a')))}</div>
        <div>quality={float(quality.get('quality_score', 0.0)):.3f}</div>
        <div>face_count={html.escape(str(quality.get('face_count', 'n/a')))}</div>
      </div>
    </div>
    """


def render_session_card(
    session: Dict[str, object],
    event_lookup: Dict[str, Dict[str, object]],
) -> str:
    event_ids = [item for item in session.get("contributing_event_ids", []) if isinstance(item, str)]
    events = [event_lookup[event_id] for event_id in event_ids if event_id in event_lookup]
    event_cards = "".join(render_event_card(event) for event in events)
    metadata = session.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    merge_reasons = metadata.get("merge_reasons", [])
    reasons_html = ""
    if isinstance(merge_reasons, list) and merge_reasons:
        reason_lines = []
        for item in merge_reasons:
            if not isinstance(item, dict):
                continue
            similarity = item.get("face_similarity")
            similarity_text = "n/a" if similarity is None else f"{float(similarity):.4f}"
            reason_lines.append(
                f"<li>{html.escape(str(item.get('incoming_event_id')))} -> "
                f"{html.escape(str(item.get('candidate_event_id')))} "
                f"decision={html.escape(str(item.get('decision')))} "
                f"reason={html.escape(str(item.get('reason')))} "
                f"similarity={similarity_text}</li>"
            )
        reasons_html = f"<ul class='reasons'>{''.join(reason_lines)}</ul>"

    return f"""
    <section class="session-card">
      <h2>{html.escape(str(session.get('entry_session_id', 'unknown')))}</h2>
      <div class="session-meta">
        <div>events={len(events)}</div>
        <div>cameras={html.escape(', '.join(str(item) for item in session.get('contributing_camera_ids', [])))}</div>
        <div>started={html.escape(str(session.get('started_at_utc', 'n/a')))}</div>
        <div>ended={html.escape(str(session.get('ended_at_utc', 'n/a')))}</div>
        <div>aggregate_quality={float(session.get('aggregate_quality_score', 0.0)):.3f}</div>
      </div>
      {reasons_html}
      <div class="event-grid">
        {event_cards}
      </div>
    </section>
    """


def render_decisions(decisions: List[Dict[str, object]]) -> str:
    interesting = [
        item
        for item in decisions
        if item.get("decision") in {"ambiguous", "created_new_session", "kept_separate"}
    ]
    if not interesting:
        return ""

    rows = []
    for item in interesting:
        similarity = item.get("face_similarity")
        similarity_text = "n/a" if similarity is None else f"{float(similarity):.4f}"
        delta = item.get("time_delta_seconds")
        delta_text = "n/a" if delta is None else f"{float(delta):.2f}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('incoming_event_id')))}</td>"
            f"<td>{html.escape(str(item.get('candidate_event_id')))}</td>"
            f"<td>{html.escape(str(item.get('decision')))}</td>"
            f"<td>{html.escape(str(item.get('reason')))}</td>"
            f"<td>{delta_text}</td>"
            f"<td>{similarity_text}</td>"
            "</tr>"
        )

    return f"""
    <section class="decision-card">
      <h2>Non-Merge / Ambiguity Decisions</h2>
      <table>
        <thead>
          <tr>
            <th>Incoming Event</th>
            <th>Candidate Event</th>
            <th>Decision</th>
            <th>Reason</th>
            <th>Delta s</th>
            <th>Similarity</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>
    """


def render_entry_session_html(
    summary: Dict[str, object],
    sessions: List[Dict[str, object]],
    event_lookup: Dict[str, Dict[str, object]],
    decisions: List[Dict[str, object]],
) -> str:
    session_cards = "".join(render_session_card(session, event_lookup) for session in sessions)
    decision_section = render_decisions(decisions)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Entry Session Review</title>
  <style>
    body {{
      font-family: Segoe UI, Arial, sans-serif;
      margin: 24px;
      background: #f4f1ea;
      color: #1f1f1f;
    }}
    .summary {{
      margin-bottom: 20px;
      padding: 16px;
      border: 1px solid #d8cfbf;
      border-radius: 12px;
      background: #fffdf8;
    }}
    .session-card, .decision-card {{
      margin-bottom: 20px;
      padding: 16px;
      border: 1px solid #d8cfbf;
      border-radius: 12px;
      background: #fffdf8;
    }}
    .session-meta {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 12px;
      color: #444;
    }}
    .event-grid {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
    }}
    .event-card {{
      width: 250px;
      border: 1px solid #e3dbc9;
      border-radius: 10px;
      background: #faf7ef;
      padding: 12px;
    }}
    .event-image img {{
      width: 226px;
      max-width: 100%;
      border: 1px solid #d7cfbf;
      border-radius: 8px;
      margin-bottom: 10px;
    }}
    .missing {{
      width: 226px;
      height: 226px;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px dashed #b9af9b;
      border-radius: 8px;
      margin-bottom: 10px;
      background: #ece6d9;
      color: #555;
    }}
    .event-title {{
      font-weight: 700;
      margin-bottom: 8px;
      word-break: break-word;
    }}
    .reasons {{
      margin-top: 0;
      color: #444;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #ddd4c5;
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f0eadc;
    }}
  </style>
</head>
<body>
  <div class="summary">
    <h1>Entry Session Review</h1>
    <div>entry_event_count={html.escape(str(summary.get('entry_event_count', 'n/a')))}</div>
    <div>entry_session_count={html.escape(str(summary.get('entry_session_count', 'n/a')))}</div>
    <div>merge_window_seconds={html.escape(str(summary.get('merge_window_seconds', 'n/a')))}</div>
    <div>min_face_similarity={html.escape(str(summary.get('min_face_similarity', 'n/a')))}</div>
    <div>ambiguity_face_similarity={html.escape(str(summary.get('ambiguity_face_similarity', 'n/a')))}</div>
    <div>min_same_camera_similarity={html.escape(str(summary.get('min_same_camera_similarity', 'n/a')))}</div>
  </div>
  {decision_section}
  {session_cards}
</body>
</html>
"""


def write_entry_session_html(output_root: Path) -> Path:
    summary = load_json(output_root / "run_summary.json")
    event_lookup = load_entry_events(output_root / "entry_events")
    sessions = load_entry_sessions(output_root / "entry_sessions")
    decisions = load_merge_decisions(output_root / "review" / "merge_decisions.json")
    html_text = render_entry_session_html(summary, sessions, event_lookup, decisions)
    output_path = output_root / "review" / "entry_session_review.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote entry session review HTML to {output_path}")
    return output_path
