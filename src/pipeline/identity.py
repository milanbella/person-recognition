from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from pipeline.config import (
    DEFAULT_EMBEDDING_RUNS_DIR,
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_IDENTITY_RUNS_DIR,
)
from pipeline.review import load_best_image_from_summary


DEFAULT_MATCH_THRESHOLD = 0.68
DEFAULT_MIN_FACE_COUNT = 3
EVENT_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")


@dataclass
class EventEmbedding:
    event_name: str
    event_dir: Path
    embedding: np.ndarray
    face_count: int
    best_image: str | None
    timestamp_key: str


@dataclass
class IdentityState:
    identity_id: str
    prototype: np.ndarray
    event_names: List[str]
    face_counts: List[int]


@dataclass
class Assignment:
    event_name: str
    face_count: int
    best_image: str | None
    decision: str
    assigned_identity: str | None
    best_score: float | None


def build_identity_match_argparser(
    description: str = "Step 9: assign local identities from event mean embeddings.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--embedding-runs-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_RUNS_DIR,
        help="Directory containing event embedding outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_IDENTITY_RUNS_DIR,
        help="Directory where identity-matching outputs will be written.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help="Minimum cosine similarity required to match an existing local identity.",
    )
    parser.add_argument(
        "--min-face-count",
        type=int,
        default=DEFAULT_MIN_FACE_COUNT,
        help="Minimum face_count required before an event is allowed to create or update identities.",
    )
    return parser


def build_identity_html_argparser(
    description: str = "Generate an HTML reviewer for local identity assignments.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--identity-runs-dir",
        type=Path,
        default=DEFAULT_IDENTITY_RUNS_DIR,
        help="Directory containing assignments.json and gallery.json.",
    )
    parser.add_argument(
        "--embedding-runs-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_RUNS_DIR,
        help="Directory containing per-event embedding summaries.",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help="Directory containing saved evidence event folders.",
    )
    return parser


def extract_timestamp_key(event_name: str) -> str:
    match = EVENT_TIMESTAMP_RE.search(event_name)
    if match:
        return match.group(1)
    return event_name


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32, copy=True)
    return (vector / norm).astype(np.float32, copy=False)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_n = l2_normalize(left)
    right_n = l2_normalize(right)
    return float(np.dot(left_n, right_n))


def load_event_embeddings(embedding_runs_dir: Path) -> List[EventEmbedding]:
    if not embedding_runs_dir.exists():
        raise FileNotFoundError(f"Embedding runs directory not found: {embedding_runs_dir}")

    events: List[EventEmbedding] = []
    for event_dir in sorted(path for path in embedding_runs_dir.iterdir() if path.is_dir()):
        embedding_path = event_dir / "mean_embedding.npy"
        summary_path = event_dir / "summary.json"
        if not embedding_path.exists() or not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        events.append(
            EventEmbedding(
                event_name=event_dir.name,
                event_dir=event_dir,
                embedding=np.load(embedding_path).astype(np.float32),
                face_count=int(summary.get("face_count", 0)),
                best_image=summary.get("best_image"),
                timestamp_key=extract_timestamp_key(event_dir.name),
            )
        )

    if not events:
        raise RuntimeError(f"No event embeddings found under {embedding_runs_dir}")

    return sorted(events, key=lambda event: event.timestamp_key)


def best_identity_match(
    event: EventEmbedding,
    identities: List[IdentityState],
) -> tuple[IdentityState | None, float]:
    best_identity: IdentityState | None = None
    best_score = -1.0
    for identity in identities:
        score = cosine_similarity(event.embedding, identity.prototype)
        if score > best_score:
            best_score = score
            best_identity = identity
    return best_identity, best_score


def create_identity(identity_index: int, event: EventEmbedding) -> IdentityState:
    return IdentityState(
        identity_id=f"person_{identity_index:03d}",
        prototype=l2_normalize(event.embedding),
        event_names=[event.event_name],
        face_counts=[event.face_count],
    )


def update_identity(identity: IdentityState, event: EventEmbedding) -> None:
    prior_count = len(identity.event_names)
    prior_prototype = identity.prototype.copy()
    identity.event_names.append(event.event_name)
    identity.face_counts.append(event.face_count)
    identity.prototype = l2_normalize(
        ((prior_prototype * prior_count) + event.embedding) / (prior_count + 1)
    )


def run_matcher(
    events: List[EventEmbedding],
    match_threshold: float,
    min_face_count: int,
) -> tuple[List[Dict[str, Any]], List[IdentityState]]:
    identities: List[IdentityState] = []
    assignments: List[Dict[str, Any]] = []

    for event in events:
        matched_identity, best_score = best_identity_match(event, identities)

        if event.face_count < min_face_count:
            assignments.append(
                {
                    "event_name": event.event_name,
                    "face_count": event.face_count,
                    "best_image": event.best_image,
                    "decision": "skip_low_face_count",
                    "assigned_identity": None,
                    "best_score": None if matched_identity is None else best_score,
                }
            )
            continue

        if matched_identity is not None and best_score >= match_threshold:
            update_identity(matched_identity, event)
            assignments.append(
                {
                    "event_name": event.event_name,
                    "face_count": event.face_count,
                    "best_image": event.best_image,
                    "decision": "matched_existing",
                    "assigned_identity": matched_identity.identity_id,
                    "best_score": best_score,
                }
            )
            continue

        new_identity = create_identity(len(identities) + 1, event)
        identities.append(new_identity)
        assignments.append(
            {
                "event_name": event.event_name,
                "face_count": event.face_count,
                "best_image": event.best_image,
                "decision": "created_new_identity",
                "assigned_identity": new_identity.identity_id,
                "best_score": None if matched_identity is None else best_score,
            }
        )

    return assignments, identities


def write_identity_outputs(
    output_dir: Path,
    assignments: List[Dict[str, Any]],
    identities: List[IdentityState],
    *,
    embedding_runs_dir: Path,
    match_threshold: float,
    min_face_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = output_dir / "assignments.json"
    gallery_path = output_dir / "gallery.json"

    assignments_path.write_text(json.dumps(assignments, indent=2), encoding="utf-8")
    gallery_path.write_text(
        json.dumps(
            [
                {
                    "identity_id": identity.identity_id,
                    "event_names": identity.event_names,
                    "face_counts": identity.face_counts,
                    "event_count": len(identity.event_names),
                }
                for identity in identities
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    run_config = {
        "embedding_runs_dir": str(embedding_runs_dir),
        "match_threshold": match_threshold,
        "min_face_count": min_face_count,
        "assignments_file": str(assignments_path),
        "gallery_file": str(gallery_path),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


def print_identity_summary(assignments: List[Dict[str, Any]], identities: List[IdentityState]) -> None:
    print(f"Created {len(identities)} local identities.")
    for identity in identities:
        print(
            f"IDENTITY {identity.identity_id} events={len(identity.event_names)} "
            f"members={', '.join(identity.event_names)}"
        )

    for item in assignments:
        print(
            f"EVENT {item['event_name']} decision={item['decision']} "
            f"assigned={item['assigned_identity']} score={item['best_score']}"
        )


def run_identity_pipeline(
    *,
    embedding_runs_dir: Path,
    output_dir: Path,
    match_threshold: float,
    min_face_count: int,
) -> tuple[List[Dict[str, Any]], List[IdentityState]]:
    events = load_event_embeddings(embedding_runs_dir)
    assignments, identities = run_matcher(
        events=events,
        match_threshold=match_threshold,
        min_face_count=min_face_count,
    )
    write_identity_outputs(
        output_dir,
        assignments,
        identities,
        embedding_runs_dir=embedding_runs_dir,
        match_threshold=match_threshold,
        min_face_count=min_face_count,
    )
    print_identity_summary(assignments, identities)
    return assignments, identities


def load_assignments(identity_runs_dir: Path) -> List[Assignment]:
    assignments_path = identity_runs_dir / "assignments.json"
    if not assignments_path.exists():
        raise FileNotFoundError(
            f"assignments.json not found. Run 04_replay_identity.py first: {assignments_path}"
        )

    raw_items = json.loads(assignments_path.read_text(encoding="utf-8"))
    return [
        Assignment(
            event_name=item["event_name"],
            face_count=int(item.get("face_count", 0)),
            best_image=item.get("best_image"),
            decision=item["decision"],
            assigned_identity=item.get("assigned_identity"),
            best_score=item.get("best_score"),
        )
        for item in raw_items
    ]


def resolve_best_image_path(
    event_name: str,
    best_image: str | None,
    embedding_runs_dir: Path,
    evidence_dir: Path,
) -> Path | None:
    resolved_image = best_image
    if resolved_image is None:
        resolved_image = load_best_image_from_summary(embedding_runs_dir / event_name / "summary.json")
    if resolved_image is None:
        return None

    image_path = evidence_dir / event_name / resolved_image
    if image_path.exists():
        return image_path
    return None


def image_uri(path: Path) -> str:
    return path.resolve().as_uri()


def render_assignment_card(
    assignment: Assignment,
    embedding_runs_dir: Path,
    evidence_dir: Path,
) -> str:
    image_path = resolve_best_image_path(
        assignment.event_name,
        assignment.best_image,
        embedding_runs_dir,
        evidence_dir,
    )
    image_html = (
        f'<img src="{image_uri(image_path)}" alt="{html.escape(assignment.event_name)}">'
        if image_path is not None
        else '<div class="missing">No image</div>'
    )
    score_text = "n/a" if assignment.best_score is None else f"{assignment.best_score:.4f}"

    return f"""
    <div class="event-card">
      <div class="event-image">{image_html}</div>
      <div class="event-meta">
        <div class="event-name">{html.escape(assignment.event_name)}</div>
        <div>decision={html.escape(assignment.decision)}</div>
        <div>face_count={assignment.face_count}</div>
        <div>best_score={score_text}</div>
        <div>best_image={html.escape(assignment.best_image or "n/a")}</div>
      </div>
    </div>
    """


def render_identity_section(
    identity_id: str,
    assignments: List[Assignment],
    embedding_runs_dir: Path,
    evidence_dir: Path,
) -> str:
    cards = "".join(
        render_assignment_card(assignment, embedding_runs_dir, evidence_dir)
        for assignment in assignments
    )
    return f"""
    <section class="identity-section">
      <h2>{html.escape(identity_id)} <span class="count">({len(assignments)} events)</span></h2>
      <div class="events-grid">
        {cards}
      </div>
    </section>
    """


def render_skipped_section(
    assignments: List[Assignment],
    embedding_runs_dir: Path,
    evidence_dir: Path,
) -> str:
    if not assignments:
        return ""

    cards = "".join(
        render_assignment_card(assignment, embedding_runs_dir, evidence_dir)
        for assignment in assignments
    )
    return f"""
    <section class="identity-section skipped">
      <h2>Skipped / Unassigned <span class="count">({len(assignments)} events)</span></h2>
      <div class="events-grid">
        {cards}
      </div>
    </section>
    """


def render_identity_html(
    grouped: Dict[str, List[Assignment]],
    skipped: List[Assignment],
    embedding_runs_dir: Path,
    evidence_dir: Path,
) -> str:
    sections = "".join(
        render_identity_section(identity_id, grouped[identity_id], embedding_runs_dir, evidence_dir)
        for identity_id in sorted(grouped.keys())
    )
    skipped_section = render_skipped_section(skipped, embedding_runs_dir, evidence_dir)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Local Identity Review</title>
  <style>
    body {{
      font-family: Segoe UI, Arial, sans-serif;
      margin: 24px;
      background: #f4f1ea;
      color: #1e1e1e;
    }}
    h1 {{
      margin-bottom: 8px;
    }}
    .intro {{
      margin-bottom: 24px;
      color: #474747;
      max-width: 900px;
    }}
    .identity-section {{
      background: #fffdf8;
      border: 1px solid #d6cebf;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 22px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }}
    .identity-section.skipped {{
      background: #fbf7f1;
    }}
    h2 {{
      margin-top: 0;
      margin-bottom: 14px;
    }}
    .count {{
      color: #666;
      font-size: 16px;
      font-weight: 400;
    }}
    .events-grid {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .event-card {{
      width: 260px;
      background: #faf7ef;
      border: 1px solid #e0d9ca;
      border-radius: 10px;
      padding: 12px;
    }}
    .event-image img {{
      width: 236px;
      max-width: 100%;
      border-radius: 8px;
      border: 1px solid #d8d0c0;
      margin-bottom: 10px;
    }}
    .missing {{
      width: 236px;
      height: 236px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #ece6d9;
      border: 1px dashed #b9af9b;
      border-radius: 8px;
      margin-bottom: 10px;
      color: #555;
    }}
    .event-name {{
      font-weight: 700;
      margin-bottom: 8px;
      word-break: break-word;
    }}
    .event-meta {{
      font-size: 14px;
      color: #333;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <h1>Local Identity Review</h1>
  <div class="intro">
    Each section is one local gallery identity from the offline matcher. Review whether the grouped event crops
    actually belong to the same person. Skipped items are shown separately.
  </div>
  {sections}
  {skipped_section}
</body>
</html>
"""


def write_identity_html(
    identity_runs_dir: Path,
    embedding_runs_dir: Path,
    evidence_dir: Path,
) -> Path:
    assignments = load_assignments(identity_runs_dir)

    grouped: Dict[str, List[Assignment]] = {}
    skipped: List[Assignment] = []

    for assignment in assignments:
        if assignment.assigned_identity is None:
            skipped.append(assignment)
            continue
        grouped.setdefault(assignment.assigned_identity, []).append(assignment)

    html_text = render_identity_html(grouped, skipped, embedding_runs_dir, evidence_dir)
    output_path = identity_runs_dir / "identity_review.html"
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote identity review HTML to {output_path}")
    return output_path
