import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


DEFAULT_EMBEDDING_RUNS_DIR = Path(__file__).resolve().parent / "embedding_runs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "identity_runs"
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 9: assign local identities from event mean embeddings."
    )
    parser.add_argument(
        "--embedding-runs-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_RUNS_DIR,
        help="Directory containing event embedding outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where identity-matching outputs will be written.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.68,
        help="Minimum cosine similarity required to match an existing local identity.",
    )
    parser.add_argument(
        "--min-face-count",
        type=int,
        default=3,
        help="Minimum face_count required before an event is allowed to create or update identities.",
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


def write_outputs(
    output_dir: Path,
    assignments: List[Dict[str, Any]],
    identities: List[IdentityState],
    args: argparse.Namespace,
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
        "embedding_runs_dir": str(args.embedding_runs_dir),
        "match_threshold": args.match_threshold,
        "min_face_count": args.min_face_count,
        "assignments_file": str(assignments_path),
        "gallery_file": str(gallery_path),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


def print_summary(assignments: List[Dict[str, Any]], identities: List[IdentityState]) -> None:
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


def main() -> None:
    args = build_argparser().parse_args()
    print("Step 9: offline local identity matching from event mean embeddings.")

    events = load_event_embeddings(args.embedding_runs_dir)
    assignments, identities = run_matcher(
        events=events,
        match_threshold=args.match_threshold,
        min_face_count=args.min_face_count,
    )
    write_outputs(args.output_dir, assignments, identities, args)
    print_summary(assignments, identities)


if __name__ == "__main__":
    main()
