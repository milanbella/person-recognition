import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


DEFAULT_EMBEDDING_RUNS_DIR = Path(__file__).resolve().parent / "embedding_runs"


@dataclass
class EventEmbedding:
    event_name: str
    event_dir: Path
    embedding: np.ndarray
    face_count: int
    best_image: str | None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review cosine similarities between saved event mean embeddings."
    )
    parser.add_argument(
        "--embedding-runs-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_RUNS_DIR,
        help="Directory containing event embedding outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many nearest neighbors to print for each event.",
    )
    return parser


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
        embedding = np.load(embedding_path).astype(np.float32)
        events.append(
            EventEmbedding(
                event_name=event_dir.name,
                event_dir=event_dir,
                embedding=embedding,
                face_count=int(summary.get("face_count", 0)),
                best_image=summary.get("best_image"),
            )
        )

    if not events:
        raise RuntimeError(f"No event embeddings found under {embedding_runs_dir}")

    return events


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.clip(norms, 1e-12, None)
    normalized = vectors / safe_norms
    return normalized @ normalized.T


def write_similarity_csv(
    output_path: Path,
    events: List[EventEmbedding],
    similarities: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_name", *[event.event_name for event in events]])
        for row_index, event in enumerate(events):
            row = [event.event_name]
            row.extend(f"{similarities[row_index, col_index]:.6f}" for col_index in range(len(events)))
            writer.writerow(row)


def print_neighbor_report(events: List[EventEmbedding], similarities: np.ndarray, top_k: int) -> None:
    for row_index, event in enumerate(events):
        print(
            f"EVENT {event.event_name} face_count={event.face_count} "
            f"best_image={event.best_image or 'n/a'}"
        )
        neighbor_indices = np.argsort(similarities[row_index])[::-1]
        printed = 0
        for neighbor_index in neighbor_indices:
            if neighbor_index == row_index:
                continue
            neighbor = events[neighbor_index]
            print(
                f"  NEIGHBOR {neighbor.event_name} "
                f"cosine={similarities[row_index, neighbor_index]:.4f}"
            )
            printed += 1
            if printed >= top_k:
                break


def build_run_summary(events: List[EventEmbedding], similarities: np.ndarray) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for row_index, event in enumerate(events):
        neighbors = []
        neighbor_indices = np.argsort(similarities[row_index])[::-1]
        for neighbor_index in neighbor_indices:
            if neighbor_index == row_index:
                continue
            neighbor = events[neighbor_index]
            neighbors.append(
                {
                    "event_name": neighbor.event_name,
                    "cosine_similarity": float(similarities[row_index, neighbor_index]),
                }
            )

        result.append(
            {
                "event_name": event.event_name,
                "face_count": event.face_count,
                "best_image": event.best_image,
                "neighbors": neighbors,
            }
        )
    return result


def main() -> None:
    args = build_argparser().parse_args()
    events = load_event_embeddings(args.embedding_runs_dir)
    vectors = np.stack([event.embedding for event in events], axis=0)
    similarities = cosine_similarity_matrix(vectors)

    matrix_csv = args.embedding_runs_dir / "cosine_similarity_matrix.csv"
    run_summary_json = args.embedding_runs_dir / "similarity_review.json"

    write_similarity_csv(matrix_csv, events, similarities)
    run_summary = build_run_summary(events, similarities)
    run_summary_json.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(f"Loaded {len(events)} event mean embeddings from {args.embedding_runs_dir}")
    print(f"Wrote cosine similarity matrix to {matrix_csv}")
    print(f"Wrote similarity review to {run_summary_json}")
    print_neighbor_report(events, similarities, args.top_k)


if __name__ == "__main__":
    main()
