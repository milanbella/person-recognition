from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from pipeline.config import DEFAULT_EMBEDDING_RUNS_DIR, DEFAULT_EVIDENCE_DIR


@dataclass
class EventEmbedding:
    event_name: str
    event_dir: Path
    embedding: np.ndarray
    face_count: int
    best_image: str | None


@dataclass
class Neighbor:
    event_name: str
    cosine_similarity: float


@dataclass
class EventReview:
    event_name: str
    face_count: int
    best_image: str | None
    event_dir: Path
    neighbors: List[Neighbor]


def build_similarity_review_argparser(
    description: str = "Review cosine similarities between saved event mean embeddings.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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


def build_similarity_html_argparser(
    description: str = "Generate an HTML reviewer for event embedding similarities.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--embedding-runs-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_RUNS_DIR,
        help="Directory containing embedding outputs and similarity_review.json.",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help="Directory containing saved evidence event folders.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many nearest neighbors to show for each event.",
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


def run_similarity_review(embedding_runs_dir: Path, top_k: int) -> List[Dict[str, object]]:
    events = load_event_embeddings(embedding_runs_dir)
    vectors = np.stack([event.embedding for event in events], axis=0)
    similarities = cosine_similarity_matrix(vectors)

    matrix_csv = embedding_runs_dir / "cosine_similarity_matrix.csv"
    run_summary_json = embedding_runs_dir / "similarity_review.json"

    write_similarity_csv(matrix_csv, events, similarities)
    run_summary = build_run_summary(events, similarities)
    run_summary_json.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(f"Loaded {len(events)} event mean embeddings from {embedding_runs_dir}")
    print(f"Wrote cosine similarity matrix to {matrix_csv}")
    print(f"Wrote similarity review to {run_summary_json}")
    print_neighbor_report(events, similarities, top_k)
    return run_summary


def load_similarity_review(embedding_runs_dir: Path, evidence_dir: Path, top_k: int) -> List[EventReview]:
    review_path = embedding_runs_dir / "similarity_review.json"
    if not review_path.exists():
        raise FileNotFoundError(
            f"similarity_review.json not found. Run 02_replay_similarity_review.py first: {review_path}"
        )

    raw_items = json.loads(review_path.read_text(encoding="utf-8"))
    result: List[EventReview] = []
    for item in raw_items:
        result.append(
            EventReview(
                event_name=item["event_name"],
                face_count=int(item.get("face_count", 0)),
                best_image=item.get("best_image"),
                event_dir=evidence_dir / item["event_name"],
                neighbors=[
                    Neighbor(
                        event_name=neighbor["event_name"],
                        cosine_similarity=float(neighbor["cosine_similarity"]),
                    )
                    for neighbor in item.get("neighbors", [])[:top_k]
                ],
            )
        )
    return result


def image_uri(path: Path) -> str:
    return path.resolve().as_uri()


def best_image_path(event_dir: Path, best_image: str | None) -> Path | None:
    if best_image is None:
        return None
    path = event_dir / best_image
    if path.exists():
        return path
    return None


def load_best_image_from_summary(summary_path: Path) -> str | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary.get("best_image")


def render_event_card(event: EventReview, evidence_dir: Path, embedding_runs_dir: Path) -> str:
    event_image = best_image_path(event.event_dir, event.best_image)
    event_img_html = (
        f'<img src="{image_uri(event_image)}" alt="{html.escape(event.event_name)}">'
        if event_image is not None
        else '<div class="missing">No best image</div>'
    )

    neighbors_html: List[str] = []
    for neighbor in event.neighbors:
        neighbor_dir = evidence_dir / neighbor.event_name
        neighbor_image = best_image_path(
            neighbor_dir,
            load_best_image_from_summary(embedding_runs_dir / neighbor.event_name / "summary.json"),
        )
        neighbor_img_html = (
            f'<img src="{image_uri(neighbor_image)}" alt="{html.escape(neighbor.event_name)}">'
            if neighbor_image is not None
            else '<div class="missing">No best image</div>'
        )
        neighbors_html.append(
            f"""
            <div class="neighbor-card">
              <div class="neighbor-score">cosine {neighbor.cosine_similarity:.4f}</div>
              <div class="neighbor-name">{html.escape(neighbor.event_name)}</div>
              {neighbor_img_html}
            </div>
            """
        )

    return f"""
    <section class="event-card">
      <div class="event-main">
        <div class="event-meta">
          <h2>{html.escape(event.event_name)}</h2>
          <div>face_count={event.face_count}</div>
          <div>best_image={html.escape(event.best_image or "n/a")}</div>
        </div>
        <div class="event-image">{event_img_html}</div>
      </div>
      <div class="neighbors">
        {''.join(neighbors_html)}
      </div>
    </section>
    """


def render_similarity_html(events: List[EventReview], evidence_dir: Path, embedding_runs_dir: Path) -> str:
    cards = "".join(
        render_event_card(event, evidence_dir, embedding_runs_dir) for event in events
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Embedding Similarity Review</title>
  <style>
    body {{
      font-family: Segoe UI, Arial, sans-serif;
      margin: 24px;
      background: #f3f0e8;
      color: #1f1f1f;
    }}
    h1 {{
      margin-bottom: 8px;
    }}
    .intro {{
      margin-bottom: 24px;
      color: #404040;
    }}
    .event-card {{
      border: 1px solid #d0c8b8;
      background: #fffdf8;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 20px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }}
    .event-main {{
      display: flex;
      gap: 20px;
      align-items: flex-start;
      margin-bottom: 16px;
    }}
    .event-meta {{
      min-width: 360px;
    }}
    .event-meta h2 {{
      margin: 0 0 10px 0;
      font-size: 20px;
    }}
    .event-image img, .neighbor-card img {{
      width: 220px;
      max-width: 100%;
      border-radius: 8px;
      border: 1px solid #d8d2c4;
    }}
    .neighbors {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .neighbor-card {{
      width: 240px;
      background: #faf7ef;
      border: 1px solid #e2dbcb;
      border-radius: 10px;
      padding: 12px;
    }}
    .neighbor-score {{
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .neighbor-name {{
      font-size: 13px;
      color: #4d4d4d;
      margin-bottom: 10px;
      word-break: break-word;
    }}
    .missing {{
      width: 220px;
      height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #ece7db;
      border-radius: 8px;
      border: 1px dashed #bdb5a3;
      color: #555;
    }}
  </style>
</head>
<body>
  <h1>Embedding Similarity Review</h1>
  <div class="intro">
    Each event shows its best evidence crop and the top similar events by cosine similarity.
    Use this to visually decide whether high-similarity pairs are actually the same person.
  </div>
  {cards}
</body>
</html>
"""


def write_similarity_html(
    embedding_runs_dir: Path,
    evidence_dir: Path,
    top_k: int,
) -> Path:
    events = load_similarity_review(embedding_runs_dir, evidence_dir, top_k)
    html_text = render_similarity_html(events, evidence_dir, embedding_runs_dir)
    output_path = embedding_runs_dir / "similarity_review.html"
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote HTML review to {output_path}")
    return output_path
