import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


DEFAULT_IDENTITY_RUNS_DIR = Path(__file__).resolve().parent / "identity_runs"
DEFAULT_EMBEDDING_RUNS_DIR = Path(__file__).resolve().parent / "embedding_runs"
DEFAULT_EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"


@dataclass
class Assignment:
    event_name: str
    face_count: int
    best_image: str | None
    decision: str
    assigned_identity: str | None
    best_score: float | None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an HTML reviewer for local identity assignments."
    )
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


def load_assignments(identity_runs_dir: Path) -> List[Assignment]:
    assignments_path = identity_runs_dir / "assignments.json"
    if not assignments_path.exists():
        raise FileNotFoundError(
            f"assignments.json not found. Run phase7_local_identity_matcher.py first: {assignments_path}"
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


def load_best_image_from_summary(summary_path: Path) -> str | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary.get("best_image")


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


def render_html(
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


def main() -> None:
    args = build_argparser().parse_args()
    assignments = load_assignments(args.identity_runs_dir)

    grouped: Dict[str, List[Assignment]] = {}
    skipped: List[Assignment] = []

    for assignment in assignments:
        if assignment.assigned_identity is None:
            skipped.append(assignment)
            continue
        grouped.setdefault(assignment.assigned_identity, []).append(assignment)

    html_text = render_html(grouped, skipped, args.embedding_runs_dir, args.evidence_dir)
    output_path = args.identity_runs_dir / "identity_review.html"
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote identity review HTML to {output_path}")


if __name__ == "__main__":
    main()
