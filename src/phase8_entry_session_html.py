import argparse
import html
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "entry_session_runs"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an HTML reviewer for EntryEvent and EntrySessionPacket artifacts."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory containing entry_events, entry_sessions, and review outputs.",
    )
    return parser


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


def render_html(
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


def main() -> None:
    args = build_argparser().parse_args()
    summary = load_json(args.output_root / "run_summary.json")
    event_lookup = load_entry_events(args.output_root / "entry_events")
    sessions = load_entry_sessions(args.output_root / "entry_sessions")
    decisions = load_merge_decisions(args.output_root / "review" / "merge_decisions.json")
    html_text = render_html(summary, sessions, event_lookup, decisions)
    output_path = args.output_root / "review" / "entry_session_review.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote entry session review HTML to {output_path}")


if __name__ == "__main__":
    main()
