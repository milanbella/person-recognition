from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"),
)
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from pipeline.config import (
    DEFAULT_EMBEDDING_DET_HEIGHT,
    DEFAULT_EMBEDDING_DET_THRESH,
    DEFAULT_EMBEDDING_DET_WIDTH,
    DEFAULT_EMBEDDING_MODEL_PACK,
    DEFAULT_EMBEDDING_RUNS_DIR,
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_INSIGHTFACE_CACHE_ROOT,
)


@dataclass
class FaceEmbeddingResult:
    image_name: str
    det_score: float
    bbox: List[float]
    embedding: np.ndarray
    embedding_norm: float


def build_embedding_argparser(
    description: str = "Step 8: run ArcFace embeddings on saved entrance evidence crops.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_embedding_args(parser)
    return parser


def add_embedding_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help="Directory containing saved evidence event folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_RUNS_DIR,
        help="Directory where embedding outputs will be written.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_INSIGHTFACE_CACHE_ROOT,
        help="InsightFace model cache root.",
    )
    parser.add_argument(
        "--model-pack",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL_PACK,
        help="InsightFace model pack name.",
    )
    parser.add_argument(
        "--det-width",
        type=int,
        default=DEFAULT_EMBEDDING_DET_WIDTH,
        help="Face detector input width.",
    )
    parser.add_argument(
        "--det-height",
        type=int,
        default=DEFAULT_EMBEDDING_DET_HEIGHT,
        help="Face detector input height.",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=DEFAULT_EMBEDDING_DET_THRESH,
        help="Minimum face detection threshold inside evidence crops.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Optional cap on how many event folders to process. 0 means all.",
    )
    return parser


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32, copy=True)
    return (vector / norm).astype(np.float32, copy=False)


def select_runtime(cache_root: Path, model_pack: str) -> Tuple[List[str], int]:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        try:
            session = ort.InferenceSession(
                str(cache_root / "models" / model_pack / "w600k_r50.onnx"),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            applied = session.get_providers()
            if "CUDAExecutionProvider" in applied:
                print("Using ONNX Runtime CUDAExecutionProvider for ArcFace.")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
        except Exception as exc:
            print(f"CUDAExecutionProvider unavailable for ArcFace, falling back to CPU: {exc}")

    print("Using ONNX Runtime CPUExecutionProvider for ArcFace.")
    return ["CPUExecutionProvider"], -1


def build_face_analyzer(
    cache_root: Path,
    model_pack: str,
    det_size: Tuple[int, int],
    det_thresh: float,
) -> FaceAnalysis:
    cache_root.mkdir(parents=True, exist_ok=True)
    providers, ctx_id = select_runtime(cache_root, model_pack)
    app = FaceAnalysis(
        name=model_pack,
        root=str(cache_root),
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
    return app


def list_event_dirs(evidence_dir: Path, max_events: int) -> List[Path]:
    if not evidence_dir.exists():
        raise FileNotFoundError(f"Evidence directory not found: {evidence_dir}")

    event_dirs = [path for path in sorted(evidence_dir.iterdir()) if path.is_dir()]
    if max_events > 0:
        event_dirs = event_dirs[:max_events]
    return event_dirs


def iter_event_images(event_dir: Path) -> Sequence[Path]:
    return [
        path
        for path in sorted(event_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


def choose_best_face(faces: Sequence[Any]) -> Any | None:
    if not faces:
        return None

    def rank(face: Any) -> Tuple[float, float]:
        x1, y1, x2, y2 = [float(value) for value in face.bbox]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return float(face.det_score), area

    return max(faces, key=rank)


def analyze_event(event_dir: Path, analyzer: FaceAnalysis) -> List[FaceEmbeddingResult]:
    results: List[FaceEmbeddingResult] = []

    for image_path in iter_event_images(event_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"EMBED_WARNING event={event_dir.name} image={image_path.name} unreadable")
            continue

        faces = analyzer.get(image)
        face = choose_best_face(faces)
        if face is None:
            print(f"EMBED_WARNING event={event_dir.name} image={image_path.name} no_face")
            continue

        embedding = np.asarray(face.embedding, dtype=np.float32)
        normalized = l2_normalize(embedding)
        bbox = [float(value) for value in face.bbox.tolist()]

        results.append(
            FaceEmbeddingResult(
                image_name=image_path.name,
                det_score=float(face.det_score),
                bbox=bbox,
                embedding=normalized,
                embedding_norm=float(np.linalg.norm(embedding)),
            )
        )

    return results


def write_event_outputs(
    event_dir: Path,
    output_root: Path,
    results: Sequence[FaceEmbeddingResult],
) -> Dict[str, Any]:
    output_dir = output_root / event_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "event_dir": str(event_dir),
        "output_dir": str(output_dir),
        "face_count": len(results),
        "embedding_dim": 0,
        "images": [],
        "mean_embedding_file": None,
    }

    if not results:
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    embeddings = np.stack([result.embedding for result in results], axis=0)
    mean_embedding = l2_normalize(np.mean(embeddings, axis=0))

    embeddings_path = output_dir / "embeddings.npy"
    mean_embedding_path = output_dir / "mean_embedding.npy"
    np.save(embeddings_path, embeddings)
    np.save(mean_embedding_path, mean_embedding)

    best = max(results, key=lambda item: item.det_score)
    summary["embedding_dim"] = int(embeddings.shape[1])
    summary["mean_embedding_file"] = mean_embedding_path.name
    summary["best_image"] = best.image_name
    summary["images"] = [
        {
            "image_name": result.image_name,
            "det_score": result.det_score,
            "bbox": result.bbox,
            "embedding_norm": result.embedding_norm,
        }
        for result in results
    ]

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_embedding_pipeline(
    *,
    evidence_dir: Path,
    output_dir: Path,
    cache_root: Path,
    model_pack: str,
    det_size: Tuple[int, int],
    det_thresh: float,
    max_events: int,
) -> List[Dict[str, Any]]:
    analyzer = build_face_analyzer(
        cache_root=cache_root,
        model_pack=model_pack,
        det_size=det_size,
        det_thresh=det_thresh,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    event_dirs = list_event_dirs(evidence_dir, max_events)
    print(f"Processing {len(event_dirs)} evidence event folders from {evidence_dir}")

    run_summary: List[Dict[str, Any]] = []
    for event_dir in event_dirs:
        results = analyze_event(event_dir, analyzer)
        summary = write_event_outputs(event_dir, output_dir, results)
        run_summary.append(summary)
        print(
            f"EMBED_EVENT event={event_dir.name} faces={summary['face_count']} "
            f"embedding_dim={summary['embedding_dim']}"
        )

    run_summary_path = output_dir / "run_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(f"Wrote run summary to {run_summary_path}")
    return run_summary
