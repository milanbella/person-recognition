from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from pipeline.config import (
    DEFAULT_EMBEDDING_DET_HEIGHT,
    DEFAULT_EMBEDDING_DET_THRESH,
    DEFAULT_EMBEDDING_DET_WIDTH,
    DEFAULT_EMBEDDING_MODEL_PACK,
    DEFAULT_INSIGHTFACE_CACHE_ROOT,
)
from pipeline.embedding import build_face_analyzer, l2_normalize
from pipeline.tracking import Track


DEFAULT_FACE_MATCH_THRESHOLD = 0.68
DEFAULT_FACE_MIN_DET_SCORE = 0.45


@dataclass
class FaceIdentity:
    identity_id: str
    prototype: np.ndarray
    observation_count: int


@dataclass
class RecognizedFace:
    bbox: tuple[int, int, int, int]
    det_score: float
    identity_id: str
    best_score: float | None
    track_id: int | None


def add_face_identity_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--enable-face-recognition",
        action="store_true",
        help="Run InsightFace/ArcFace on replay frames and assign local face identities.",
    )
    parser.add_argument(
        "--face-match-threshold",
        type=float,
        default=DEFAULT_FACE_MATCH_THRESHOLD,
        help="Cosine similarity required to match an existing replay-local face identity.",
    )
    parser.add_argument(
        "--face-min-det-score",
        type=float,
        default=DEFAULT_FACE_MIN_DET_SCORE,
        help="Minimum InsightFace face detection score to accept a face.",
    )
    parser.add_argument(
        "--face-cache-root",
        type=Path,
        default=DEFAULT_INSIGHTFACE_CACHE_ROOT,
        help="InsightFace model cache root.",
    )
    parser.add_argument(
        "--face-model-pack",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL_PACK,
        help="InsightFace model pack name.",
    )
    parser.add_argument(
        "--face-det-width",
        type=int,
        default=DEFAULT_EMBEDDING_DET_WIDTH,
        help="InsightFace face detector input width.",
    )
    parser.add_argument(
        "--face-det-height",
        type=int,
        default=DEFAULT_EMBEDDING_DET_HEIGHT,
        help="InsightFace face detector input height.",
    )
    parser.add_argument(
        "--face-det-thresh",
        type=float,
        default=DEFAULT_EMBEDDING_DET_THRESH,
        help="InsightFace detector threshold.",
    )
    return parser


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(l2_normalize(left), l2_normalize(right)))


def associate_face_to_track(
    bbox: tuple[int, int, int, int],
    tracks: Sequence[Track],
) -> int | None:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    candidates = [
        track
        for track in tracks
        if track.x1 <= cx <= track.x2 and track.y1 <= cy <= track.y2
    ]
    if not candidates:
        return None

    def area(track: Track) -> int:
        return max(0, track.x2 - track.x1) * max(0, track.y2 - track.y1)

    return min(candidates, key=area).track_id


class LocalFaceIdentityMatcher:
    def __init__(
        self,
        *,
        cache_root: Path,
        model_pack: str,
        det_size: tuple[int, int],
        det_thresh: float,
        match_threshold: float,
        min_det_score: float,
    ) -> None:
        self.analyzer = build_face_analyzer(
            cache_root=cache_root,
            model_pack=model_pack,
            det_size=det_size,
            det_thresh=det_thresh,
        )
        self.match_threshold = match_threshold
        self.min_det_score = min_det_score
        self.identities: list[FaceIdentity] = []

    def recognize(
        self,
        frame: np.ndarray,
        *,
        tracks: Sequence[Track],
    ) -> list[RecognizedFace]:
        faces = self.analyzer.get(frame)
        results: list[RecognizedFace] = []
        for face in faces:
            if float(face.det_score) < self.min_det_score:
                continue
            embedding = np.asarray(face.embedding, dtype=np.float32)
            if embedding.size == 0:
                continue
            normalized = l2_normalize(embedding)
            identity, score = self._match_or_create(normalized)
            bbox = self._bbox_from_face(face, frame.shape)
            results.append(
                RecognizedFace(
                    bbox=bbox,
                    det_score=float(face.det_score),
                    identity_id=identity.identity_id,
                    best_score=score,
                    track_id=associate_face_to_track(bbox, tracks),
                )
            )
        return results

    def _match_or_create(self, embedding: np.ndarray) -> tuple[FaceIdentity, float | None]:
        best_identity: FaceIdentity | None = None
        best_score = -1.0
        for identity in self.identities:
            score = cosine_similarity(embedding, identity.prototype)
            if score > best_score:
                best_score = score
                best_identity = identity

        if best_identity is not None and best_score >= self.match_threshold:
            self._update_identity(best_identity, embedding)
            return best_identity, best_score

        identity = FaceIdentity(
            identity_id=f"face_person_{len(self.identities) + 1:03d}",
            prototype=embedding,
            observation_count=1,
        )
        self.identities.append(identity)
        return identity, None if best_identity is None else best_score

    def _update_identity(self, identity: FaceIdentity, embedding: np.ndarray) -> None:
        count = identity.observation_count
        identity.prototype = l2_normalize(((identity.prototype * count) + embedding) / (count + 1))
        identity.observation_count += 1

    @staticmethod
    def _bbox_from_face(face: Any, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = [int(round(float(value))) for value in face.bbox]
        return (
            max(0, min(width - 1, x1)),
            max(0, min(height - 1, y1)),
            max(0, min(width - 1, x2)),
            max(0, min(height - 1, y2)),
        )


def draw_recognized_faces(
    frame: np.ndarray,
    faces: Sequence[RecognizedFace],
) -> None:
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        color = (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        score_text = "new" if face.best_score is None else f"{face.best_score:.2f}"
        track_text = "" if face.track_id is None else f" T{face.track_id}"
        label = f"{face.identity_id}{track_text} {score_text}"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
