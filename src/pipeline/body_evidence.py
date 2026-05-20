from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from pipeline.config import DEFAULT_BODY_EVIDENCE_BACKEND
from pipeline.tracking import Track
from pipeline.visit_identity import BodyAppearance, extract_body_appearance


@dataclass
class BodyEvidence:
    track_id: int
    appearance: BodyAppearance | None
    embedding: np.ndarray | None = None
    quality: float = 1.0
    backend: str = DEFAULT_BODY_EVIDENCE_BACKEND


class BodyEvidenceExtractor(Protocol):
    def extract(
        self,
        frame: np.ndarray,
        *,
        tracks: Sequence[Track],
    ) -> dict[int, BodyEvidence]:
        ...


class HsvBodyEvidenceExtractor:
    def extract(
        self,
        frame: np.ndarray,
        *,
        tracks: Sequence[Track],
    ) -> dict[int, BodyEvidence]:
        evidence: dict[int, BodyEvidence] = {}
        for track in tracks:
            if track.status == "REMOVED":
                continue
            evidence[track.track_id] = BodyEvidence(
                track_id=track.track_id,
                appearance=extract_body_appearance(frame, track),
            )
        return evidence


def add_body_evidence_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--body-backend",
        choices=["hsv"],
        default=DEFAULT_BODY_EVIDENCE_BACKEND,
        help="Body evidence backend used for visit matching. Current default is HSV color histograms.",
    )
    return parser


def build_body_evidence_extractor(args: argparse.Namespace) -> BodyEvidenceExtractor:
    backend = getattr(args, "body_backend", DEFAULT_BODY_EVIDENCE_BACKEND)
    if backend != "hsv":
        raise ValueError(f"Unsupported body evidence backend: {backend}")
    return HsvBodyEvidenceExtractor()
