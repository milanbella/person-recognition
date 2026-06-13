from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from pipeline.depth import DepthSample
from pipeline.face_identity import RecognizedFace
from pipeline.tracking import Track
from pipeline.body_evidence import BodyEvidence
from pipeline.visit_identity import BodyAppearance, VisitAssignment


DEFAULT_ENTRANCE_MERGE_WINDOW_SECONDS = 2.0
DEFAULT_OBSERVER_MATCH_THRESHOLD = 0.58
DEFAULT_OBSERVER_VISIT_MAX_AGE_SECONDS = 1800.0

VISIT_ORIGIN_ENTRANCE = "entrance_confirmed"
VISIT_ORIGIN_OBSERVER = "observer_only"

CAMERA_ROLE_ENTRANCE = "entrance"
CAMERA_ROLE_OBSERVER = "observer"
CAMERA_ROLE_ENTRANCE_OBSERVER = "entrance_observer"
CAMERA_ROLE_CHOICES = [
    CAMERA_ROLE_ENTRANCE,
    CAMERA_ROLE_OBSERVER,
    CAMERA_ROLE_ENTRANCE_OBSERVER,
]


def is_entrance_enabled(camera_role: str) -> bool:
    return camera_role in {CAMERA_ROLE_ENTRANCE, CAMERA_ROLE_ENTRANCE_OBSERVER}


def is_observer_enabled(camera_role: str) -> bool:
    return camera_role in {CAMERA_ROLE_OBSERVER, CAMERA_ROLE_ENTRANCE_OBSERVER}


@dataclass
class FrameEvidence:
    device_id: str
    host_seconds: float
    camera_role: str
    tracks: Sequence[Track]
    depth_samples_by_track: Mapping[int, DepthSample]
    recognized_faces: Sequence[RecognizedFace]
    body_evidence_by_track: Mapping[int, BodyEvidence]


@dataclass
class TrackVisitEvidence:
    camera_role: str
    device_id: str
    track_id: int
    host_seconds: float
    track_bbox: tuple[int, int, int, int]
    face_identity_ids: tuple[str, ...] = ()
    body_appearance: BodyAppearance | None = None
    depth_mm: float | None = None


@dataclass
class ShopVisit:
    visit_id: int
    origin: str
    created_host_seconds: float
    last_seen_host_seconds: float
    last_device_id: str
    last_track_id: int
    appearance: BodyAppearance | None = None
    depth_mm: float | None = None
    observation_count: int = 0
    face_identity_ids: set[str] = field(default_factory=set)
    entrance_observation_times: list[float] = field(default_factory=list)
    observer_observation_count: int = 0
    merged_visit_ids: set[int] = field(default_factory=set)


@dataclass
class VisitRegistryDecision:
    assignment: VisitAssignment
    decision: str
    reason: str
    score: float | None = None
    matched_visit_id: int | None = None
    score_breakdown: dict[str, float | int | str | None] | None = None


def add_visit_registry_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--camera-role",
        nargs="*",
        choices=CAMERA_ROLE_CHOICES,
        default=None,
        help=(
            "Role for each --device-id in synced replay. Defaults to entrance for every device. "
            "Use entrance_observer for entrance cameras that should also contribute observer evidence. "
            "Use observer for in-shop cameras that should match existing visits or create observer-only visits."
        ),
    )
    parser.add_argument(
        "--entrance-merge-window-seconds",
        type=float,
        default=DEFAULT_ENTRANCE_MERGE_WINDOW_SECONDS,
        help="Maximum time gap for entrance-camera plane events to merge into one entrance-confirmed visit.",
    )
    parser.add_argument(
        "--observer-match-threshold",
        type=float,
        default=None,
        help="Minimum body/depth/face score for observer track evidence to attach to an active visit.",
    )
    parser.add_argument(
        "--observer-visit-max-age-seconds",
        type=float,
        default=DEFAULT_OBSERVER_VISIT_MAX_AGE_SECONDS,
        help="How long inactive visits remain match candidates for observer cameras.",
    )
    parser.add_argument(
        "--log-visit-decisions",
        action="store_true",
        help="Print visit registry assignment and merge decisions for tuning.",
    )
    return parser


def build_track_visit_evidence(frame_evidence: FrameEvidence) -> dict[int, TrackVisitEvidence]:
    faces_by_track: dict[int, set[str]] = {}
    for face in frame_evidence.recognized_faces:
        if face.track_id is None:
            continue
        faces_by_track.setdefault(face.track_id, set()).add(face.identity_id)

    track_evidence_by_id: dict[int, TrackVisitEvidence] = {}
    for track in frame_evidence.tracks:
        if track.status == "REMOVED":
            continue
        depth_sample = frame_evidence.depth_samples_by_track.get(track.track_id)
        body_evidence = frame_evidence.body_evidence_by_track.get(track.track_id)
        track_evidence_by_id[track.track_id] = TrackVisitEvidence(
            camera_role=frame_evidence.camera_role,
            device_id=frame_evidence.device_id,
            track_id=track.track_id,
            host_seconds=frame_evidence.host_seconds,
            track_bbox=(track.x1, track.y1, track.x2, track.y2),
            face_identity_ids=tuple(sorted(faces_by_track.get(track.track_id, set()))),
            body_appearance=None if body_evidence is None else body_evidence.appearance,
            depth_mm=None if depth_sample is None else depth_sample.depth_mm,
        )
    return track_evidence_by_id


class VisitRegistry:
    def __init__(
        self,
        *,
        entrance_merge_window_seconds: float = DEFAULT_ENTRANCE_MERGE_WINDOW_SECONDS,
        observer_match_threshold: float = DEFAULT_OBSERVER_MATCH_THRESHOLD,
        observer_visit_max_age_seconds: float = DEFAULT_OBSERVER_VISIT_MAX_AGE_SECONDS,
        log_decisions: bool = False,
    ) -> None:
        self.entrance_merge_window_seconds = entrance_merge_window_seconds
        self.observer_match_threshold = observer_match_threshold
        self.observer_visit_max_age_seconds = observer_visit_max_age_seconds
        self.log_decisions = log_decisions
        self.next_visit_id = 1
        self.visits: dict[int, ShopVisit] = {}
        self.track_to_visit: dict[tuple[str, int], int] = {}
        self.face_to_visit: dict[str, int] = {}

    def resolve_existing_track(self, track_evidence: TrackVisitEvidence) -> VisitRegistryDecision | None:
        observation = track_evidence
        visit_id = self.track_to_visit.get((observation.device_id, observation.track_id))
        if visit_id is None:
            return None
        visit = self.visits.get(visit_id)
        if visit is None:
            return None
        self._update_visit(visit, observation)
        return self._decision(
            observation=observation,
            visit=visit,
            decision="existing_track_mapping",
            reason="track_already_bound_to_visit",
            score=None,
            matched_visit_id=visit.visit_id,
        )

    def resolve_entrance_track(self, track_evidence: TrackVisitEvidence) -> VisitRegistryDecision:
        observation = track_evidence
        existing = self.resolve_existing_track(observation)
        if existing is not None:
            visit = self.visits[existing.assignment.visit_id]
            if visit.origin != VISIT_ORIGIN_ENTRANCE:
                visit.origin = VISIT_ORIGIN_ENTRANCE
                existing.assignment.origin = visit.origin
            if observation.host_seconds not in visit.entrance_observation_times:
                visit.entrance_observation_times.append(observation.host_seconds)
            return existing

        visit = self._find_entrance_time_match(observation)
        if visit is not None:
            self._bind_track(observation, visit)
            self._update_visit(visit, observation)
            visit.entrance_observation_times.append(observation.host_seconds)
            return self._decision(
                observation=observation,
                visit=visit,
                decision="entrance_merged",
                reason="entrance_event_time_window",
                score=None,
                matched_visit_id=visit.visit_id,
            )

        visit = self._find_exact_face_match(observation)
        if visit is not None:
            self._bind_track(observation, visit)
            self._update_visit(visit, observation)
            visit.origin = VISIT_ORIGIN_ENTRANCE
            visit.entrance_observation_times.append(observation.host_seconds)
            return self._decision(
                observation=observation,
                visit=visit,
                decision="entrance_merged",
                reason="known_face_promoted_to_entrance",
                score=None,
                matched_visit_id=visit.visit_id,
            )

        visit = self._create_visit(observation, origin=VISIT_ORIGIN_ENTRANCE)
        visit.entrance_observation_times.append(observation.host_seconds)
        return self._decision(
            observation=observation,
            visit=visit,
            decision="new_entrance_visit",
            reason="no_entrance_time_match",
            score=None,
            matched_visit_id=None,
        )

    def resolve_observer_track(self, track_evidence: TrackVisitEvidence) -> VisitRegistryDecision:
        observation = track_evidence
        existing = self.resolve_existing_track(observation)
        if existing is not None:
            visit = self.visits[existing.assignment.visit_id]
            visit.observer_observation_count += 1
            return existing

        visit = self._find_exact_face_match(observation)
        if visit is not None:
            self._bind_track(observation, visit)
            self._update_visit(visit, observation)
            visit.observer_observation_count += 1
            return self._decision(
                observation=observation,
                visit=visit,
                decision="observer_reused",
                reason="known_face_mapping",
                score=None,
                matched_visit_id=visit.visit_id,
            )

        visit, score, score_breakdown = self._find_best_observer_match(
            observation,
            preferred_origin=VISIT_ORIGIN_ENTRANCE,
        )
        if visit is None:
            visit, score, score_breakdown = self._find_best_observer_match(
                observation,
                preferred_origin=VISIT_ORIGIN_OBSERVER,
            )

        if visit is not None and score >= self.observer_match_threshold:
            self._bind_track(observation, visit)
            self._update_visit(visit, observation)
            visit.observer_observation_count += 1
            return self._decision(
                observation=observation,
                visit=visit,
                decision="observer_reused",
                reason="body_depth_time_score_above_threshold",
                score=score,
                matched_visit_id=visit.visit_id,
                score_breakdown=score_breakdown,
            )

        visit = self._create_visit(observation, origin=VISIT_ORIGIN_OBSERVER)
        visit.observer_observation_count += 1
        return self._decision(
            observation=observation,
            visit=visit,
            decision="new_observer_only_visit",
            reason="no_active_visit_match" if score is None else "best_score_below_threshold",
            score=score,
            matched_visit_id=None,
            score_breakdown=score_breakdown,
        )

    def _find_entrance_time_match(self, observation: TrackVisitEvidence) -> ShopVisit | None:
        best_visit: ShopVisit | None = None
        best_gap = float("inf")
        for visit in self.visits.values():
            if visit.origin != VISIT_ORIGIN_ENTRANCE:
                continue
            for event_time in visit.entrance_observation_times:
                gap = abs(observation.host_seconds - event_time)
                if gap <= self.entrance_merge_window_seconds and gap < best_gap:
                    best_gap = gap
                    best_visit = visit
        return best_visit

    def _find_exact_face_match(self, observation: TrackVisitEvidence) -> ShopVisit | None:
        for face_id in observation.face_identity_ids:
            visit_id = self.face_to_visit.get(face_id)
            if visit_id is not None and visit_id in self.visits:
                return self.visits[visit_id]
        return None

    def _find_best_observer_match(
        self,
        observation: TrackVisitEvidence,
        *,
        preferred_origin: str,
    ) -> tuple[ShopVisit | None, float | None, dict[str, float | int | str | None] | None]:
        best_visit: ShopVisit | None = None
        best_score: float | None = None
        best_breakdown: dict[str, float | int | str | None] | None = None
        for visit in self.visits.values():
            if visit.origin != preferred_origin:
                continue
            age_seconds = observation.host_seconds - visit.last_seen_host_seconds
            if age_seconds < 0.0 or age_seconds > self.observer_visit_max_age_seconds:
                continue
            score, breakdown = self._score_observer_candidate(observation, visit, age_seconds)
            if best_score is None or score > best_score:
                best_score = score
                best_visit = visit
                best_breakdown = breakdown
        return best_visit, best_score, best_breakdown

    def _score_observer_candidate(
        self,
        observation: TrackVisitEvidence,
        visit: ShopVisit,
        age_seconds: float,
    ) -> tuple[float, dict[str, float | int | str | None]]:
        appearance_score = _appearance_similarity(observation.body_appearance, visit.appearance)
        depth_score = _depth_similarity(observation.depth_mm, visit.depth_mm)
        time_score = max(0.0, 1.0 - (age_seconds / max(self.observer_visit_max_age_seconds, 1e-6)))
        face_score = 1.0 if set(observation.face_identity_ids) & visit.face_identity_ids else 0.0
        entrance_bonus = 0.05 if visit.origin == VISIT_ORIGIN_ENTRANCE else 0.0
        score = (
            (0.55 * appearance_score)
            + (0.15 * depth_score)
            + (0.15 * time_score)
            + (0.15 * face_score)
            + entrance_bonus
        )
        clamped_score = max(0.0, min(1.0, score))
        return clamped_score, {
            "candidate_visit_id": visit.visit_id,
            "candidate_origin": visit.origin,
            "age_seconds": age_seconds,
            "appearance_score": appearance_score,
            "depth_score": depth_score,
            "time_score": time_score,
            "face_score": face_score,
            "entrance_bonus": entrance_bonus,
            "weighted_score": clamped_score,
        }

    def _create_visit(self, observation: TrackVisitEvidence, *, origin: str) -> ShopVisit:
        visit = ShopVisit(
            visit_id=self.next_visit_id,
            origin=origin,
            created_host_seconds=observation.host_seconds,
            last_seen_host_seconds=observation.host_seconds,
            last_device_id=observation.device_id,
            last_track_id=observation.track_id,
        )
        self.next_visit_id += 1
        self.visits[visit.visit_id] = visit
        self._bind_track(observation, visit)
        self._update_visit(visit, observation)
        return visit

    def _bind_track(self, observation: TrackVisitEvidence, visit: ShopVisit) -> None:
        self.track_to_visit[(observation.device_id, observation.track_id)] = visit.visit_id

    def _update_visit(self, visit: ShopVisit, observation: TrackVisitEvidence) -> None:
        count = visit.observation_count
        if observation.body_appearance is not None:
            if visit.appearance is None:
                visit.appearance = observation.body_appearance
            else:
                merged_hist = ((visit.appearance.histogram * count) + observation.body_appearance.histogram) / (count + 1)
                norm = float(np.linalg.norm(merged_hist))
                if norm > 1e-8:
                    merged_hist = merged_hist / norm
                visit.appearance = BodyAppearance(
                    histogram=merged_hist.astype(np.float32),
                    aspect_ratio=(
                        (visit.appearance.aspect_ratio * count) + observation.body_appearance.aspect_ratio
                    )
                    / (count + 1),
                    height_px=int(
                        round(((visit.appearance.height_px * count) + observation.body_appearance.height_px) / (count + 1))
                    ),
                )
        if observation.depth_mm is not None:
            visit.depth_mm = (
                observation.depth_mm
                if visit.depth_mm is None
                else ((visit.depth_mm * count) + observation.depth_mm) / (count + 1)
            )
        visit.face_identity_ids.update(observation.face_identity_ids)
        for face_id in observation.face_identity_ids:
            self.face_to_visit[face_id] = visit.visit_id
        visit.last_seen_host_seconds = observation.host_seconds
        visit.last_device_id = observation.device_id
        visit.last_track_id = observation.track_id
        visit.observation_count += 1

    def _decision(
        self,
        *,
        observation: TrackVisitEvidence,
        visit: ShopVisit,
        decision: str,
        reason: str,
        score: float | None,
        matched_visit_id: int | None,
        score_breakdown: dict[str, float | int | str | None] | None = None,
    ) -> VisitRegistryDecision:
        result = VisitRegistryDecision(
            assignment=VisitAssignment(
                visit_id=visit.visit_id,
                track_id=observation.track_id,
                device_id=observation.device_id,
                face_identity_ids=tuple(sorted(visit.face_identity_ids)),
                matched_score=score,
                origin=visit.origin,
            ),
            decision=decision,
            reason=reason,
            score=score,
            matched_visit_id=matched_visit_id,
            score_breakdown=score_breakdown,
        )
        if self.log_decisions:
            score_text = "none" if score is None else f"{score:.3f}"
            breakdown_text = ""
            if score_breakdown is not None:
                breakdown_text = (
                    " "
                    f"candidate_visit_id={score_breakdown.get('candidate_visit_id')} "
                    f"appearance={float(score_breakdown.get('appearance_score', 0.0)):.3f} "
                    f"depth={float(score_breakdown.get('depth_score', 0.0)):.3f} "
                    f"time={float(score_breakdown.get('time_score', 0.0)):.3f} "
                    f"face={float(score_breakdown.get('face_score', 0.0)):.3f} "
                    f"bonus={float(score_breakdown.get('entrance_bonus', 0.0)):.3f}"
                )
            print(
                f"VISIT_REGISTRY device_id={observation.device_id} track_id={observation.track_id} "
                f"visit_id={visit.visit_id} origin={visit.origin} decision={decision} "
                f"reason={reason} score={score_text} time={observation.host_seconds:.3f}"
                f"{breakdown_text}"
            )
        return result


def _appearance_similarity(left: BodyAppearance | None, right: BodyAppearance | None) -> float:
    if left is None or right is None:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left.histogram, right.histogram))))


def _depth_similarity(left_mm: float | None, right_mm: float | None) -> float:
    if left_mm is None or right_mm is None:
        return 0.0
    return max(0.0, 1.0 - (abs(left_mm - right_mm) / 800.0))
