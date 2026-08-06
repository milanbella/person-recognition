from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from pipeline.operator_models import AnalysisResult


ENTRY_BEFORE_MILLISECONDS = 2000
ENTRY_AFTER_MILLISECONDS = 5000
LEAVE_BEFORE_MILLISECONDS = 2000
LEAVE_AFTER_MILLISECONDS = 5000
SHELF_BEFORE_MILLISECONDS = 1000
SHELF_AFTER_MILLISECONDS = 5000


def analyze_test_run(
    *,
    run: Mapping[str, Any],
    subjects: Sequence[Mapping[str, Any]],
    physical_visits: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, int | None]]:
    results: list[AnalysisResult] = []
    consumed_event_ids: set[int] = set()
    physical_visit_mappings = _physical_visit_mappings(
        physical_visits=physical_visits,
        annotations=annotations,
        events=events,
    )

    for annotation in annotations:
        annotation_type = annotation["annotationType"]
        annotation_id = int(annotation["annotationId"])
        event_type = {
            "physical_entry": "entry_accepted",
            "physical_leave": "leave_accepted",
            "shelf_approach": "shelf_approach",
            "shelf_departure": "shelf_departure",
        }.get(annotation_type)
        if event_type is None:
            continue

        before_ms, after_ms = {
            "physical_entry": (
                ENTRY_BEFORE_MILLISECONDS,
                ENTRY_AFTER_MILLISECONDS,
            ),
            "physical_leave": (
                LEAVE_BEFORE_MILLISECONDS,
                LEAVE_AFTER_MILLISECONDS,
            ),
            "shelf_approach": (
                SHELF_BEFORE_MILLISECONDS,
                SHELF_AFTER_MILLISECONDS,
            ),
            "shelf_departure": (
                SHELF_BEFORE_MILLISECONDS,
                SHELF_AFTER_MILLISECONDS,
            ),
        }[annotation_type]
        expected_time = int(annotation["receivedAtUnixMilliseconds"])
        physical_visit_id = annotation.get("physicalVisitId")
        mapped_visit_id = physical_visit_mappings.get(physical_visit_id)
        shelf_id = annotation.get("payload", {}).get("shelfId")
        match = _nearest_event(
            events,
            event_type=event_type,
            expected_time=expected_time,
            before_milliseconds=before_ms,
            after_milliseconds=after_ms,
            visit_id=(
                None
                if annotation_type == "physical_entry"
                else mapped_visit_id
            ),
            shelf_id=None if shelf_id is None else int(shelf_id),
            excluded_event_ids=consumed_event_ids,
        )
        if match is None:
            results.append(
                AnalysisResult(
                    rule_id=f"{annotation_type}:{annotation_id}",
                    status="fail",
                    summary=_missing_summary(annotation_type, shelf_id),
                    expected_annotation_id=annotation_id,
                    details={
                        "physicalVisitId": physical_visit_id,
                        "mappedSystemVisitId": mapped_visit_id,
                        "matchingWindowMilliseconds": {
                            "before": before_ms,
                            "after": after_ms,
                        },
                    },
                )
            )
            continue

        event_id = int(match["eventId"])
        consumed_event_ids.add(event_id)
        latency = int(match["occurredAtUnixMilliseconds"]) - expected_time
        matched_visit_id = match.get("visitId")
        status = "pass"
        summary = _matched_summary(
            annotation_type,
            matched_visit_id=matched_visit_id,
            shelf_id=shelf_id,
        )
        if (
            mapped_visit_id is not None
            and annotation_type != "physical_entry"
            and matched_visit_id != mapped_visit_id
        ):
            status = "fail"
            summary = (
                f"{event_type} used visit {matched_visit_id}; "
                f"expected visit {mapped_visit_id}."
            )
        results.append(
            AnalysisResult(
                rule_id=f"{annotation_type}:{annotation_id}",
                status=status,
                summary=summary,
                expected_annotation_id=annotation_id,
                matched_event_id=event_id,
                latency_milliseconds=latency,
                details={"event": match},
            )
        )

    results.extend(
        _visit_continuity_results(
            physical_visits=physical_visits,
            annotations=annotations,
            mappings=physical_visit_mappings,
        )
    )
    results.extend(
        _customer_results(
            subjects=subjects,
            physical_visits=physical_visits,
            annotations=annotations,
            mappings=physical_visit_mappings,
        )
    )

    monitored_event_types = {
        "entry_accepted",
        "leave_accepted",
        "shelf_approach",
        "shelf_departure",
    }
    event_verdicts = {
        int(annotation["payload"]["systemEventId"]): annotation
        for annotation in annotations
        if annotation["annotationType"]
        in {"system_event_correct", "system_event_incorrect"}
        and annotation.get("payload", {}).get("systemEventId") is not None
    }
    for event in events:
        event_id = int(event["eventId"])
        if (
            event.get("eventType") in monitored_event_types
            and event_id not in consumed_event_ids
        ):
            verdict = event_verdicts.get(event_id)
            if verdict is not None:
                correct = verdict["annotationType"] == "system_event_correct"
                results.append(
                    AnalysisResult(
                        rule_id=f"event-verdict:{event_id}",
                        status="pass" if correct else "fail",
                        summary=(
                            f"Operator confirmed {event['eventType']} event "
                            f"{event_id}."
                            if correct
                            else f"Operator rejected {event['eventType']} event "
                            f"{event_id} as inconsistent with physical reality."
                        ),
                        expected_annotation_id=int(verdict["annotationId"]),
                        matched_event_id=event_id,
                        details={"event": event},
                    )
                )
                consumed_event_ids.add(event_id)
                continue
            results.append(
                AnalysisResult(
                    rule_id=f"unexpected:{event['eventType']}:{event_id}",
                    status="fail",
                    summary=(
                        f"Unexpected system event {event['eventType']} "
                        f"for visit {event.get('visitId')}."
                    ),
                    matched_event_id=event_id,
                    details={"event": event},
                )
            )

    for annotation in annotations:
        annotation_type = annotation.get("annotationType")
        if annotation_type not in {
            "world_state_claim_correct",
            "world_state_claim_incorrect",
            "physical_subject_state",
        }:
            continue
        payload = annotation.get("payload", {})
        annotation_id = int(annotation["annotationId"])
        claim = payload.get("claim", "subject_state")
        if annotation_type == "world_state_claim_correct":
            results.append(
                AnalysisResult(
                    rule_id=f"world-state-claim:{annotation_id}",
                    status="pass",
                    summary=(
                        f"Operator confirmed world-state claim {claim}="
                        f"{payload.get('systemValue')!r}."
                    ),
                    expected_annotation_id=annotation_id,
                    details={
                        "claim": claim,
                        "systemValue": payload.get("systemValue"),
                        "worldStateRef": payload.get("worldStateRef"),
                    },
                )
            )
        elif annotation_type == "world_state_claim_incorrect":
            results.append(
                AnalysisResult(
                    rule_id=f"world-state-claim:{annotation_id}",
                    status="fail",
                    summary=(
                        f"World-state claim {claim} was {payload.get('systemValue')!r}; "
                        f"physical value was {payload.get('physicalValue')!r}."
                    ),
                    expected_annotation_id=annotation_id,
                    details={
                        "claim": claim,
                        "systemValue": payload.get("systemValue"),
                        "physicalValue": payload.get("physicalValue"),
                        "worldStateRef": payload.get("worldStateRef"),
                        "reason": payload.get("reason"),
                    },
                )
            )
        else:
            results.append(
                AnalysisResult(
                    rule_id=f"physical-subject-state:{annotation_id}",
                    status="inconclusive",
                    summary=(
                        f"Recorded physical subject fact {claim}="
                        f"{payload.get('physicalValue')!r} without a selected system claim."
                    ),
                    expected_annotation_id=annotation_id,
                    details=dict(payload),
                )
            )

    counts = Counter(result.status for result in results)
    overall = (
        "fail"
        if counts["fail"]
        else "pending"
        if counts["pending"]
        else "pass"
    )
    report = {
        "runId": run["runId"],
        "scenario": run["scenario"],
        "status": overall,
        "summary": {
            "pass": counts["pass"],
            "fail": counts["fail"],
            "pending": counts["pending"],
            "inconclusive": counts["inconclusive"],
        },
        "physicalVisitMappings": physical_visit_mappings,
        "results": [result.as_payload() for result in results],
    }
    return report, physical_visit_mappings


def _physical_visit_mappings(
    *,
    physical_visits: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, int | None]:
    evidence_by_visit: dict[str, list[int]] = defaultdict(list)
    for annotation in annotations:
        physical_visit_id = annotation.get("physicalVisitId")
        reference = annotation.get("observationRef")
        if (
            physical_visit_id
            and annotation.get("annotationType") == "observation_is_subject"
            and isinstance(reference, Mapping)
            and reference.get("observedVisitId") is not None
        ):
            evidence_by_visit[str(physical_visit_id)].append(
                int(reference["observedVisitId"])
            )

    result: dict[str, int | None] = {}
    for visit in physical_visits:
        physical_visit_id = str(visit["physicalVisitId"])
        candidates = evidence_by_visit.get(physical_visit_id, [])
        if candidates:
            result[physical_visit_id] = Counter(candidates).most_common(1)[0][0]
            continue
        entry_event = _nearest_event(
            events,
            event_type="entry_accepted",
            expected_time=int(visit["enteredAtUnixMilliseconds"]),
            before_milliseconds=ENTRY_BEFORE_MILLISECONDS,
            after_milliseconds=ENTRY_AFTER_MILLISECONDS,
        )
        result[physical_visit_id] = (
            None
            if entry_event is None or entry_event.get("visitId") is None
            else int(entry_event["visitId"])
        )
    return result


def _visit_continuity_results(
    *,
    physical_visits: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    mappings: Mapping[str, int | None],
) -> list[AnalysisResult]:
    results: list[AnalysisResult] = []
    observations: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for annotation in annotations:
        reference = annotation.get("observationRef")
        physical_visit_id = annotation.get("physicalVisitId")
        if (
            annotation.get("annotationType") != "observation_is_subject"
            or not physical_visit_id
            or not isinstance(reference, Mapping)
            or reference.get("observedVisitId") is None
        ):
            continue
        observations[str(physical_visit_id)].append(
            (
                int(reference["cameraIndex"]),
                int(reference["trackId"]),
                int(reference["observedVisitId"]),
            )
        )

    for visit in physical_visits:
        physical_visit_id = str(visit["physicalVisitId"])
        confirmed = observations.get(physical_visit_id, [])
        if not confirmed:
            results.append(
                AnalysisResult(
                    rule_id=f"visit-continuity:{physical_visit_id}",
                    status="pending",
                    summary="No person observation was confirmed for this physical visit.",
                    details={"physicalVisitId": physical_visit_id},
                )
            )
            continue
        system_visits = sorted({item[2] for item in confirmed})
        tracks = sorted({(item[0], item[1]) for item in confirmed})
        if len(system_visits) == 1:
            results.append(
                AnalysisResult(
                    rule_id=f"visit-continuity:{physical_visit_id}",
                    status="pass",
                    summary=(
                        f"{len(tracks)} confirmed camera tracks remained on "
                        f"visit {system_visits[0]}."
                    ),
                    details={
                        "physicalVisitId": physical_visit_id,
                        "systemVisitIds": system_visits,
                        "cameraTracks": tracks,
                    },
                )
            )
        else:
            results.append(
                AnalysisResult(
                    rule_id=f"visit-continuity:{physical_visit_id}",
                    status="fail",
                    summary=(
                        "One physical visit was confirmed on multiple system "
                        f"visits: {system_visits}."
                    ),
                    details={
                        "physicalVisitId": physical_visit_id,
                        "systemVisitIds": system_visits,
                        "cameraTracks": tracks,
                    },
                )
            )

    by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for visit in physical_visits:
        by_subject[str(visit["subjectId"])].append(visit)
    for subject_id, visits in by_subject.items():
        mapped = [
            mappings.get(str(visit["physicalVisitId"]))
            for visit in sorted(visits, key=lambda item: int(item["ordinal"]))
        ]
        non_null = [visit_id for visit_id in mapped if visit_id is not None]
        if len(non_null) >= 2:
            status = "pass" if len(set(non_null)) == len(non_null) else "fail"
            results.append(
                AnalysisResult(
                    rule_id=f"reentry:{subject_id}",
                    status=status,
                    summary=(
                        "Re-entries mapped to distinct visits."
                        if status == "pass"
                        else f"Multiple physical visits reused system visits: {mapped}."
                    ),
                    details={"subjectId": subject_id, "systemVisitIds": mapped},
                )
            )
    return results


def _customer_results(
    *,
    subjects: Sequence[Mapping[str, Any]],
    physical_visits: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    mappings: Mapping[str, int | None],
) -> list[AnalysisResult]:
    expected_by_subject = {
        str(subject["subjectId"]): subject.get("expectedCustomerId")
        for subject in subjects
    }
    observed_by_physical_visit: dict[str, list[str]] = defaultdict(list)
    for annotation in annotations:
        reference = annotation.get("observationRef")
        physical_visit_id = annotation.get("physicalVisitId")
        if (
            annotation.get("annotationType") == "observation_is_subject"
            and physical_visit_id
            and isinstance(reference, Mapping)
            and reference.get("observedCustomerId") is not None
        ):
            observed_by_physical_visit[str(physical_visit_id)].append(
                str(reference["observedCustomerId"])
            )

    results: list[AnalysisResult] = []
    for visit in physical_visits:
        subject_id = str(visit["subjectId"])
        expected = expected_by_subject.get(subject_id)
        if expected is None:
            continue
        physical_visit_id = str(visit["physicalVisitId"])
        observed = observed_by_physical_visit.get(physical_visit_id, [])
        if not observed:
            results.append(
                AnalysisResult(
                    rule_id=f"customer:{physical_visit_id}",
                    status="pending",
                    summary=(
                        f"Expected customer {expected}, but no confirmed observation "
                        "has a customer assignment."
                    ),
                    details={
                        "expectedCustomerId": expected,
                        "mappedSystemVisitId": mappings.get(physical_visit_id),
                    },
                )
            )
            continue
        wrong = sorted({value for value in observed if value != str(expected)})
        results.append(
            AnalysisResult(
                rule_id=f"customer:{physical_visit_id}",
                status="fail" if wrong else "pass",
                summary=(
                    f"Customer assignment matches {expected}."
                    if not wrong
                    else f"Expected customer {expected}; observed {wrong}."
                ),
                details={
                    "expectedCustomerId": expected,
                    "observedCustomerIds": sorted(set(observed)),
                },
            )
        )
    return results


def _nearest_event(
    events: Sequence[Mapping[str, Any]],
    *,
    event_type: str,
    expected_time: int,
    before_milliseconds: int,
    after_milliseconds: int,
    visit_id: int | None = None,
    shelf_id: int | None = None,
    excluded_event_ids: set[int] | None = None,
) -> Mapping[str, Any] | None:
    candidates = []
    excluded = excluded_event_ids or set()
    for event in events:
        if event.get("eventType") != event_type:
            continue
        if int(event["eventId"]) in excluded:
            continue
        if visit_id is not None and event.get("visitId") != visit_id:
            continue
        payload = event.get("payload", {})
        event_shelf_id = payload.get("shelfId") if isinstance(payload, Mapping) else None
        if shelf_id is not None and event_shelf_id != shelf_id:
            continue
        delta = int(event["occurredAtUnixMilliseconds"]) - expected_time
        if -before_milliseconds <= delta <= after_milliseconds:
            candidates.append((abs(delta), event))
    return None if not candidates else min(candidates, key=lambda item: item[0])[1]


def _missing_summary(annotation_type: str, shelf_id: Any) -> str:
    if annotation_type == "physical_entry":
        return "No accepted ENTRY event matched the physical entry."
    if annotation_type == "physical_leave":
        return "No accepted LEAVE event matched the physical leave."
    return (
        f"No matching {annotation_type.replace('_', ' ')} event "
        f"was observed for shelf {shelf_id}."
    )


def _matched_summary(
    annotation_type: str,
    *,
    matched_visit_id: Any,
    shelf_id: Any,
) -> str:
    if annotation_type == "physical_entry":
        return f"Physical entry matched system visit {matched_visit_id}."
    if annotation_type == "physical_leave":
        return f"Physical leave matched system visit {matched_visit_id}."
    return (
        f"{annotation_type.replace('_', ' ').title()} matched shelf "
        f"{shelf_id} for visit {matched_visit_id}."
    )
