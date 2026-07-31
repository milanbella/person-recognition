import unittest

from pipeline.operator_analysis import analyze_test_run


class OperatorAnalysisTests(unittest.TestCase):
    def test_matches_lifecycle_and_track_split_to_one_visit(self) -> None:
        run = {"runId": "run-1", "scenario": "route"}
        subjects = [
            {
                "subjectId": "milan",
                "displayName": "Milan",
                "expectedCustomerId": "184",
            }
        ]
        physical_visits = [
            {
                "physicalVisitId": "physical-1",
                "subjectId": "milan",
                "ordinal": 1,
                "enteredAtUnixMilliseconds": 1000,
                "leftAtUnixMilliseconds": 5000,
            }
        ]
        annotations = [
            {
                "annotationId": 1,
                "annotationType": "physical_entry",
                "physicalVisitId": "physical-1",
                "receivedAtUnixMilliseconds": 1000,
                "payload": {},
                "observationRef": None,
            },
            {
                "annotationId": 2,
                "annotationType": "observation_is_subject",
                "physicalVisitId": "physical-1",
                "receivedAtUnixMilliseconds": 2000,
                "payload": {},
                "observationRef": {
                    "cameraIndex": 0,
                    "trackId": 3,
                    "observedVisitId": 4,
                    "observedCustomerId": "184",
                },
            },
            {
                "annotationId": 3,
                "annotationType": "observation_is_subject",
                "physicalVisitId": "physical-1",
                "receivedAtUnixMilliseconds": 3000,
                "payload": {},
                "observationRef": {
                    "cameraIndex": 1,
                    "trackId": 8,
                    "observedVisitId": 4,
                    "observedCustomerId": "184",
                },
            },
            {
                "annotationId": 4,
                "annotationType": "physical_leave",
                "physicalVisitId": "physical-1",
                "receivedAtUnixMilliseconds": 5000,
                "payload": {},
                "observationRef": None,
            },
        ]
        events = [
            {
                "eventId": 1,
                "eventType": "entry_accepted",
                "occurredAtUnixMilliseconds": 1200,
                "visitId": 4,
                "payload": {},
            },
            {
                "eventId": 2,
                "eventType": "leave_accepted",
                "occurredAtUnixMilliseconds": 5300,
                "visitId": 4,
                "payload": {},
            },
        ]

        report, mappings = analyze_test_run(
            run=run,
            subjects=subjects,
            physical_visits=physical_visits,
            annotations=annotations,
            events=events,
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(mappings["physical-1"], 4)
        continuity = next(
            result
            for result in report["results"]
            if result["ruleId"] == "visit-continuity:physical-1"
        )
        self.assertEqual(continuity["status"], "pass")
        self.assertIn("2 confirmed camera tracks", continuity["summary"])

    def test_reports_missing_leave(self) -> None:
        report, _mappings = analyze_test_run(
            run={"runId": "run-1", "scenario": "route"},
            subjects=[],
            physical_visits=[
                {
                    "physicalVisitId": "physical-1",
                    "subjectId": "milan",
                    "ordinal": 1,
                    "enteredAtUnixMilliseconds": 1000,
                    "leftAtUnixMilliseconds": 5000,
                }
            ],
            annotations=[
                {
                    "annotationId": 1,
                    "annotationType": "physical_leave",
                    "physicalVisitId": "physical-1",
                    "receivedAtUnixMilliseconds": 5000,
                    "payload": {},
                    "observationRef": None,
                }
            ],
            events=[],
        )
        self.assertEqual(report["status"], "fail")
        self.assertIn("LEAVE", report["results"][0]["summary"])

    def test_operator_can_confirm_or_reject_generated_event(self) -> None:
        events = [
            {
                "eventId": 11,
                "eventType": "entry_accepted",
                "occurredAtUnixMilliseconds": 1200,
                "visitId": 4,
                "payload": {},
            },
            {
                "eventId": 12,
                "eventType": "leave_accepted",
                "occurredAtUnixMilliseconds": 2200,
                "visitId": 4,
                "payload": {},
            },
        ]
        annotations = [
            {
                "annotationId": 21,
                "annotationType": "system_event_correct",
                "receivedAtUnixMilliseconds": 1300,
                "payload": {"systemEventId": 11},
                "observationRef": None,
            },
            {
                "annotationId": 22,
                "annotationType": "system_event_incorrect",
                "receivedAtUnixMilliseconds": 2300,
                "payload": {"systemEventId": 12},
                "observationRef": None,
            },
        ]

        report, _mappings = analyze_test_run(
            run={"runId": "run-1", "scenario": "route"},
            subjects=[],
            physical_visits=[],
            annotations=annotations,
            events=events,
        )

        by_rule = {result["ruleId"]: result for result in report["results"]}
        self.assertEqual(by_rule["event-verdict:11"]["status"], "pass")
        self.assertEqual(by_rule["event-verdict:12"]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
