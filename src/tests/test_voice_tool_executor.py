import unittest

from voice_agent.event_bridge import VoiceEventBridge
from voice_agent.tool_executor import (
    VoiceToolError,
    VoiceToolExecutor,
    realtime_tool_definitions,
)


class FakeOperatorClient:
    def __init__(self) -> None:
        self.annotations = []
        self.requested_observation_indexes = []
        self.requested_camera_state_indexes = []
        self.context = {
            "subjects": [{"subjectId": "milan"}],
            "events": [
                {
                    "eventId": 11,
                    "eventType": "entry_accepted",
                    "visitId": 4,
                    "payload": {},
                }
            ],
            "verdicts": {},
        }

    def state(self):
        return {
            "activeRun": {"runId": "run-1"},
            "shelves": [{"shelfId": 3}],
            "cameras": [
                {"id": 0, "deviceId": "camera-a", "status": "active"},
                {"id": 4, "deviceId": "camera-e", "status": "active"},
            ],
            "visits": [{"visitId": 4, "lastCameraIndex": 4}],
        }

    def voice_context(self, run_id):
        return self.context

    def create_annotation(self, run_id, payload):
        annotation = {"annotationId": len(self.annotations) + 1, **payload}
        self.annotations.append(annotation)
        return {"annotation": annotation}

    def observation(self, camera_index):
        self.requested_observation_indexes.append(camera_index)
        return {
            "camera": {
                "id": camera_index,
                "deviceId": "camera-a",
            },
            "frame": {
                "rgbSequenceNumber": 123,
                "hostSyncedSeconds": 4.5,
            },
            "observations": [
                {
                    "trackId": 9,
                    "cameraIndex": camera_index,
                    "visitId": 4,
                    "customerId": "customer-4",
                }
            ],
        }

    def world_state(self):
        return {"revision": 7, "visits": []}

    def subject_world_state(self, run_id, subject_id):
        return {
            "revision": 7,
            "processInstanceId": "process-1",
            "worldStateRef": {
                "snapshotId": "snapshot-1",
                "revision": 7,
                "processInstanceId": "process-1",
                "queriedAtUnixMilliseconds": 100,
            },
            "resolution": {"status": "confirmed", "visitId": 4},
            "claims": {
                "visitId": 4,
                "shelfPositionId": 3,
                "shelfPositionDistanceMm": 1250.0,
                "shelfPositionFreshness": "current",
                "inside": True,
                "visibleOnCameraIndexes": [0, 4],
                "productId": "001",
                "productLabel": "cola",
            },
            "visit": {
                "visitId": 4,
                "faceIdentityIds": [f"face_person_{index:03d}" for index in range(500)],
                "currentTracks": [{"trackId": 9, "cameraIndex": 0}],
                "shelfCandidates": [
                    {
                        "shelfId": 3,
                        "cameraIndex": 0,
                        "trackId": 9,
                        "distanceMm": 1250.0,
                        "personDepthMm": 4100.0,
                        "personDepthRoi": [10, 20, 30, 40],
                    }
                ],
            },
        }

    def visit_world_state(self, visit_id):
        return {"visit": {"visitId": visit_id}}

    def shelf_world_state(self, shelf_id):
        return {"shelf": {"shelfId": shelf_id}}

    def product_world_state(self, visit_id):
        return {
            "visitId": visit_id,
            "freshness": "current",
            "productRecognition": {
                "bestCandidate": {"productId": "001", "label": "cola"}
            },
        }

    def camera_world_state(self, camera_index):
        self.requested_camera_state_indexes.append(camera_index)
        return {
            "camera": {"cameraIndex": camera_index},
            "tracks": [{"trackId": 9, "sourceCameraIndex": camera_index}],
        }


class VoiceToolExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeOperatorClient()
        self.bridge = VoiceEventBridge()
        self.bridge.refresh(self.client.context)
        self.bridge.next_event()
        self.executor = VoiceToolExecutor(
            self.client,
            self.bridge,
            expected_run_id="run-1",
        )

    def test_confirms_existing_event_through_annotation_api(self) -> None:
        result = self.executor.execute("confirm_system_event", {"event_id": 11})
        self.assertTrue(result.payload["ok"])
        self.assertEqual(self.client.annotations[0]["annotationType"], "system_event_correct")
        self.assertIsNone(self.bridge.current_event())

    def test_legacy_shelf_event_tools_are_not_exposed(self) -> None:
        tools = {item["name"] for item in realtime_tool_definitions()}
        self.assertNotIn("report_missing_shelf_approach", tools)
        self.assertNotIn("report_missing_shelf_leave", tools)

    def test_physical_shelf_fact_preserves_depth_and_observation_evidence(self) -> None:
        result = self.executor.execute(
            "record_physical_subject_state",
            {
                "claim": "shelfPositionId",
                "physical_value": 3,
                "shelf_id": 3,
            },
        )

        annotation = result.payload["annotation"]
        self.assertEqual(annotation["shelfId"], 3)
        self.assertEqual(annotation["observationRef"]["rgbSequenceNumber"], 123)
        self.assertEqual(
            annotation["systemShelfDiagnostic"]["personDepthRoi"],
            [10, 20, 30, 40],
        )

    def test_physical_shelf_fact_schema_accepts_optional_shelf_id(self) -> None:
        definition = {
            item["name"]: item for item in realtime_tool_definitions()
        }["record_physical_subject_state"]

        properties = definition["parameters"]["properties"]
        self.assertEqual(properties["shelf_id"]["minimum"], 1)

    def test_rejects_unknown_shelf_and_changed_run(self) -> None:
        with self.assertRaises(VoiceToolError) as shelf_error:
            self.executor.execute(
                "record_physical_subject_state",
                {"claim": "shelf", "physical_value": 99, "shelf_id": 99},
            )
        self.assertEqual(shelf_error.exception.code, "unknown_shelf")
        self.client.state = lambda: {"activeRun": {"runId": "run-2"}, "shelves": []}
        with self.assertRaises(VoiceToolError) as run_error:
            self.executor.execute("report_missing_entry", {})
        self.assertEqual(run_error.exception.code, "run_changed")

    def test_queries_and_confirms_revision_bound_subject_claim(self) -> None:
        result = self.executor.execute(
            "get_subject_state",
            {"claim": "shelfPositionId"},
        )
        self.assertEqual(result.payload["selectedClaim"]["systemValue"], 3)

        confirmed = self.executor.execute("confirm_last_state_claim", {})
        self.assertTrue(confirmed.payload["ok"])
        annotation = self.client.annotations[-1]
        self.assertEqual(annotation["annotationType"], "world_state_claim_correct")
        self.assertEqual(annotation["worldStateRef"]["snapshotId"], "snapshot-1")
        self.assertEqual(annotation["systemValue"], 3)

    def test_product_query_resolves_current_subject_visit(self) -> None:
        result = self.executor.execute("get_product_state", {})

        product = result.payload["productState"]["productRecognition"]
        self.assertEqual(product["bestCandidate"]["productId"], "001")
        definitions = {item["name"] for item in realtime_tool_definitions()}
        self.assertIn("get_product_state", definitions)

    def test_subject_state_payload_omits_unbounded_face_and_track_history(self) -> None:
        result = self.executor.execute(
            "get_subject_state",
            {"claim": "visitId"},
        )

        visit = result.payload["subjectState"]["visit"]
        self.assertEqual(visit["visitId"], 4)
        self.assertNotIn("faceIdentityIds", visit)
        self.assertNotIn("currentTracks", visit)
        self.assertNotIn("shelfCandidates", visit)

    def test_physical_shelf_fact_atomically_confirms_pending_shelf_claim(self) -> None:
        self.executor.execute("get_subject_state", {"claim": "shelfPositionId"})

        result = self.executor.execute(
            "record_physical_subject_state",
            {"claim": "shelf", "physical_value": 3, "shelf_id": 3},
        )

        annotation = result.payload["annotation"]
        self.assertEqual(annotation["annotationType"], "world_state_claim_correct")
        self.assertEqual(annotation["claim"], "shelfPositionId")
        self.assertEqual(annotation["systemValue"], 3)
        self.assertNotIn("physicalValue", annotation)
        self.assertEqual(annotation["shelfId"], 3)
        with self.assertRaises(VoiceToolError) as consumed:
            self.executor.execute(
                "correct_last_state_claim",
                {"physical_value": 1},
            )
        self.assertEqual(consumed.exception.code, "no_state_claim")

    def test_physical_shelf_fact_rejects_conflicting_shelf_fields(self) -> None:
        with self.assertRaises(VoiceToolError) as mismatch:
            self.executor.execute(
                "record_physical_subject_state",
                {"claim": "shelf", "physical_value": 2, "shelf_id": 3},
            )
        self.assertEqual(mismatch.exception.code, "shelf_value_mismatch")

    def test_correction_requires_a_prior_specific_claim(self) -> None:
        with self.assertRaises(VoiceToolError) as missing:
            self.executor.execute(
                "correct_last_state_claim",
                {"physical_value": 1},
            )
        self.assertEqual(missing.exception.code, "no_state_claim")

        self.executor.execute("get_subject_state", {"claim": "shelfPositionId"})
        corrected = self.executor.execute(
            "correct_last_state_claim",
            {"physical_value": 1, "reason": "Standing at shelf one"},
        )
        self.assertEqual(
            corrected.payload["annotation"]["annotationType"],
            "world_state_claim_incorrect",
        )
        self.assertEqual(self.client.annotations[-1]["physicalValue"], 1)

    def test_shelf_claim_correction_captures_candidate_diagnostics(self) -> None:
        self.executor.execute("get_subject_state", {"claim": "shelfPositionId"})
        corrected = self.executor.execute(
            "correct_last_state_claim",
            {"physical_value": 3, "reason": "Physically at shelf three"},
        )

        annotation = corrected.payload["annotation"]
        self.assertEqual(annotation["shelfId"], 3)
        self.assertEqual(annotation["observationRef"]["trackId"], 9)
        self.assertEqual(
            annotation["systemShelfDiagnostic"]["personDepthMm"],
            4100.0,
        )

    def test_camera_tools_use_one_based_numbers_at_voice_boundary(self) -> None:
        result = self.executor.execute("get_camera_state", {"camera_number": 5})

        self.assertEqual(self.client.requested_camera_state_indexes, [4])
        self.assertEqual(result.payload["cameraState"]["camera"]["cameraNumber"], 5)
        self.assertEqual(
            result.payload["cameraState"]["tracks"][0]["sourceCameraNumber"],
            5,
        )
        self.assertNotIn("cameraIndex", result.payload["cameraState"]["camera"])

    def test_current_shop_state_translates_camera_numbers_both_directions(self) -> None:
        result = self.executor.execute(
            "get_current_shop_state",
            {"camera_number": 1},
        )

        self.assertEqual(self.client.requested_observation_indexes, [0])
        self.assertEqual(
            [camera["cameraNumber"] for camera in result.payload["cameras"]],
            [1, 5],
        )
        self.assertEqual(result.payload["visits"][0]["lastCameraNumber"], 5)
        self.assertEqual(
            result.payload["observation"]["observations"][0]["cameraNumber"],
            1,
        )

    def test_subject_visibility_is_presented_one_based_but_annotated_canonically(self) -> None:
        result = self.executor.execute(
            "get_subject_state",
            {"claim": "visibleOnCameraNumbers"},
        )

        self.assertEqual(result.payload["selectedClaim"]["systemValue"], [1, 5])
        claims = result.payload["subjectState"]["claims"]
        self.assertEqual(claims["visibleOnCameraNumbers"], [1, 5])
        self.assertNotIn("visibleOnCameraIndexes", claims)

        self.executor.execute("confirm_last_state_claim", {})
        annotation = self.client.annotations[-1]
        self.assertEqual(annotation["claim"], "visibleOnCameraIndexes")
        self.assertEqual(annotation["systemValue"], [0, 4])

    def test_rejects_zero_based_camera_number(self) -> None:
        with self.assertRaises(VoiceToolError) as error:
            self.executor.execute("get_camera_state", {"camera_number": 0})
        self.assertEqual(error.exception.code, "invalid_camera_number")

    def test_camera_tool_schemas_expose_only_one_based_camera_number(self) -> None:
        definitions = {item["name"]: item for item in realtime_tool_definitions()}
        for name in ("get_camera_state", "get_current_shop_state"):
            properties = definitions[name]["parameters"]["properties"]
            self.assertNotIn("camera_index", properties)
            self.assertEqual(properties["camera_number"]["minimum"], 1)


if __name__ == "__main__":
    unittest.main()
