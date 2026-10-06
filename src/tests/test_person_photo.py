import threading
import time
import unittest
import tempfile
import os
import stat
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from pipeline.person_photo import PersonPhotoCapture, PersonPhotoError
from pipeline.tracking import Track
from pipeline.visit_identity import VisitAssignment


class PersonPhotoTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.photo_directory = Path(directory.name) / "photos"

    def capture(self, tracks, track_id=None, evidence=False):
        coordinator = PersonPhotoCapture(camera_device_ids=["camera"], photo_directory=self.photo_directory)
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=1) as executor:
            method = coordinator.capture_person_evidence if evidence else coordinator.capture_person
            future = executor.submit(method, 0, track_id)
            deadline = time.monotonic() + 2
            while not coordinator.has_pending_requests():
                if time.monotonic() > deadline:
                    self.fail("Request was not registered")
                threading.Event().wait(0.001)
            assignments = {4: VisitAssignment(7, 4, "camera", (), None)} if evidence else {}
            customers = {7: "customer-7"} if evidence else {}
            coordinator.observe_frame(
                camera_index=0, device_id="camera", rgb_sequence_number=99,
                observed_at_unix_milliseconds=1000, frame=frame, tracks=tracks,
                assignments=assignments, customer_ids_by_visit=customers,
            )
            customers[7] = "changed-after-frame"
            if assignments:
                assignments[4].visit_id = 99
            try:
                return future.result(timeout=1)
            finally:
                self.assertFalse(coordinator.has_pending_requests())

    def test_raw_person_crop_without_face_or_visit(self):
        jpeg, track, sequence = self.capture([Track(4, 100, 50, 400, 650, 0.9, status="TRACKED")])
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, (600, 300, 3))
        self.assertEqual((track, sequence), (4, 99))
        self.assertEqual(next(self.photo_directory.glob("*.jpg")).read_bytes(), jpeg)

    def test_saved_photo_matches_response_and_has_unique_absolute_path(self):
        tracks = [Track(4, 100, 50, 400, 650, 0.9, status="TRACKED")]
        jpeg, evidence = self.capture(tracks, evidence=True)
        path = Path(evidence["photoPath"])
        self.assertTrue(path.is_absolute())
        self.assertEqual(path.read_bytes(), jpeg)
        _, second = self.capture(tracks, evidence=True)
        self.assertNotEqual(second["photoPath"], str(path))
        self.assertEqual(list(self.photo_directory.glob("*.tmp")), [])

    def test_storage_failure_is_reported_and_request_is_cleaned_up(self):
        with patch.object(Path, "mkdir", side_effect=PermissionError("Denied")):
            with self.assertLogs("pipeline.person_photo", level="ERROR"):
                with self.assertRaises(PersonPhotoError) as raised:
                    self.capture([Track(4, 100, 50, 400, 650, 0.9, status="TRACKED")])
        self.assertEqual(raised.exception.code, "CAPTURE_STORAGE_FAILED")

    def test_photo_permissions_are_set_before_publishing(self):
        calls = []
        original_chmod = Path.chmod
        original_replace = Path.replace

        def chmod(path, mode):
            calls.append(("chmod", path.suffix, mode))
            return original_chmod(path, mode)

        def replace(path, target):
            calls.append(("replace", path.suffix))
            return original_replace(path, target)

        with patch.object(Path, "chmod", chmod), patch.object(Path, "replace", replace):
            self.capture([Track(4, 100, 50, 400, 650, 0.9, status="TRACKED")])
        self.assertEqual(calls, [
            ("chmod", "", 0o755), ("chmod", ".tmp", 0o666), ("replace", ".tmp"),
        ])

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits require Linux/Unix")
    def test_restrictive_umask_does_not_restrict_photo_permissions(self):
        self.photo_directory.mkdir(mode=0o700)
        previous = os.umask(0o077)
        try:
            _, evidence = self.capture(
                [Track(4, 100, 50, 400, 650, 0.9, status="TRACKED")], evidence=True,
            )
        finally:
            os.umask(previous)
        self.assertEqual(stat.S_IMODE(Path(evidence["photoPath"]).stat().st_mode), 0o666)
        self.assertEqual(stat.S_IMODE(self.photo_directory.stat().st_mode), 0o755)

    def test_permission_failure_does_not_publish_photo(self):
        with patch.object(Path, "chmod", side_effect=PermissionError("Denied")):
            with self.assertLogs("pipeline.person_photo", level="ERROR"):
                with self.assertRaises(PersonPhotoError) as raised:
                    self.capture([Track(4, 100, 50, 400, 650, 0.9, status="TRACKED")])
        self.assertEqual(raised.exception.code, "CAPTURE_STORAGE_FAILED")
        self.assertEqual(list(self.photo_directory.iterdir()), [])

    def test_multiple_people_require_selection(self):
        tracks = [Track(i, 100, 50, 400, 650, 0.9, status="TRACKED") for i in (4, 5)]
        with self.assertRaises(PersonPhotoError) as raised:
            self.capture(tracks)
        self.assertEqual(raised.exception.code, "MULTIPLE_PEOPLE")
        self.assertEqual(self.capture(tracks, 5)[1], 5)

    def test_evidence_is_copied_with_the_captured_frame(self):
        _, evidence = self.capture(
            [Track(4, 100, 50, 400, 650, 0.9, status="TRACKED")], evidence=True,
        )
        self.assertEqual(evidence["visitId"], 7)
        self.assertEqual(evidence["customerId"], "customer-7")
        self.assertEqual(evidence["rgbSequenceNumber"], 99)
        self.assertEqual(evidence["observedAtUnixMilliseconds"], 1000)
        self.assertEqual(evidence["cropBox"], [100, 50, 400, 650])
        self.assertEqual((evidence["width"], evidence["height"]), (300, 600))

    def test_unknown_identity_remains_null(self):
        _, evidence = self.capture(
            [Track(5, 100, 50, 400, 650, 0.9, status="TRACKED")], evidence=True,
        )
        self.assertIsNone(evidence["visitId"])
        self.assertIsNone(evidence["customerId"])

    def test_lost_person_is_not_captured(self):
        with self.assertRaises(PersonPhotoError) as raised:
            self.capture([Track(4, 100, 50, 400, 650, 0.9, status="LOST")])
        self.assertEqual(raised.exception.code, "PERSON_NOT_VISIBLE")
