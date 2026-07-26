import csv
import tempfile
import unittest
from pathlib import Path

from pipeline.visit_registry import (
    VISIT_ORIGIN_ENTRANCE,
    VISIT_STATUS_CLOSED,
    ShopVisit,
    VisitRegistry,
)
from replay_synced_rgbd_streams import ReplayArtifactWriter


class ReplayArtifactTests(unittest.TestCase):
    def test_final_visit_csv_writes_closed_visit_fields(self) -> None:
        registry = VisitRegistry()
        registry.visits[7] = ShopVisit(
            visit_id=7,
            origin=VISIT_ORIGIN_ENTRANCE,
            created_host_seconds=10.0,
            last_seen_host_seconds=20.0,
            last_device_id="camera-a",
            last_track_id=3,
            status=VISIT_STATUS_CLOSED,
            closed_host_seconds=21.0,
        )

        with tempfile.TemporaryDirectory() as temporary:
            writer = ReplayArtifactWriter(Path(temporary))
            try:
                writer.write_final_visits(registry)
            finally:
                writer.close()

            with (Path(temporary) / "final_visits.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["status"], VISIT_STATUS_CLOSED)
        self.assertEqual(rows[0]["closed_host_seconds"], "21.0")


if __name__ == "__main__":
    unittest.main()
