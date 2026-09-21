from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from step3.check2_replay import EXPECTED_CROSS_COSINE, sha256_file, verify_bundle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "step3" / "results" / "run_20260824"


class Check2ReplayTests(unittest.TestCase):
    def test_saved_bundle_replays(self) -> None:
        report = verify_bundle(RESULTS)

        self.assertEqual(report["status"], "GEOMETRY_REPLAYED_CAUSAL_REPAIR_FAILED")
        self.assertEqual(report["tensor"]["rows"], 200)
        self.assertEqual(report["tensor"]["width"], 2560)
        self.assertAlmostEqual(
            report["tensor"]["cross_cosine"], EXPECTED_CROSS_COSINE, places=6
        )
        self.assertGreater(report["tensor"]["text_reconstruction_cosine"], 0.999999)
        self.assertGreater(report["tensor"]["vision_reconstruction_cosine"], 0.999999)

    def test_file_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"step3")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"step3").hexdigest())


if __name__ == "__main__":
    unittest.main()
