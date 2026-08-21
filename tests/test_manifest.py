import json
from pathlib import Path
import tempfile
import unittest

from safe_residual_rl.data.manifest import SplitEntry, SplitManifest, load_manifest, synthetic_manifest


class ManifestTest(unittest.TestCase):
    def test_round_trip_and_hash(self):
        manifest = synthetic_manifest({"train": 2, "validation": 1, "test": 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest.write_immutable(path)
            loaded = load_manifest(path)
            self.assertEqual(manifest.sha256, loaded.sha256)
            manifest.write_immutable(path)

    def test_tampering_is_detected(self):
        manifest = synthetic_manifest({"train": 1, "validation": 1, "test": 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest.write_immutable(path)
            document = json.loads(path.read_text())
            document["robot_id"] = "tampered"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_manifest(path)

    def test_session_leakage_is_rejected(self):
        entries = (
            SplitEntry("g1", "same", "p1", "A", "train", "x"),
            SplitEntry("g2", "same", "p2", "A", "validation", "x"),
            SplitEntry("g3", "s3", "p3", "B", "test", "x"),
        )
        manifest = SplitManifest("bad", "ur5", "SYNTHETIC", entries)
        with self.assertRaisesRegex(ValueError, "session leakage"):
            manifest.validate()


if __name__ == "__main__":
    unittest.main()
