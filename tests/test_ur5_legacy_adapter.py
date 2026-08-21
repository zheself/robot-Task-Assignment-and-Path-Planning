import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from safe_residual_rl.data import load_ur5_static_csvs


class UR5LegacyAdapterTest(unittest.TestCase):
    def _write(self, path, header, row):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerow(row)

    def test_converts_once_and_excludes_known_risks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = [f"a{i}" for i in range(1, 7)] + ["x", "y", "z", "x-real", "y-real", "z-real"]
            row = [180, 0, 0, 0, 0, 0, 1000, 0, 0, 1001, 2, 3]
            self._write(root / "20250101" / "usable.csv", header, row)
            self._write(root / "data_all.csv", header, row)
            self._write(root / "data08.csv", header, row)
            result = load_ur5_static_csvs(root, joint_unit="degree", length_unit="mm")
            self.assertEqual(result.audit["total_rows"], 1)
            self.assertAlmostEqual(result.dataset.q_rad[0, 0], np.pi)
            self.assertAlmostEqual(result.dataset.x_nominal_m[0, 0], 1.0)
            reasons = {item["reason"] for item in result.audit["excluded_files"]}
            self.assertEqual(reasons, {"merged_duplicate_risk", "real_xyz_header_order_suspect"})

    def test_units_are_mandatory_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_ur5_static_csvs(Path(directory), joint_unit="guess", length_unit="mm")


if __name__ == "__main__":
    unittest.main()
