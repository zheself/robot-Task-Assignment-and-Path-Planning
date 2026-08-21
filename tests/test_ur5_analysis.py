import unittest
import hashlib
import json
import numpy as np

from safe_residual_rl.data.synthetic import MeasurementDataset
from safe_residual_rl.data.ur5_analysis import candidate_split_document, masks_from_candidate_split, match_cross_date_static_case


class UR5AnalysisTest(unittest.TestCase):
    def test_candidate_hash_scope_is_reproducible(self):
        document = candidate_split_document()
        recorded = document.pop("sha256")
        document.pop("hash_scope")
        payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.assertEqual(recorded, hashlib.sha256(payload.encode("utf-8")).hexdigest())

    def test_candidate_primary_roles_do_not_overlap(self):
        document = candidate_split_document()
        files = document["roles"]["train"] + document["roles"]["validation"] + document["roles"]["test_cross_date"]
        n = len(files)
        dataset = MeasurementDataset(
            q_rad=np.zeros((n, 6)), x_nominal_m=np.zeros((n, 3)), x_measured_m=np.zeros((n, 3)),
            session_id=np.array([f"ur5::{name}" for name in files]), path_id=np.array(files),
            date_id=np.array(["unverified"] * n), evidence_level="REAL_STATIC_UNVERIFIED_METADATA",
        )
        masks = masks_from_candidate_split(dataset, document)
        self.assertEqual(sum(int(masks[role].sum()) for role in ("train", "validation", "test_cross_date")), n)

    def test_static_matching_is_thresholded_not_row_zip(self):
        q_a = np.zeros((2, 6)); q_a[1, 0] = 0.1
        q_b = q_a[[1, 0]] + np.deg2rad(0.001)
        q = np.vstack((q_a, q_b))
        x_a = np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.3]])
        x_b = x_a[[1, 0]] + 1e-5
        x = np.vstack((x_a, x_b))
        dataset = MeasurementDataset(
            q, x, x + 0.001,
            np.array(["ur5::20250806/10.csv"] * 2 + ["ur5::20250807/10Pos.csv"] * 2),
            np.array(["a"] * 2 + ["b"] * 2), np.array(["A", "A", "B", "B"]),
            "REAL_STATIC_UNVERIFIED_METADATA",
        )
        result = match_cross_date_static_case(dataset)
        self.assertEqual(result["matched_pairs"], 2)
        self.assertEqual(result["pairs"][0]["second_index"], 1)


if __name__ == "__main__":
    unittest.main()
