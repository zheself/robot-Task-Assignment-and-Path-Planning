import unittest
import numpy as np

from safe_residual_rl.data import RealSupportTrajectoryGenerator
from safe_residual_rl.data.synthetic import MeasurementDataset
from safe_residual_rl.kinematics import UR5Kinematics
from safe_residual_rl.models import RidgeErrorPrior, fit_calibration_profile


class CalibrationAndTrajectoryTest(unittest.TestCase):
    def test_calibration_centers_each_session_without_noise_claim(self):
        rng = np.random.default_rng(2); q = rng.normal(size=(20, 6)) * 0.1
        kin = UR5Kinematics(); x = kin.position_batch(q); session = np.array(["a"] * 10 + ["b"] * 10)
        error = np.tile([0.001, 0, 0], (20, 1)); error[10:] += [0, 0.001, 0]
        dataset = MeasurementDataset(q, x, x + error, session, session, np.array(["A"] * 20), "REAL_STATIC")
        prior = RidgeErrorPrior().fit(q, x, error)
        profile = fit_calibration_profile("ridge", prior, dataset, np.ones(20, dtype=bool))
        self.assertIn("unexplained_residual_not_measurement_noise", profile.summary["terminology"].values())
        self.assertEqual(profile.session_shift_proxy_m.shape, (2, 3))

    def test_trajectory_generators_are_smooth_and_holdout_is_ood(self):
        rng = np.random.default_rng(5); center = np.array([0.1, -1.2, 1.3, -1.1, -1.3, 0.2])
        q = center + rng.normal(0, 0.08, size=(100, 6)); kin = UR5Kinematics(); x = kin.position_batch(q)
        generator = RealSupportTrajectoryGenerator(kin).fit(q, x)
        sine = generator.generate("sine", 1, 50); smooth = generator.generate("smooth_random", 2, 50)
        holdout = generator.generate("workspace_holdout", 3, 50)
        self.assertEqual(sine.q_ref_rad.shape, (50, 6)); self.assertEqual(smooth.x_ref_m.shape, (50, 3))
        self.assertLess(np.max(np.abs(np.diff(smooth.q_ref_rad, axis=0))), 0.2)
        self.assertGreater(holdout.ood["outside_training_joint_bounds_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
