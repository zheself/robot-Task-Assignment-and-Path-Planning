import unittest
import numpy as np
from safe_residual_rl.data import SyntheticErrorField, generate_measurement_dataset, generate_reference_path
from safe_residual_rl.envs import ResidualTrajectoryEnv, Trajectory
from safe_residual_rl.kinematics import UR5Kinematics
from safe_residual_rl.models import RidgeErrorPrior
from safe_residual_rl.safety import SafetyProjector


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = UR5Kinematics()
        cls.error_field = SyntheticErrorField(seed=4)
        cls.projector = SafetyProjector(cls.kinematics)

    def test_prior_improves_held_out_synthetic_measurements(self):
        train = generate_measurement_dataset(self.kinematics, self.error_field, 1, "train", "A", 4, 40)
        test = generate_measurement_dataset(self.kinematics, self.error_field, 2, "test", "B", 2, 40)
        self.assertFalse(set(train.path_id) & set(test.path_id))
        prior = RidgeErrorPrior().fit(train.q_rad, train.x_nominal_m, train.error_m)
        prediction = prior.predict(test.q_rad, test.x_nominal_m)
        prior_error = np.mean(np.linalg.norm(prediction - test.error_m, axis=1))
        zero_error = np.mean(np.linalg.norm(test.error_m, axis=1))
        self.assertLess(prior_error, 0.55 * zero_error)

    def test_safety_projector_enforces_bounds(self):
        q = np.array([0.1, -1.2, 1.3, -1.1, -1.4, 0.2])
        result = self.projector.project_world(q, np.array([0.2, -0.1, 0.15]))
        self.assertTrue(result.clipped)
        self.assertIn("cartesian_step", result.reasons)
        self.assertLessEqual(np.max(np.abs(result.delta_q_rad)), self.projector.max_joint_step_rad + 1e-12)
        self.assertLessEqual(np.linalg.norm(result.executed_world_m), self.projector.max_cartesian_step_m + 1e-3)

    def _make_env(self):
        q_ref, x_ref = generate_reference_path(self.kinematics, 8, 20)
        return ResidualTrajectoryEnv(self.kinematics, Trajectory(q_ref, x_ref, "test_path"),
                                     self.error_field, self.projector, noise_std_m=0.0001)

    def test_environment_seed_is_reproducible(self):
        first, second = self._make_env(), self._make_env()
        observation_a, info_a = first.reset(seed=99)
        observation_b, info_b = second.reset(seed=99)
        np.testing.assert_allclose(observation_a, observation_b)
        np.testing.assert_allclose(info_a["error_m"], info_b["error_m"])
        for _ in range(4):
            action = np.array([0.0002, -0.0001, 0.00005])
            result_a, result_b = first.step(action), second.step(action)
            np.testing.assert_allclose(result_a[0], result_b[0])
            self.assertEqual(result_a[1], result_b[1])

    def test_residual_bound_is_separate_and_reported(self):
        env = self._make_env()
        env.reset(seed=1)
        _, _, _, _, info = env.step(np.array([0.1, 0.0, 0.0]))
        self.assertLess(env.action_bound_m, self.projector.max_cartesian_step_m)
        self.assertTrue(info["safety_clipped"])
        self.assertIn("residual_action_bound", info["safety_reasons"])


if __name__ == "__main__":
    unittest.main()
