import unittest
import numpy as np

from safe_residual_rl.data import SyntheticErrorField, generate_reference_path
from safe_residual_rl.envs import ResidualTrajectoryEnv, Trajectory
from safe_residual_rl.kinematics import UR5Kinematics
from safe_residual_rl.safety import SafetyProjector


class SequenceObservationTest(unittest.TestCase):
    def make_env(self, delay=2, history=4):
        kin = UR5Kinematics(); q, x = generate_reference_path(kin, 19, length=20)
        return ResidualTrajectoryEnv(
            kin, Trajectory(q, x, "history"), SyntheticErrorField(7), SafetyProjector(kin),
            action_delay_steps=delay, history_length=history, noise_std_m=0.0,
        )

    def test_history_mask_grows_without_leaking_future(self):
        env = self.make_env(); obs, _ = env.reset(seed=1); layout = env.observation_layout
        np.testing.assert_array_equal(obs[layout["history_validity"]], [0, 0, 0, 1])
        env.step(np.array([0.0002, 0, 0])); obs = env.observation
        np.testing.assert_array_equal(obs[layout["history_validity"]], [0, 0, 1, 1])
        actions = obs[layout["action_history_m"]].reshape(4, 3)
        np.testing.assert_allclose(actions[-1], 0.0)  # two-step delay: requested action is not yet applied

    def test_delay_state_is_explicit_one_hot(self):
        env = self.make_env(delay=3); obs, _ = env.reset(seed=1)
        np.testing.assert_array_equal(obs[env.observation_layout["delay_one_hot"]], [0, 0, 0, 1])

    def test_seeded_history_rollout_is_deterministic(self):
        first, second = self.make_env(), self.make_env()
        for env in (first, second): env.reset(seed=17)
        for _ in range(5):
            a = np.array([0.0001, -0.0001, 0.0])
            result_a, result_b = first.step(a), second.step(a)
            np.testing.assert_allclose(result_a[0], result_b[0])


if __name__ == "__main__":
    unittest.main()
