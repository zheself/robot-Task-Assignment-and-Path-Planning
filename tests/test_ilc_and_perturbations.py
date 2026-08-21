import unittest
import numpy as np

from safe_residual_rl.algorithms.ilc import train_ilc_actions
from safe_residual_rl.data import SyntheticErrorField, generate_reference_path
from safe_residual_rl.envs import ResidualTrajectoryEnv, Trajectory
from safe_residual_rl.kinematics import UR5Kinematics
from safe_residual_rl.safety import SafetyProjector


class IlcAndPerturbationTest(unittest.TestCase):
    def setUp(self):
        self.nominal = UR5Kinematics()
        self.error = SyntheticErrorField(seed=31)
        self.projector = SafetyProjector(self.nominal)
        q, x = generate_reference_path(self.nominal, seed=41, length=24)
        self.trajectory = Trajectory(q, x, "repeat_path")

    def factory(self, seed):
        return ResidualTrajectoryEnv(
            self.nominal, self.trajectory, self.error, self.projector,
            noise_std_m=0.0, episode_drift_m=np.array([0.7, -0.4, 0.2]) * 1e-3,
        )

    def episode_rmse(self, actions):
        env = self.factory(3)
        env.reset(seed=3)
        errors = [env.current_error_m]
        for action in actions:
            _, _, done, _, info = env.step(action)
            errors.append(info["error_m"])
            if done:
                break
        return np.sqrt(np.mean(np.sum(np.asarray(errors) ** 2, axis=1)))

    def test_ilc_reduces_deterministic_repeat_error(self):
        zero = np.zeros((len(self.trajectory.q_ref_rad) - 1, 3))
        learned = train_ilc_actions(self.factory, seed=3, iterations=8)
        self.assertLess(self.episode_rmse(learned), 0.7 * self.episode_rmse(zero))

    def test_hidden_plant_perturbation_changes_output_not_observation_shape(self):
        perturbed_dh = self.nominal.dh.copy()
        perturbed_dh[1, 2] += 0.0002
        perturbed = UR5Kinematics(perturbed_dh)
        base = self.factory(5)
        shifted = ResidualTrajectoryEnv(
            self.nominal, self.trajectory, self.error, self.projector,
            plant_kinematics=perturbed, joint_zero_offset_rad=np.full(6, 0.0005),
            tcp_offset_m=np.array([0.0002, 0.0, -0.0001]), noise_std_m=0.0,
        )
        obs_base, info_base = base.reset(seed=5)
        obs_shifted, info_shifted = shifted.reset(seed=5)
        self.assertEqual(obs_base.shape, obs_shifted.shape)
        self.assertGreater(np.linalg.norm(info_base["error_m"] - info_shifted["error_m"]), 1e-5)


if __name__ == "__main__":
    unittest.main()
