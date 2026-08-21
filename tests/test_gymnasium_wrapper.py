import unittest
import numpy as np

try:
    from gymnasium.utils.env_checker import check_env
    from safe_residual_rl.envs.gymnasium_env import GymnasiumResidualEnv, normalize_observation
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

from safe_residual_rl.data import SyntheticErrorField, generate_reference_path
from safe_residual_rl.envs import ResidualTrajectoryEnv, Trajectory
from safe_residual_rl.kinematics import UR5Kinematics
from safe_residual_rl.safety import SafetyProjector


@unittest.skipUnless(GYM_AVAILABLE, "gymnasium is an optional runtime dependency")
class GymnasiumWrapperTest(unittest.TestCase):
    def test_checker_and_normalized_action_scaling(self):
        kinematics = UR5Kinematics()
        q, x = generate_reference_path(kinematics, 77, length=16)
        core = ResidualTrajectoryEnv(
            kinematics, Trajectory(q, x, "gym_test"), SyntheticErrorField(8), SafetyProjector(kinematics)
        )
        env = GymnasiumResidualEnv(core)
        check_env(env, warn=True, skip_render_check=True)
        env.reset(seed=4)
        _, _, _, _, info = env.step(np.array([0.5, 0.0, 0.0], dtype=np.float32))
        self.assertAlmostEqual(info["requested_action_local_m"][0], 0.5 * core.action_bound_m)

    def test_error_is_presented_in_millimetres_to_policy(self):
        kinematics = UR5Kinematics(); q, x = generate_reference_path(kinematics, 88, length=16)
        core = ResidualTrajectoryEnv(
            kinematics, Trajectory(q, x, "scale_test"), SyntheticErrorField(8), SafetyProjector(kinematics),
            history_length=4,
        )
        observation, _ = core.reset(seed=2)
        layout = core.observation_layout
        observation[layout["error_history_m"]][-3:] = [0.001, -0.002, 0.0005]
        scaled = normalize_observation(observation, core.action_bound_m, layout)
        np.testing.assert_allclose(scaled[layout["error_history_m"]][-3:], [1.0, -2.0, 0.5])


if __name__ == "__main__":
    unittest.main()
