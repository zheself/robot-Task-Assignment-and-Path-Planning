import unittest
import numpy as np
from safe_residual_rl.kinematics import UR5Kinematics, path_frames


class KinematicsTest(unittest.TestCase):
    def setUp(self):
        self.kinematics = UR5Kinematics()
        self.q = np.array([0.1, -1.2, 1.3, -1.1, -1.4, 0.2])

    def test_forward_is_rigid_transform(self):
        transform = self.kinematics.forward(self.q)
        rotation = transform[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=10)
        np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])

    def test_jacobian_predicts_small_position_change(self):
        delta = np.array([1.0, -0.5, 0.25, 0.1, -0.2, 0.3]) * 1e-6
        predicted = self.kinematics.position_jacobian(self.q) @ delta
        actual = self.kinematics.position(self.q + delta) - self.kinematics.position(self.q)
        np.testing.assert_allclose(predicted, actual, atol=2e-10)

    def test_path_frames_are_orthonormal(self):
        t = np.linspace(0.0, 1.0, 20)
        points = np.column_stack((t, 0.1 * t**2, 0.05 * np.sin(t)))
        frames, _ = path_frames(points)
        for frame in frames:
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-8)
            self.assertGreater(np.linalg.det(frame), 0.999999)


if __name__ == "__main__":
    unittest.main()
