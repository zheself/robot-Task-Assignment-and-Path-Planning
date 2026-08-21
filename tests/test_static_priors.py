import unittest
import numpy as np

from safe_residual_rl.evaluation.static_metrics import TrainingSupport, error_metrics
from safe_residual_rl.models import make_static_priors


class StaticPriorTest(unittest.TestCase):
    def test_all_priors_share_shape_interface(self):
        rng = np.random.default_rng(3)
        q = rng.normal(size=(36, 6)); x = rng.normal(size=(36, 3)) * 0.1
        error = np.column_stack((0.001 * np.sin(q[:, 0]), 0.002 * x[:, 1], 0.001 * q[:, 2]))
        for name, prior in make_static_priors(seed=2).items():
            prior.fit(q, x, error)
            self.assertEqual(prior.predict(q[:4], x[:4]).shape, (4, 3), name)
            self.assertEqual(prior.predict(q[0], x[0]).shape, (3,), name)

    def test_support_is_fit_only_from_given_training_data(self):
        q = np.zeros((4, 6)); x = np.zeros((4, 3)); q[:, 0] = [0, 0.1, 0.2, 0.3]
        support = TrainingSupport().fit(q, x)
        heldout_q = q[:1].copy(); heldout_q[0, 0] = 10.0
        result = support.describe(heldout_q, x[:1])
        self.assertEqual(result["labels"][0], "outside_axis_bounds")

    def test_vector_metrics(self):
        target = np.zeros((2, 3)); prediction = np.array([[0.001, 0, 0], [0, 0.002, 0]])
        metrics = error_metrics(target, prediction)
        self.assertAlmostEqual(metrics["mae_mm"], 1.5)
        self.assertAlmostEqual(metrics["rmse_mm"], np.sqrt(2.5))


if __name__ == "__main__":
    unittest.main()
