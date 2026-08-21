import unittest

from safe_residual_rl.envs.calibrated_suite import RealCalibratedSuite


class RLDomainContractTest(unittest.TestCase):
    def test_declared_prior_sets_are_disjoint(self):
        self.assertNotIn(RealCalibratedSuite.UNSEEN_TEST_PRIOR, RealCalibratedSuite.TRAIN_PRIORS)


if __name__ == "__main__":
    unittest.main()
