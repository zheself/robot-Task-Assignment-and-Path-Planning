"""Factories for separated SIM_CALIBRATED train/validation/frozen-test domains."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from safe_residual_rl.data import (
    RealSupportTrajectoryGenerator, candidate_split_document, load_ur5_static_csvs, masks_from_candidate_split,
)
from safe_residual_rl.kinematics import UR5Kinematics
from safe_residual_rl.models import SklearnStaticPrior, fit_calibration_profile, fit_prior, make_static_priors
from safe_residual_rl.safety import SafetyProjector

from .core import ResidualTrajectoryEnv, Trajectory


class RealCalibratedSuite:
    TRAIN_PRIORS = ("ridge", "extra_trees")
    UNSEEN_TEST_PRIOR = "rbf_kernel"

    def __init__(self, real_root: Path, tree_estimators: int = 40) -> None:
        self.loaded = load_ur5_static_csvs(real_root, joint_unit="degree", length_unit="mm")
        self.dataset = self.loaded.dataset
        self.candidate_split = candidate_split_document()
        self.masks = masks_from_candidate_split(self.dataset, self.candidate_split)
        self.kinematics = UR5Kinematics()
        self.projector = SafetyProjector(self.kinematics)
        suite = make_static_priors(seed=2026)
        suite["extra_trees"] = SklearnStaticPrior(
            ExtraTreesRegressor(
                n_estimators=tree_estimators, min_samples_leaf=2, max_features=1.0,
                random_state=2026, n_jobs=1,
            )
        )
        self.models = {}
        self.profiles = {}
        train = self.masks["train"]
        for name in (*self.TRAIN_PRIORS, self.UNSEEN_TEST_PRIOR):
            model = suite[name]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_prior(model, self.dataset.q_rad[train], self.dataset.x_nominal_m[train], self.dataset.error_m[train])
            self.models[name] = model
            self.profiles[name] = fit_calibration_profile(name, model, self.dataset, train)
        self.generator = RealSupportTrajectoryGenerator(self.kinematics).fit(
            self.dataset.q_rad[train], self.dataset.x_nominal_m[train]
        )
        self.tree_estimators = tree_estimators

    def core_factory(self, role: str, scenario: str = "nominal"):
        if role not in ("train", "validation", "test"):
            raise ValueError(role)
        if role == "test":
            prior_names = (self.UNSEEN_TEST_PRIOR,)
            seed_offset = 300_000
        elif role == "validation":
            prior_names = self.TRAIN_PRIORS
            seed_offset = 200_000
        else:
            prior_names = self.TRAIN_PRIORS
            seed_offset = 100_000

        def factory(seed: int) -> ResidualTrajectoryEnv:
            rng = np.random.default_rng(seed_offset + seed)
            prior_name = prior_names[seed % len(prior_names)]
            if scenario == "workspace_holdout":
                kind = "workspace_holdout"
            else:
                kind = "sine" if seed % 2 == 0 else "smooth_random"
            generated = self.generator.generate(kind, seed_offset + seed, length=64)
            error_field = self.profiles[prior_name].sample_episode_error(
                self.models[prior_name], seed=seed_offset + 50_000 + seed
            )
            if role == "train":
                delay = int(rng.integers(0, 3))
            elif role == "validation":
                delay = 1 + int(seed % 2)
            else:
                delay = 2
            return ResidualTrajectoryEnv(
                self.kinematics,
                Trajectory(generated.q_ref_rad, generated.x_ref_m, f"{role}_{scenario}_{seed}"),
                error_field,
                self.projector,
                base_prior=self.models["ridge"].predict,
                noise_std_m=0.0,
                action_delay_steps=delay,
                residual_action_bound_m=0.002,
                history_length=4,
                seed=seed,
            )
        factory.role = role
        factory.prior_names = prior_names
        factory.scenario = scenario
        return factory
