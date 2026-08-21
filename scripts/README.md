# CLI roadmap

Implemented:

```text
audit_legacy_data.py          read-only preliminary CSV inventory
build_synthetic_manifest.py  immutable grouped split manifest + SHA-256
run_pre_advisor_smoke.py      synthetic data -> prior -> environment -> learning -> report
train_sb3_smoke.py            Gymnasium checker + CPU SAC/TD3 smoke
audit_ur5_legacy.py           strict read-only UR5 static CSV adapter/audit
plot_smoke_results.py         compact synthetic comparison CSV/figure
analyze_ur5_static.py         per-file REAL_STATIC quality, FK diagnostics, matched repeat case, candidate split
benchmark_real_static_priors.py grouped zero/mean/Ridge/tree/RBF/MLP prior benchmark
calibrate_real_simulator.py  unexplained residual/session-shift proxy and multi-prior diagnostic
train_sequence_rl.py         three-seed multi-prior SAC/TD3 with validation-only checkpoint selection
evaluate_frozen_rl.py        frozen unseen-RBF-prior test; no checkpoint selection
plot_real_calibrated_results.py compact REAL_STATIC/SIM_CALIBRATED figures and CSVs
```

Planned, in order:

```text
build_dataset.py              validated conversion to SI Parquet
build_splits.py               real-data grouped manifest
verify_kinematics.py          FK/Jacobian vs trusted poses
train_error_prior.py          B1--B3 and calibration report
run_env_smoke.py              expanded environment checker/scenario fixtures
evaluate_non_rl_baselines.py  standalone paper-facing baseline runner
train_residual_rl.py          multi-seed paper training after smoke
evaluate_scenarios.py         frozen multi-scenario protocol
```
