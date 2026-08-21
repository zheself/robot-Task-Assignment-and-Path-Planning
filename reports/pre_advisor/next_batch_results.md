# Next-batch implementation result

Date: 2026-08-03  
Evidence labels: `SYNTHETIC` and `REAL_STATIC_UNVERIFIED_METADATA`

## Completed engineering gates

- Canonical immutable split manifest with SHA-256, overwrite refusal, tamper detection and session/path leakage tests.
- Separate physical interfaces: core residual action in metres; Gymnasium policy action in normalized `[-1,1]^3`.
- Residual action hard bound of 2 mm, separate from the total Cartesian projection bound of 6 mm.
- Hidden plant perturbations for DH link parameters, joint-zero offsets and TCP offsets; hidden values do not enter observations.
- Mean-bias, projected supervised, fixed-feedback and repeated-path ILC baselines.
- Gymnasium `check_env`, CPU SAC and TD3 train/save/evaluate pipeline.
- Strict UR5 legacy CSV adapter, draft data card and read-only audit.
- Six-scenario synthetic comparison table and figure generation.

## SB3 smoke result

Training used 3000 steps per algorithm, one training seed per algorithm and three held-out evaluation seeds. This is an integration smoke test, not a paper comparison.

| Method | RMSE (mm) | P95 (mm) | Action TV (mm) | Safety clip rate |
|---|---:|---:|---:|---:|
| projected supervised prior | **1.126** | **1.484** | **0.000** | 0.000 |
| SAC | 1.259 | 1.671 | 7.209 | 0.000 |
| TD3 | 1.269 | 1.865 | 15.252 | 0.000 |

Conclusion: SAC/TD3 integration works and residual bounding eliminated unsafe large exploration, but neither short-run policy beats the strong supervised prior. This result must be retained. Before longer RL training, the next method work should focus on history/observability, explicit delay state, base-policy advantage shaping and multi-seed learning curves—not merely more timesteps.

## Expanded non-RL/smoke matrix

Six held-out conditions are now executable: unseen path, cross-date bias, workspace holdout, noise plus delay, kinematic perturbation, and combined shift. ILC is evaluated under a separate repeated-execution protocol; it must not be described as zero-shot generalization.

The machine-readable run is under ignored `outputs/pre_advisor_smoke/summary.json`; compact CSV and figure are generated under `reports/pre_advisor/`.

## UR5 static-data audit

With runtime assumptions `degree` and `mm`, the adapter accepted 1340 rows from 14 files. The measured position-error norm has mean 2.788 mm, median 2.720 mm, P95 4.684 mm and maximum 5.447 mm. It automatically excludes:

- `data_all.csv`: merged/component duplicate risk;
- `data08.csv`: declared measured XYZ column order is suspect.

No real split has been frozen. Dates/frames/TCP/device accuracy and file semantics remain unverified, and the dataset remains explicitly `not_an_offline_rl_dataset`.
