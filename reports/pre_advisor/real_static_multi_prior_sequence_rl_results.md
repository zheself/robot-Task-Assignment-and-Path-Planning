# REAL_STATIC calibration and multi-prior sequence-RL batch

Date: 2026-08-04  
Evidence labels: `REAL_STATIC`, `SIM_CALIBRATED`, `SIM_STRESS`  
Candidate split status: `candidate_unverified_not_frozen`  
Candidate split hash: `642442e03908a6351cf7f04e3b279af248c12a35725bf1807a56abc71813156f`

All source units (`degree`, `mm`), coordinate frames, TCP definitions and date semantics remain unverified. Static CSV rows are not treated as transitions, and generated paths are not real robot trajectories.

## 1. REAL_STATIC quality and matched repeat case

The strict adapter accepted 1340 rows from 14 files and continued to exclude `data_all.csv` and `data08.csv`. Whole files remain the minimum grouping unit.

The candidate roles are:

- train: root `data01...data07` plus `20250806/建模数据1.csv` — 749 rows;
- validation: `20250806/验证数据.csv` — 28 rows;
- cross-date test: `20250807/验证数据(1).csv` — 96 rows;
- reserved repeat case: `20250806/10.csv` and `20250807/10Pos.csv` — 19 rows;
- external frame/TCP diagnostic: `20250714/建模数据.csv` — 97 accepted rows.

Nine of ten 2025-08-06 points match the nine 2025-08-07 points under maximum joint difference 0.02° and nominal-TCP difference 0.2 mm. Actual matches are much tighter: nominal TCP differences are below 0.055 mm. The paired error-vector change norm is mean 0.084 mm, P95 0.161 mm and maximum 0.194 mm. This is a credible `REAL_STATIC` repeated-point case study, but not a continuous path.

Current DH/FK agrees with most files to median FK–nominal differences of about 0.8–1.4 mm. `20250714/建模数据.csv` is the exception at about 49.7 mm and is therefore not used for simulator calibration. It likely reflects a different frame/TCP/convention, but that interpretation is unverified.

## 2. Static prior benchmark

All priors use `q_rad[6] + nominal_TCP_m[3] -> position_error_m[3]`. Input/target transforms are fitted on candidate training groups only. These models are simulator priors and supervised baselines, not a paper contribution.

| Model | Validation RMSE mm | Cross-date test RMSE mm | Test P95 mm |
|---|---:|---:|---:|
| zero error | 1.853 | 3.944 | 4.946 |
| mean bias | 2.002 | 2.417 | 3.296 |
| Ridge | 1.975 | 7.093 | 9.206 |
| ExtraTrees | **1.825** | 1.751 | 2.148 |
| Random Forest | 1.883 | **1.653** | **2.142** |
| RBF kernel | 1.987 | 3.951 | 4.955 |
| lightweight MLP | 2.008 | 3.231 | 4.264 |

ExtraTrees is selected by validation RMSE; Random Forest happens to be best on the frozen cross-date group and is not retroactively selected. Ridge extrapolates badly across the candidate date split. RBF behaves close to zero prediction under the shift. MLP required training-target standardization to avoid numerical divergence but remains weaker than the tree models.

## 3. Simulator calibration and trajectories

Three simulator families were constructed from Ridge, compact ExtraTrees and RBF priors. Terminology is fixed as follows:

- prediction residual after removing each file mean: `unexplained residual`, not measurement noise;
- differences among file-level residual means: `session-shift proxy`, not confirmed physical calibration drift.

| Prior | Unexplained residual P95 mm | Session-shift proxy P95 mm |
|---|---:|---:|
| Ridge | 1.140 | 1.244 |
| compact ExtraTrees | 0.476 | 0.372 |
| RBF kernel | 0.431 | 0.170 |

Continuous paths are generated in joint space from the training support and passed through UR5 FK. CSV row order is never used. Mean trajectory P95 standardized support distances are 0.469 for sine, 0.618 for smooth random, and 2.439 for workspace holdout. Holdout paths are outside the training joint bounds for 100% of points and are labeled `SIM_STRESS`.

In a diagnostic CEM check with unseen RBF simulator prior, single-prior training achieved 0.628 mm RMSE while multi-prior training achieved 0.781 mm versus 0.820 mm for the projected Ridge base. Multi-prior CEM was not automatically superior and had higher action variation (43.8 mm total variation) plus 0.5% safety clipping. CEM remains an integration diagnostic, not the proposed method.

## 4. Sequence observation and RL protocol

The observation now includes four-step error history, four-step applied-action history, history-validity mask, and explicit 0–3 step delay one-hot state. Gymnasium still exposes `[-1,1]^3`; internal residuals remain metres with 2 mm residual and 6 mm total Cartesian projection limits.

Training and evaluation are separated:

- train: Ridge/ExtraTrees simulator families, domain-randomized generated paths;
- validation: disjoint seeds and paths from the same two prior families; validation RMSE selects checkpoints;
- frozen test: unseen RBF simulator prior, five fixed test seeds; never accessed by checkpoint selection.

SAC and TD3 each used three training seeds (`401,402,403`) and 4000 steps. Validation curves usually worsen after early checkpoints, so adding steps does not yield consistent improvement.

| Frozen scenario | Projected Ridge base | SAC, aggregate | TD3, aggregate |
|---|---:|---:|---:|
| unseen RBF prior + unseen path (`SIM_CALIBRATED`) | **1.086** | 1.111 | 1.116 |
| unseen RBF prior + workspace holdout (`SIM_STRESS`) | 7.471 | **7.445** | 7.455 |

Failure conclusion: neither SAC nor TD3 beats the projected base on the primary unseen-path test. The 0.016–0.026 mm holdout improvement is negligible relative to approximately 7.45 mm absolute error and does not establish robust OOD generalization. No seeds were removed and no further timestep increase was used to search for a favorable result.

## 5. Verification and remaining limits

Twenty-seven tests pass, including FK/Jacobian, hard safety bounds, grouped split leakage, prior interfaces, training-only support fitting, matched-point thresholds, trajectory smoothness/OOD, history masks, delayed applied actions, explicit delay state, Gymnasium checking and deterministic seeded rollouts.

Remaining evidence gaps:

- candidate file roles and all coordinate/TCP/unit metadata require confirmation before a final real manifest;
- the 20250714 50 mm FK mismatch is unresolved;
- simulator dynamics are limited to positioning priors, unexplained residual sampling, session-shift proxies and action delay;
- there is no real action/transition log, force/contact model, flexible sheet model or rolling-quality evidence;
- current RL does not outperform the strong projected base on the primary frozen test.
