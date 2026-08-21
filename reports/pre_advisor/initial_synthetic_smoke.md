# Initial synthetic end-to-end smoke result

> Historical first run. Superseded for current baseline/scenario values by `next_batch_results.md` and `synthetic_method_comparison.csv`.

Run status: passed  
Evidence level: `synthetic_for_pipeline_validation_only`  
Purpose: verify research software interfaces before verified real metadata and continuous logs are available.

## What ran

The command below generated disjoint synthetic train/validation/cross-date sessions, fitted the static error prior on the training split only, optimized a three-parameter CEM feedback policy on separate training paths, and evaluated four methods on the same five seeds in four held-out scenarios.

```bash
python scripts/run_pre_advisor_smoke.py
```

The run used 480 training points, 240 same-date validation points and 240 held-out cross-date points. The static prior obtained 0.345 mm validation RMSE and 1.486 mm cross-date RMSE. The degradation under the unseen date bias is intentional and confirms that the test contains a shift not fitted by the prior.

## Mean paired trajectory RMSE (mm)

| Scenario | No compensation | Supervised prior | Fixed feedback | CEM smoke policy |
|---|---:|---:|---:|---:|
| unseen path | 4.393 | 0.382 | 1.469 | **0.354** |
| cross-date drift | 5.320 | 1.334 | 1.629 | **1.049** |
| workspace holdout | 4.339 | 0.684 | 1.697 | **0.576** |
| noise + 2-step delay | 4.963 | 1.078 | 1.674 | **0.937** |

The CEM policy's mean safety-clip rate was 0 in all four scenarios. The naive fixed unit-gain feedback had 0.063–0.140 clip rate and 124–184 mm total action variation, so it is not an acceptable final controller. This is a useful diagnostic: sequential compensation requires delay-aware gain/policy design and explicit smoothness/safety objectives.

## Tests

Six dependency-light tests passed:

- rigid-transform FK;
- finite-difference Jacobian consistency;
- orthonormal path-local frames;
- held-out synthetic prior improvement;
- Cartesian and joint-step safety bounds;
- deterministic seeded environment rollouts.

## What this does and does not establish

This run establishes that the data → prior → environment → mechanism projection → policy learning → paired evaluation → report path executes coherently. It does not establish real-robot accuracy, rolling-hemming quality, factory safety, successful sim-to-real transfer, or superiority of a paper method. CEM is only a low-cost learning-loop smoke test; SAC/TD3 and the proposed method have not yet been run.

The detailed machine-readable run remains under ignored `outputs/pre_advisor_smoke/summary.json`. Future promoted results must record dependency lock, commit hash, scenario config and multiple training seeds.
