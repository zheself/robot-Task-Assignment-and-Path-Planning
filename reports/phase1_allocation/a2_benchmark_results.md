# A2 leakage-safe geometric benchmark pilot results

Evidence: **SIM_GEOMETRIC**. This frozen engineering pilot contains programmatic continuous curves and proxy workcells, not real factory trajectories or a paper-scale training corpus.

- Manifest SHA-256: `3158b1f079f6896bc67de134e00a095ded077a15fabb22c85a415f7d95895b4a`
- Instances: 19; solver runs: 95; recorded failed/unverified runs: 23.
- Frozen-test and stress labels are marked `evaluation_only`; they cannot be used for fitting or model selection.
- `assignment_mip_gap` certifies only the assignment MILP formulation. `best_observed_relative_gap` is descriptive, not a certified scheduling optimality gap.
- Failure status counts: `schedule_infeasible`=23.

| split | method | verified | median runtime (s) | mean makespan (s) | mean load variance (s²) |
|---|---|---:|---:|---:|---:|
| frozen_test | assignment_milp | 4/6 | 0.133514 | 45.169540 | 13.484138 |
| frozen_test | deterministic_lns | 4/6 | 0.188698 | 38.463040 | 7.861711 |
| frozen_test | greedy | 4/6 | 0.008425 | 43.091878 | 48.847997 |
| frozen_test | hungarian | 3/6 | 0.008745 | 48.925717 | 10.847349 |
| frozen_test | load_balanced | 4/6 | 0.008472 | 44.326118 | 14.236703 |
| stress | assignment_milp | 0/1 | 10.044093 | n/a | n/a |
| stress | deterministic_lns | 0/1 | 0.069955 | n/a | n/a |
| stress | greedy | 0/1 | 0.034964 | n/a | n/a |
| stress | hungarian | 0/1 | 0.044696 | n/a | n/a |
| stress | load_balanced | 0/1 | 0.034942 | n/a | n/a |
| train | assignment_milp | 8/8 | 0.012805 | 38.473080 | 14.577169 |
| train | deterministic_lns | 8/8 | 0.205316 | 31.959316 | 11.716013 |
| train | greedy | 8/8 | 0.005267 | 38.957009 | 74.205871 |
| train | hungarian | 8/8 | 0.005300 | 34.649144 | 9.289063 |
| train | load_balanced | 8/8 | 0.005226 | 37.996096 | 31.338021 |
| validation | assignment_milp | 3/4 | 0.019757 | 27.864537 | 5.677795 |
| validation | deterministic_lns | 3/4 | 0.131234 | 24.530976 | 3.767379 |
| validation | greedy | 2/4 | 0.003877 | 27.931174 | 17.588910 |
| validation | hungarian | 2/4 | 0.003866 | 24.378440 | 8.390130 |
| validation | load_balanced | 3/4 | 0.004358 | 28.821603 | 5.608103 |

## Candidate label availability

- train: 8/8 instances have a verified candidate; usage is `fit_only`.
- validation: 3/4 instances have a verified candidate; usage is `selection_only`.
- frozen_test: 5/6 instances have a verified candidate; usage is `evaluation_only`.
- stress: 0/1 instances have a verified candidate; usage is `failure_only`.

## Boundaries and next gate

Pilot v1 validates A2 plumbing and freezes its own evidence. It does not establish GNN superiority, real collision safety, real cycle time, path executability, or physical quality improvement. A3 remains blocked until a new paper-scale A2 manifest with substantially more independent groups is preregistered and evaluated.

## Reproduction

`PYTHONPATH=src .venv/bin/python scripts/run_a2_benchmark.py`
