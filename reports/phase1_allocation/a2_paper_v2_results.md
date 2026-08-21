# A2 paper-scale v2 benchmark results

Evidence: **SIM_GEOMETRIC**. These are programmatic continuous-process workcells, not real trajectories, collision certificates or physical-quality evidence.

- Manifest SHA-256: `c9f532e72db33324e8f84677001cab028eb9a8e5634857998375ae7ed99f0842`
- Instances: 408; independent task groups: 216; solver runs: 2040.
- Split instances: {'frozen_test': 144, 'stress': 24, 'train': 192, 'validation': 48}.
- Failed/unverified method runs: 600; status counts: {'infeasible': 30, 'schedule_infeasible': 570}.
- Preregistered A2 gate: **FAILED**.

## Acceptance checks

- PASS — `zero_schema_failures`
- PASS — `zero_split_leakage`
- PASS — `zero_hash_failures`
- PASS — `zero_unexpected_solver_status`
- PASS — `minimum_train_candidate_coverage`
- PASS — `minimum_validation_candidate_coverage`
- FAIL — `minimum_frozen_cell_candidate_coverage`
- PASS — `designed_infeasible_detection_rate`
- Train candidate coverage: 0.948; validation: 0.938.
- Frozen cell coverage: {'iid_medium': 0.875, 'iid_small': 1.0, 'ood_dense_precedence': 0.375, 'ood_resource_bottleneck': 0.5416666666666666, 'ood_scale': 0.125, 'ood_tight_windows': 0.125}.
- Designed-infeasible detection: 1.000.

## Frozen-test verified-plan rates

| cell | method | verified groups rate [95% CI] | instances | runtime median [Q1,Q3] s |
|---|---|---:|---:|---:|
| iid_medium | assignment_milp | 0.8750 [0.7500,1.0000] | 21/24 | 0.2989 [0.0872,0.7559] |
| iid_medium | deterministic_lns | 0.8333 [0.7083,0.9583] | 20/24 | 0.3183 [0.2496,0.3831] |
| iid_medium | greedy | 0.8333 [0.7083,0.9583] | 20/24 | 0.0111 [0.0089,0.0133] |
| iid_medium | hungarian | 0.8333 [0.7083,0.9583] | 20/24 | 0.0115 [0.0093,0.0139] |
| iid_medium | load_balanced | 0.8333 [0.7083,0.9583] | 20/24 | 0.0108 [0.0088,0.0132] |
| iid_small | assignment_milp | 1.0000 [1.0000,1.0000] | 24/24 | 0.0218 [0.0128,0.0668] |
| iid_small | deterministic_lns | 1.0000 [1.0000,1.0000] | 24/24 | 0.1666 [0.1427,0.1848] |
| iid_small | greedy | 1.0000 [1.0000,1.0000] | 24/24 | 0.0044 [0.0040,0.0054] |
| iid_small | hungarian | 1.0000 [1.0000,1.0000] | 24/24 | 0.0045 [0.0040,0.0055] |
| iid_small | load_balanced | 1.0000 [1.0000,1.0000] | 24/24 | 0.0043 [0.0039,0.0053] |
| ood_dense_precedence | assignment_milp | 0.2500 [0.1250,0.3750] | 6/24 | 1.3546 [0.3211,2.0577] |
| ood_dense_precedence | deterministic_lns | 0.2917 [0.1250,0.4583] | 7/24 | 0.0321 [0.0270,0.4292] |
| ood_dense_precedence | greedy | 0.2917 [0.1667,0.4167] | 7/24 | 0.0160 [0.0138,0.0176] |
| ood_dense_precedence | hungarian | 0.3333 [0.2083,0.4583] | 8/24 | 0.0182 [0.0155,0.0200] |
| ood_dense_precedence | load_balanced | 0.2917 [0.1250,0.4583] | 7/24 | 0.0154 [0.0137,0.0178] |
| ood_resource_bottleneck | assignment_milp | 0.4583 [0.2917,0.6250] | 11/24 | 0.9449 [0.0716,1.5587] |
| ood_resource_bottleneck | deterministic_lns | 0.4583 [0.2917,0.6250] | 11/24 | 0.0368 [0.0291,0.4912] |
| ood_resource_bottleneck | greedy | 0.5000 [0.2917,0.7083] | 12/24 | 0.0163 [0.0149,0.0185] |
| ood_resource_bottleneck | hungarian | 0.5000 [0.2917,0.7083] | 12/24 | 0.0190 [0.0168,0.0205] |
| ood_resource_bottleneck | load_balanced | 0.4583 [0.2917,0.6250] | 11/24 | 0.0158 [0.0149,0.0185] |
| ood_scale | assignment_milp | 0.1250 [0.0000,0.3333] | 3/24 | 3.0346 [2.9308,3.0450] |
| ood_scale | deterministic_lns | 0.0833 [0.0000,0.2500] | 2/24 | 0.0552 [0.0513,0.0730] |
| ood_scale | greedy | 0.0417 [0.0000,0.1250] | 1/24 | 0.0280 [0.0261,0.0363] |
| ood_scale | hungarian | 0.1250 [0.0000,0.2917] | 3/24 | 0.0366 [0.0313,0.0474] |
| ood_scale | load_balanced | 0.0833 [0.0000,0.2500] | 2/24 | 0.0282 [0.0258,0.0367] |
| ood_tight_windows | assignment_milp | 0.0833 [0.0000,0.2083] | 2/24 | 0.2483 [0.0307,0.5530] |
| ood_tight_windows | deterministic_lns | 0.0833 [0.0000,0.2083] | 2/24 | 0.0194 [0.0148,0.0277] |
| ood_tight_windows | greedy | 0.0833 [0.0000,0.2083] | 2/24 | 0.0094 [0.0075,0.0131] |
| ood_tight_windows | hungarian | 0.1250 [0.0000,0.2500] | 3/24 | 0.0109 [0.0081,0.0145] |
| ood_tight_windows | load_balanced | 0.0833 [0.0000,0.2083] | 2/24 | 0.0099 [0.0075,0.0117] |

## Statistical interpretation

Holm-adjusted paired score comparisons with p<0.05: 10/32 testable comparisons. Full effect sizes, confidence intervals and jointly verified group counts are in `a2_paper_v2_pairwise.csv`.
Quality comparisons are conditional on pairwise jointly verified variants; failed plans receive no imputed quality score. Stress results are descriptive only.

## Candidate-label availability

- train: 182/192 verified candidates; planned use `fit_only`.
- validation: 45/48 verified candidates; planned use `selection_only`.
- frozen_test: 73/144 verified candidates; planned use `evaluation_only`.
- stress: 0/24 verified candidates; planned use `failure_only`.

## Boundaries

`proxy_admissible` is only edge-mask coverage for atomic assignment units. `schedule_infeasible` is a method failure, not proof of global infeasibility. Assignment MIP gap is not joint scheduling or path-planning optimality. No GNN was trained in A2.

## Reproduction

`PYTHONPATH=src .venv/bin/python scripts/run_a2_paper_v2.py`
