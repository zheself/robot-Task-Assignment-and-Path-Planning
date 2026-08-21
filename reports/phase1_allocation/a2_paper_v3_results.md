# A2 paper-scale v3 benchmark results

Evidence: **SIM_GEOMETRIC**. These are programmatic continuous-process workcells, not real trajectories, collision certificates or physical-quality evidence.

- Manifest SHA-256: `a039238d50ac10a6ecc44d6001f07aa75cbc3169c9ac5c8c1800a1794742ae12`
- Instances: 408; independent task groups: 216; solver runs: 3264.
- Split instances: {'frozen_test': 144, 'stress': 24, 'train': 192, 'validation': 48}.
- Failed/unverified method runs: 788; status counts: {'infeasible': 48, 'schedule_infeasible': 740}.
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
- Train candidate coverage: 0.990; validation: 0.979.
- Frozen cell coverage: {'iid_medium': 0.9583333333333334, 'iid_small': 1.0, 'ood_dense_precedence': 0.75, 'ood_resource_bottleneck': 0.8333333333333334, 'ood_scale': 0.4583333333333333, 'ood_tight_windows': 0.5416666666666666}.
- Designed-infeasible detection: 1.000.

## Frozen-test verified-plan rates

| cell | method | verified groups rate [95% CI] | instances | runtime median [Q1,Q3] s |
|---|---|---:|---:|---:|
| iid_medium | assignment_milp | 0.7917 [0.5833,0.9583] | 19/24 | 0.4910 [0.1857,0.7406] |
| iid_medium | deterministic_lns | 0.7917 [0.5833,0.9583] | 19/24 | 0.3930 [0.2767,0.4536] |
| iid_medium | greedy | 0.7917 [0.5833,0.9583] | 19/24 | 0.0119 [0.0100,0.0161] |
| iid_medium | hungarian | 0.7917 [0.5833,0.9583] | 19/24 | 0.0128 [0.0107,0.0172] |
| iid_medium | hybrid_assignment_milp | 0.9583 [0.8750,1.0000] | 23/24 | 0.4752 [0.1858,0.7431] |
| iid_medium | hybrid_load_balanced | 0.9583 [0.8750,1.0000] | 23/24 | 0.0139 [0.0118,0.0193] |
| iid_medium | load_balanced | 0.7917 [0.5833,0.9583] | 19/24 | 0.0119 [0.0100,0.0165] |
| iid_medium | order_aware_lns | 0.9583 [0.8750,1.0000] | 23/24 | 0.5668 [0.4448,0.6410] |
| iid_small | assignment_milp | 1.0000 [1.0000,1.0000] | 24/24 | 0.0136 [0.0108,0.0253] |
| iid_small | deterministic_lns | 1.0000 [1.0000,1.0000] | 24/24 | 0.2107 [0.1510,0.2277] |
| iid_small | greedy | 1.0000 [1.0000,1.0000] | 24/24 | 0.0051 [0.0037,0.0068] |
| iid_small | hungarian | 1.0000 [1.0000,1.0000] | 24/24 | 0.0051 [0.0037,0.0070] |
| iid_small | hybrid_assignment_milp | 1.0000 [1.0000,1.0000] | 24/24 | 0.0140 [0.0107,0.0264] |
| iid_small | hybrid_load_balanced | 1.0000 [1.0000,1.0000] | 24/24 | 0.0055 [0.0039,0.0074] |
| iid_small | load_balanced | 1.0000 [1.0000,1.0000] | 24/24 | 0.0050 [0.0036,0.0068] |
| iid_small | order_aware_lns | 1.0000 [1.0000,1.0000] | 24/24 | 0.2532 [0.1751,0.2840] |
| ood_dense_precedence | assignment_milp | 0.5417 [0.3750,0.7083] | 13/24 | 1.1974 [0.5139,2.6368] |
| ood_dense_precedence | deterministic_lns | 0.5000 [0.2917,0.7083] | 12/24 | 0.2473 [0.0367,0.5739] |
| ood_dense_precedence | greedy | 0.5833 [0.3750,0.7500] | 14/24 | 0.0223 [0.0156,0.0253] |
| ood_dense_precedence | hungarian | 0.5000 [0.2917,0.7083] | 12/24 | 0.0232 [0.0167,0.0267] |
| ood_dense_precedence | hybrid_assignment_milp | 0.7500 [0.6250,0.8750] | 18/24 | 1.1694 [0.5176,2.6502] |
| ood_dense_precedence | hybrid_load_balanced | 0.7083 [0.5833,0.8333] | 17/24 | 0.0232 [0.0195,0.0297] |
| ood_dense_precedence | load_balanced | 0.5000 [0.2917,0.7083] | 12/24 | 0.0209 [0.0155,0.0237] |
| ood_dense_precedence | order_aware_lns | 0.7500 [0.6250,0.8750] | 18/24 | 0.7597 [0.5345,0.8555] |
| ood_resource_bottleneck | assignment_milp | 0.5000 [0.3750,0.6250] | 12/24 | 1.5445 [1.0281,2.6951] |
| ood_resource_bottleneck | deterministic_lns | 0.5000 [0.3333,0.6667] | 12/24 | 0.2451 [0.0330,0.6024] |
| ood_resource_bottleneck | greedy | 0.4583 [0.3333,0.5833] | 11/24 | 0.0201 [0.0166,0.0224] |
| ood_resource_bottleneck | hungarian | 0.4583 [0.3333,0.5833] | 11/24 | 0.0225 [0.0185,0.0250] |
| ood_resource_bottleneck | hybrid_assignment_milp | 0.7917 [0.5833,0.9583] | 19/24 | 1.5287 [1.0492,2.7064] |
| ood_resource_bottleneck | hybrid_load_balanced | 0.7917 [0.5833,0.9583] | 19/24 | 0.0241 [0.0231,0.0279] |
| ood_resource_bottleneck | load_balanced | 0.5000 [0.3333,0.6667] | 12/24 | 0.0202 [0.0166,0.0224] |
| ood_resource_bottleneck | order_aware_lns | 0.8333 [0.6250,1.0000] | 20/24 | 0.8999 [0.7818,1.0539] |
| ood_scale | assignment_milp | 0.1250 [0.0000,0.2500] | 3/24 | 3.0445 [2.9815,3.0530] |
| ood_scale | deterministic_lns | 0.1250 [0.0000,0.2500] | 3/24 | 0.0840 [0.0730,0.0927] |
| ood_scale | greedy | 0.1250 [0.0000,0.2500] | 3/24 | 0.0397 [0.0361,0.0468] |
| ood_scale | hungarian | 0.1250 [0.0000,0.2500] | 3/24 | 0.0474 [0.0437,0.0577] |
| ood_scale | hybrid_assignment_milp | 0.4167 [0.1667,0.6667] | 10/24 | 3.0502 [2.9882,3.0587] |
| ood_scale | hybrid_load_balanced | 0.3750 [0.1667,0.6250] | 9/24 | 0.0474 [0.0418,0.0540] |
| ood_scale | load_balanced | 0.1250 [0.0000,0.2500] | 3/24 | 0.0399 [0.0363,0.0460] |
| ood_scale | order_aware_lns | 0.4583 [0.2083,0.7083] | 11/24 | 1.0378 [0.6308,1.5960] |
| ood_tight_windows | assignment_milp | 0.2500 [0.0833,0.4583] | 6/24 | 0.6530 [0.1131,1.0494] |
| ood_tight_windows | deterministic_lns | 0.2500 [0.0833,0.4583] | 6/24 | 0.0306 [0.0214,0.1182] |
| ood_tight_windows | greedy | 0.2500 [0.0833,0.4583] | 6/24 | 0.0134 [0.0108,0.0160] |
| ood_tight_windows | hungarian | 0.2500 [0.0833,0.4583] | 6/24 | 0.0141 [0.0116,0.0175] |
| ood_tight_windows | hybrid_assignment_milp | 0.5000 [0.2500,0.7083] | 12/24 | 0.6597 [0.1076,1.0576] |
| ood_tight_windows | hybrid_load_balanced | 0.4583 [0.2083,0.7083] | 11/24 | 0.0164 [0.0131,0.0177] |
| ood_tight_windows | load_balanced | 0.2500 [0.0417,0.4583] | 6/24 | 0.0130 [0.0108,0.0159] |
| ood_tight_windows | order_aware_lns | 0.5417 [0.2917,0.7917] | 13/24 | 0.3669 [0.1959,0.5856] |

## Statistical interpretation

Holm-adjusted paired score comparisons with p<0.05: 9/64 testable comparisons. Full effect sizes, confidence intervals and jointly verified group counts are in `a2_paper_v3_pairwise.csv`.
Quality comparisons are conditional on pairwise jointly verified variants; failed plans receive no imputed quality score. Stress results are descriptive only.

## Candidate-label availability

- train: 190/192 verified candidates; planned use `fit_only`.
- validation: 47/48 verified candidates; planned use `selection_only`.
- frozen_test: 109/144 verified candidates; planned use `evaluation_only`.
- stress: 3/24 verified candidates; planned use `evaluation_only`.

## Boundaries

`proxy_admissible` is only edge-mask coverage for atomic assignment units. `schedule_infeasible` is a method failure, not proof of global infeasibility. Assignment MIP gap is not joint scheduling or path-planning optimality. No GNN was trained in A2.

## Reproduction

Run `scripts/run_a2_paper_v2.py` with the recorded config, output, and manifest paths.
