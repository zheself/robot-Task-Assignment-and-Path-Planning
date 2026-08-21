# A2 paper-scale v4 benchmark results

Evidence: **SIM_GEOMETRIC**. These are programmatic continuous-process workcells, not real trajectories, collision certificates or physical-quality evidence.

- Manifest SHA-256: `0c98f30e92697ce8b5eca724df0f7d1b7053293df1e792707487ecb6c71b5398`
- Instances: 408; independent task groups: 216; solver runs: 3264.
- Split instances: {'frozen_test': 144, 'stress': 24, 'train': 192, 'validation': 48}.
- Failed/unverified method runs: 740; status counts: {'infeasible': 48, 'schedule_infeasible': 692}.
- Preregistered A2 gate: **PASSED**.

## Acceptance checks

- PASS — `zero_schema_failures`
- PASS — `zero_split_leakage`
- PASS — `zero_hash_failures`
- PASS — `zero_unexpected_solver_status`
- PASS — `minimum_train_candidate_coverage`
- PASS — `minimum_validation_candidate_coverage`
- PASS — `minimum_frozen_cell_candidate_coverage`
- PASS — `designed_infeasible_detection_rate`
- PASS — `zero_witness_failures`
- Train candidate coverage: 0.995; validation: 1.000.
- Frozen cell coverage: {'iid_medium': 1.0, 'iid_small': 1.0, 'ood_dense_precedence': 0.8333333333333334, 'ood_resource_bottleneck': 0.9166666666666666, 'ood_scale': 0.5, 'ood_tight_windows': 0.5}.
- Designed-infeasible detection: 1.000.

## Frozen-test verified-plan rates

| cell | method | verified groups rate [95% CI] | instances | runtime median [Q1,Q3] s |
|---|---|---:|---:|---:|
| iid_medium | assignment_milp | 0.8750 [0.7500,1.0000] | 21/24 | 0.2307 [0.0646,0.6561] |
| iid_medium | deterministic_lns | 0.9167 [0.7917,1.0000] | 22/24 | 0.3850 [0.3550,0.4205] |
| iid_medium | greedy | 0.8750 [0.7500,1.0000] | 21/24 | 0.0135 [0.0114,0.0159] |
| iid_medium | hungarian | 0.9167 [0.7917,1.0000] | 22/24 | 0.0147 [0.0127,0.0165] |
| iid_medium | hybrid_assignment_milp | 1.0000 [1.0000,1.0000] | 24/24 | 0.2316 [0.0651,0.6568] |
| iid_medium | hybrid_load_balanced | 1.0000 [1.0000,1.0000] | 24/24 | 0.0155 [0.0140,0.0176] |
| iid_medium | load_balanced | 0.9167 [0.7917,1.0000] | 22/24 | 0.0140 [0.0118,0.0157] |
| iid_medium | order_aware_lns | 1.0000 [1.0000,1.0000] | 24/24 | 0.5576 [0.4911,0.6085] |
| iid_small | assignment_milp | 1.0000 [1.0000,1.0000] | 24/24 | 0.0224 [0.0144,0.0853] |
| iid_small | deterministic_lns | 1.0000 [1.0000,1.0000] | 24/24 | 0.1964 [0.1719,0.2176] |
| iid_small | greedy | 1.0000 [1.0000,1.0000] | 24/24 | 0.0054 [0.0045,0.0057] |
| iid_small | hungarian | 1.0000 [1.0000,1.0000] | 24/24 | 0.0054 [0.0045,0.0058] |
| iid_small | hybrid_assignment_milp | 1.0000 [1.0000,1.0000] | 24/24 | 0.0220 [0.0142,0.0886] |
| iid_small | hybrid_load_balanced | 1.0000 [1.0000,1.0000] | 24/24 | 0.0058 [0.0048,0.0061] |
| iid_small | load_balanced | 1.0000 [1.0000,1.0000] | 24/24 | 0.0053 [0.0044,0.0056] |
| iid_small | order_aware_lns | 1.0000 [1.0000,1.0000] | 24/24 | 0.2327 [0.2013,0.2658] |
| ood_dense_precedence | assignment_milp | 0.5417 [0.4167,0.6667] | 13/24 | 1.2322 [0.6951,2.0435] |
| ood_dense_precedence | deterministic_lns | 0.5417 [0.4167,0.6667] | 13/24 | 0.4745 [0.0392,0.5330] |
| ood_dense_precedence | greedy | 0.5417 [0.4167,0.6667] | 13/24 | 0.0211 [0.0173,0.0250] |
| ood_dense_precedence | hungarian | 0.5417 [0.4167,0.6667] | 13/24 | 0.0235 [0.0196,0.0277] |
| ood_dense_precedence | hybrid_assignment_milp | 0.6667 [0.5417,0.7917] | 16/24 | 1.2355 [0.6937,2.0442] |
| ood_dense_precedence | hybrid_load_balanced | 0.6667 [0.5417,0.7917] | 16/24 | 0.0227 [0.0203,0.0267] |
| ood_dense_precedence | load_balanced | 0.5417 [0.4167,0.6667] | 13/24 | 0.0211 [0.0173,0.0246] |
| ood_dense_precedence | order_aware_lns | 0.8333 [0.7083,0.9583] | 20/24 | 0.6533 [0.3774,0.8404] |
| ood_resource_bottleneck | assignment_milp | 0.5833 [0.3750,0.7500] | 14/24 | 1.4962 [0.7176,2.2554] |
| ood_resource_bottleneck | deterministic_lns | 0.5417 [0.3333,0.7500] | 13/24 | 0.5050 [0.0386,0.6225] |
| ood_resource_bottleneck | greedy | 0.5417 [0.3750,0.7083] | 13/24 | 0.0216 [0.0196,0.0244] |
| ood_resource_bottleneck | hungarian | 0.5000 [0.2917,0.7083] | 12/24 | 0.0245 [0.0205,0.0276] |
| ood_resource_bottleneck | hybrid_assignment_milp | 0.8750 [0.7500,1.0000] | 21/24 | 1.5023 [0.7295,2.3173] |
| ood_resource_bottleneck | hybrid_load_balanced | 0.8333 [0.7083,0.9583] | 20/24 | 0.0268 [0.0248,0.0294] |
| ood_resource_bottleneck | load_balanced | 0.5417 [0.3333,0.7500] | 13/24 | 0.0218 [0.0198,0.0247] |
| ood_resource_bottleneck | order_aware_lns | 0.9167 [0.7917,1.0000] | 22/24 | 0.9435 [0.8266,1.0956] |
| ood_scale | assignment_milp | 0.1250 [0.0000,0.2500] | 3/24 | 3.0496 [3.0429,3.0521] |
| ood_scale | deterministic_lns | 0.1667 [0.0417,0.2917] | 4/24 | 0.0844 [0.0720,0.0941] |
| ood_scale | greedy | 0.1250 [0.0000,0.2500] | 3/24 | 0.0410 [0.0359,0.0466] |
| ood_scale | hungarian | 0.1250 [0.0000,0.2500] | 3/24 | 0.0527 [0.0432,0.0573] |
| ood_scale | hybrid_assignment_milp | 0.4167 [0.2083,0.6667] | 10/24 | 3.0501 [3.0456,3.0584] |
| ood_scale | hybrid_load_balanced | 0.3750 [0.1667,0.6250] | 9/24 | 0.0444 [0.0411,0.0492] |
| ood_scale | load_balanced | 0.1667 [0.0417,0.2917] | 4/24 | 0.0414 [0.0363,0.0465] |
| ood_scale | order_aware_lns | 0.5000 [0.2917,0.7083] | 12/24 | 0.7650 [0.4210,1.4823] |
| ood_tight_windows | assignment_milp | 0.1667 [0.0417,0.2917] | 4/24 | 1.1062 [0.3977,1.5674] |
| ood_tight_windows | deterministic_lns | 0.1667 [0.0417,0.2917] | 4/24 | 0.0325 [0.0286,0.0390] |
| ood_tight_windows | greedy | 0.1667 [0.0417,0.2917] | 4/24 | 0.0161 [0.0146,0.0178] |
| ood_tight_windows | hungarian | 0.2083 [0.0833,0.3333] | 5/24 | 0.0175 [0.0159,0.0217] |
| ood_tight_windows | hybrid_assignment_milp | 0.3333 [0.1667,0.5000] | 8/24 | 1.1098 [0.4107,1.5532] |
| ood_tight_windows | hybrid_load_balanced | 0.2500 [0.0833,0.4583] | 6/24 | 0.0172 [0.0150,0.0211] |
| ood_tight_windows | load_balanced | 0.1667 [0.0417,0.2917] | 4/24 | 0.0160 [0.0145,0.0176] |
| ood_tight_windows | order_aware_lns | 0.5000 [0.3750,0.6250] | 12/24 | 0.2591 [0.1492,0.5798] |

## Statistical interpretation

Holm-adjusted paired score comparisons with p<0.05: 25/61 testable comparisons. Full effect sizes, confidence intervals and jointly verified group counts are in `a2_paper_v4_pairwise.csv`.
Quality comparisons are conditional on pairwise jointly verified variants; failed plans receive no imputed quality score. Stress results are descriptive only.

## Candidate-label availability

- train: 191/192 verified candidates; planned use `fit_only`.
- validation: 48/48 verified candidates; planned use `selection_only`.
- frozen_test: 114/144 verified candidates; planned use `evaluation_only`.
- stress: 4/24 verified candidates; planned use `evaluation_only`.

## Boundaries

Every ordinary v4 instance has a hashed constructive A1-proxy witness. A candidate `schedule_infeasible` result is therefore a method failure, not instance infeasibility. Witnesses were audit-only and were not candidate-solver inputs. Assignment MIP gap is not joint scheduling or path-planning optimality. No GNN was trained in A2.

## Reproduction

Run `scripts/run_a2_paper_v2.py` with the recorded config, output, and manifest paths.
