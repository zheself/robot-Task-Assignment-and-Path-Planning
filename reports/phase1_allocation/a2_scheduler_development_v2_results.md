# A2 scheduler development v1 results

Evidence: `SIM_GEOMETRIC`; usage: development only.
Source manifest: `c9f532e72db33324e8f84677001cab028eb9a8e5634857998375ae7ed99f0842`.
Accessed splits: `train, validation`. Frozen-test and stress were not accessed.

| split | cell | method | coverage | verified/total | mean score (verified only) | median runtime s |
|---|---|---|---:|---:|---:|---:|
| train | medium_balanced | hybrid_assignment_milp_v2 | 1.0000 | 48/48 | 47.9389 | 0.194178 |
| train | medium_balanced | hybrid_load_balanced_v2 | 1.0000 | 48/48 | 49.6286 | 0.012650 |
| train | medium_balanced | load_balanced_v1 | 1.0000 | 48/48 | 49.6286 | 0.011563 |
| train | medium_balanced | order_aware_lns_v2 | 1.0000 | 48/48 | 37.8754 | 0.489860 |
| train | medium_precedence | hybrid_assignment_milp_v2 | 0.9583 | 46/48 | 61.5992 | 0.395413 |
| train | medium_precedence | hybrid_load_balanced_v2 | 0.9583 | 46/48 | 64.7683 | 0.015897 |
| train | medium_precedence | load_balanced_v1 | 0.8750 | 42/48 | 58.2335 | 0.014159 |
| train | medium_precedence | order_aware_lns_v2 | 0.9583 | 46/48 | 48.1637 | 0.567808 |
| train | medium_resource | hybrid_assignment_milp_v2 | 0.9375 | 45/48 | 59.3108 | 0.356766 |
| train | medium_resource | hybrid_load_balanced_v2 | 0.9583 | 46/48 | 59.3302 | 0.015981 |
| train | medium_resource | load_balanced_v1 | 0.8333 | 40/48 | 55.1294 | 0.013476 |
| train | medium_resource | order_aware_lns_v2 | 0.9792 | 47/48 | 46.9333 | 0.550228 |
| train | small_sparse | hybrid_assignment_milp_v2 | 1.0000 | 48/48 | 34.2598 | 0.021542 |
| train | small_sparse | hybrid_load_balanced_v2 | 1.0000 | 48/48 | 37.0799 | 0.005316 |
| train | small_sparse | load_balanced_v1 | 1.0000 | 48/48 | 37.1296 | 0.004964 |
| train | small_sparse | order_aware_lns_v2 | 1.0000 | 48/48 | 30.5758 | 0.235162 |
| validation | medium_balanced | hybrid_assignment_milp_v2 | 1.0000 | 12/12 | 52.6701 | 0.294079 |
| validation | medium_balanced | hybrid_load_balanced_v2 | 1.0000 | 12/12 | 53.5015 | 0.014477 |
| validation | medium_balanced | load_balanced_v1 | 1.0000 | 12/12 | 53.5015 | 0.013127 |
| validation | medium_balanced | order_aware_lns_v2 | 1.0000 | 12/12 | 42.8651 | 0.530681 |
| validation | medium_precedence | hybrid_assignment_milp_v2 | 0.9167 | 11/12 | 66.4629 | 0.482392 |
| validation | medium_precedence | hybrid_load_balanced_v2 | 0.9167 | 11/12 | 71.6174 | 0.017186 |
| validation | medium_precedence | load_balanced_v1 | 0.8333 | 10/12 | 62.0190 | 0.016123 |
| validation | medium_precedence | order_aware_lns_v2 | 0.9167 | 11/12 | 48.0252 | 0.595115 |
| validation | medium_resource | hybrid_assignment_milp_v2 | 1.0000 | 12/12 | 80.3110 | 0.675258 |
| validation | medium_resource | hybrid_load_balanced_v2 | 1.0000 | 12/12 | 82.5367 | 0.018812 |
| validation | medium_resource | load_balanced_v1 | 0.8333 | 10/12 | 64.9605 | 0.016247 |
| validation | medium_resource | order_aware_lns_v2 | 1.0000 | 12/12 | 60.7367 | 0.761210 |
| validation | small_sparse | hybrid_assignment_milp_v2 | 1.0000 | 12/12 | 30.3941 | 0.014679 |
| validation | small_sparse | hybrid_load_balanced_v2 | 1.0000 | 12/12 | 31.2054 | 0.004233 |
| validation | small_sparse | load_balanced_v1 | 1.0000 | 12/12 | 31.5457 | 0.004011 |
| validation | small_sparse | order_aware_lns_v2 | 1.0000 | 12/12 | 28.6133 | 0.183923 |

`schedule_infeasible` remains a method failure, not proof of global infeasibility. The assignment MILP remains assignment-only and its gap is not a joint scheduling gap. No v2 frozen result was used for selection.
