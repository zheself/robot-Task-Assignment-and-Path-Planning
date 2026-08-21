# A2 scheduler development v1 results

Evidence: `SIM_GEOMETRIC`; usage: development only.
Source manifest: `c9f532e72db33324e8f84677001cab028eb9a8e5634857998375ae7ed99f0842`.
Accessed splits: `train, validation`. Frozen-test and stress were not accessed.

| split | cell | method | coverage | verified/total | mean score (verified only) | median runtime s |
|---|---|---|---:|---:|---:|---:|
| train | medium_balanced | deadline_assignment_milp_v2 | 1.0000 | 48/48 | 74.9248 | 0.189660 |
| train | medium_balanced | deadline_load_balanced_v2 | 1.0000 | 48/48 | 94.8827 | 0.011938 |
| train | medium_balanced | load_balanced_v1 | 1.0000 | 48/48 | 49.6286 | 0.011358 |
| train | medium_balanced | order_aware_lns_v2 | 1.0000 | 48/48 | 55.0906 | 0.417025 |
| train | medium_precedence | deadline_assignment_milp_v2 | 0.9375 | 45/48 | 92.1022 | 0.397953 |
| train | medium_precedence | deadline_load_balanced_v2 | 0.9375 | 45/48 | 116.6909 | 0.015055 |
| train | medium_precedence | load_balanced_v1 | 0.8750 | 42/48 | 58.2335 | 0.014030 |
| train | medium_precedence | order_aware_lns_v2 | 0.9375 | 45/48 | 67.7036 | 0.492470 |
| train | medium_resource | deadline_assignment_milp_v2 | 0.9375 | 45/48 | 88.4403 | 0.355865 |
| train | medium_resource | deadline_load_balanced_v2 | 0.9375 | 45/48 | 112.4916 | 0.015055 |
| train | medium_resource | load_balanced_v1 | 0.8333 | 40/48 | 55.1294 | 0.013738 |
| train | medium_resource | order_aware_lns_v2 | 0.9583 | 46/48 | 65.0515 | 0.467974 |
| train | small_sparse | deadline_assignment_milp_v2 | 1.0000 | 48/48 | 46.4557 | 0.021174 |
| train | small_sparse | deadline_load_balanced_v2 | 1.0000 | 48/48 | 56.5609 | 0.005035 |
| train | small_sparse | load_balanced_v1 | 1.0000 | 48/48 | 37.1296 | 0.004991 |
| train | small_sparse | order_aware_lns_v2 | 1.0000 | 48/48 | 39.4862 | 0.206517 |
| validation | medium_balanced | deadline_assignment_milp_v2 | 1.0000 | 12/12 | 84.8868 | 0.293610 |
| validation | medium_balanced | deadline_load_balanced_v2 | 1.0000 | 12/12 | 111.9759 | 0.013781 |
| validation | medium_balanced | load_balanced_v1 | 1.0000 | 12/12 | 53.5015 | 0.013190 |
| validation | medium_balanced | order_aware_lns_v2 | 1.0000 | 12/12 | 63.3752 | 0.467565 |
| validation | medium_precedence | deadline_assignment_milp_v2 | 0.9167 | 11/12 | 110.8370 | 0.489041 |
| validation | medium_precedence | deadline_load_balanced_v2 | 0.9167 | 11/12 | 125.2066 | 0.016623 |
| validation | medium_precedence | load_balanced_v1 | 0.8333 | 10/12 | 62.0190 | 0.016011 |
| validation | medium_precedence | order_aware_lns_v2 | 0.9167 | 11/12 | 64.7864 | 0.500286 |
| validation | medium_resource | deadline_assignment_milp_v2 | 1.0000 | 12/12 | 115.4131 | 0.668122 |
| validation | medium_resource | deadline_load_balanced_v2 | 1.0000 | 12/12 | 134.0512 | 0.017847 |
| validation | medium_resource | load_balanced_v1 | 0.8333 | 10/12 | 64.9605 | 0.016314 |
| validation | medium_resource | order_aware_lns_v2 | 1.0000 | 12/12 | 83.9568 | 0.641376 |
| validation | small_sparse | deadline_assignment_milp_v2 | 1.0000 | 12/12 | 39.7156 | 0.014556 |
| validation | small_sparse | deadline_load_balanced_v2 | 1.0000 | 12/12 | 41.4520 | 0.004018 |
| validation | small_sparse | load_balanced_v1 | 1.0000 | 12/12 | 31.5457 | 0.003994 |
| validation | small_sparse | order_aware_lns_v2 | 1.0000 | 12/12 | 35.2069 | 0.163867 |

`schedule_infeasible` remains a method failure, not proof of global infeasibility. The assignment MILP remains assignment-only and its gap is not a joint scheduling gap. No v2 frozen result was used for selection.
