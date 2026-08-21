# A2 scheduler development v1 results

Evidence: `SIM_GEOMETRIC`; usage: development only.
Source manifest: `a039238d50ac10a6ecc44d6001f07aa75cbc3169c9ac5c8c1800a1794742ae12`.
Accessed splits: `train, validation`. Frozen-test and stress were not accessed.

| split | cell | method | coverage | verified/total | mean score (verified only) | median runtime s |
|---|---|---|---:|---:|---:|---:|
| train | medium_balanced | beam_alns_v1 | 1.0000 | 48/48 | 40.9008 | 1.253353 |
| train | medium_balanced | hybrid_load_balanced_v2 | 1.0000 | 48/48 | 48.1676 | 0.012676 |
| train | medium_balanced | order_aware_lns_v2 | 1.0000 | 48/48 | 37.2857 | 0.468085 |
| train | medium_precedence | beam_alns_v1 | 0.9792 | 47/48 | 45.7126 | 1.653257 |
| train | medium_precedence | hybrid_load_balanced_v2 | 0.8750 | 42/48 | 62.5114 | 0.015236 |
| train | medium_precedence | order_aware_lns_v2 | 0.9375 | 45/48 | 44.2872 | 0.514192 |
| train | medium_resource | beam_alns_v1 | 1.0000 | 48/48 | 57.0045 | 3.147325 |
| train | medium_resource | hybrid_load_balanced_v2 | 1.0000 | 48/48 | 71.8094 | 0.017843 |
| train | medium_resource | order_aware_lns_v2 | 1.0000 | 48/48 | 54.8826 | 0.726771 |
| train | small_sparse | beam_alns_v1 | 1.0000 | 48/48 | 29.9393 | 0.192379 |
| train | small_sparse | hybrid_load_balanced_v2 | 1.0000 | 48/48 | 35.1796 | 0.005230 |
| train | small_sparse | order_aware_lns_v2 | 1.0000 | 48/48 | 29.9532 | 0.202488 |
| validation | medium_balanced | beam_alns_v1 | 1.0000 | 12/12 | 45.0465 | 1.774570 |
| validation | medium_balanced | hybrid_load_balanced_v2 | 1.0000 | 12/12 | 53.3188 | 0.013791 |
| validation | medium_balanced | order_aware_lns_v2 | 1.0000 | 12/12 | 43.5867 | 0.515962 |
| validation | medium_precedence | beam_alns_v1 | 0.9167 | 11/12 | 49.5479 | 2.392526 |
| validation | medium_precedence | hybrid_load_balanced_v2 | 0.8333 | 10/12 | 57.9482 | 0.016982 |
| validation | medium_precedence | order_aware_lns_v2 | 0.9167 | 11/12 | 50.3531 | 0.607853 |
| validation | medium_resource | beam_alns_v1 | 1.0000 | 12/12 | 54.5708 | 2.910608 |
| validation | medium_resource | hybrid_load_balanced_v2 | 1.0000 | 12/12 | 83.1298 | 0.019699 |
| validation | medium_resource | order_aware_lns_v2 | 1.0000 | 12/12 | 56.3261 | 0.703265 |
| validation | small_sparse | beam_alns_v1 | 1.0000 | 12/12 | 27.7395 | 0.137716 |
| validation | small_sparse | hybrid_load_balanced_v2 | 1.0000 | 12/12 | 32.3858 | 0.004761 |
| validation | small_sparse | order_aware_lns_v2 | 1.0000 | 12/12 | 28.3514 | 0.177022 |

`schedule_infeasible` remains a method failure, not proof of global infeasibility. The assignment MILP remains assignment-only and its gap is not a joint scheduling gap. No frozen-test or stress result was used for selection.
