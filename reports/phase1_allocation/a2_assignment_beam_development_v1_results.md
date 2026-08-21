# A2 scheduler development v1 results

Evidence: `SIM_GEOMETRIC`; usage: development only.
Source manifest: `a039238d50ac10a6ecc44d6001f07aa75cbc3169c9ac5c8c1800a1794742ae12`.
Accessed splits: `train, validation`. Frozen-test and stress were not accessed.

| split | cell | method | coverage | verified/total | mean score (verified only) | median runtime s |
|---|---|---|---:|---:|---:|---:|
| train | medium_balanced | assignment_beam_sequence_v1 | 1.0000 | 48/48 | 38.1467 | 2.337122 |
| train | medium_balanced | order_aware_lns_v2 | 1.0000 | 48/48 | 37.2857 | 0.466233 |
| train | medium_precedence | assignment_beam_sequence_v1 | 0.9583 | 46/48 | 42.8830 | 3.061307 |
| train | medium_precedence | order_aware_lns_v2 | 0.9375 | 45/48 | 44.2872 | 0.518071 |
| train | medium_resource | assignment_beam_sequence_v1 | 1.0000 | 48/48 | 54.4306 | 5.989476 |
| train | medium_resource | order_aware_lns_v2 | 1.0000 | 48/48 | 54.8826 | 0.725264 |
| train | small_sparse | assignment_beam_sequence_v1 | 1.0000 | 48/48 | 28.7661 | 0.320840 |
| train | small_sparse | order_aware_lns_v2 | 1.0000 | 48/48 | 29.9532 | 0.200609 |
| validation | medium_balanced | assignment_beam_sequence_v1 | 1.0000 | 12/12 | 42.9877 | 3.296712 |
| validation | medium_balanced | order_aware_lns_v2 | 1.0000 | 12/12 | 43.5867 | 0.519094 |
| validation | medium_precedence | assignment_beam_sequence_v1 | 0.8333 | 10/12 | 44.5405 | 4.624822 |
| validation | medium_precedence | order_aware_lns_v2 | 0.9167 | 11/12 | 50.3531 | 0.616417 |
| validation | medium_resource | assignment_beam_sequence_v1 | 1.0000 | 12/12 | 52.4873 | 5.480960 |
| validation | medium_resource | order_aware_lns_v2 | 1.0000 | 12/12 | 56.3261 | 0.720110 |
| validation | small_sparse | assignment_beam_sequence_v1 | 1.0000 | 12/12 | 26.6071 | 0.249153 |
| validation | small_sparse | order_aware_lns_v2 | 1.0000 | 12/12 | 28.3514 | 0.180051 |

`schedule_infeasible` remains a method failure, not proof of global infeasibility. The assignment MILP remains assignment-only and its gap is not a joint scheduling gap. No frozen-test or stress result was used for selection.
