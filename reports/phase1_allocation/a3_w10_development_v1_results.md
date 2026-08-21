# A3 W10 train/validation development results

Evidence: **SIM_GEOMETRIC**. Frozen-test and stress were not accessed.

- Development gate: **PASSED**.
- Matrix: 3 families × 3 seeds; train 192, validation 48.
- Validation-selected family: **edge_mlp**.

## Registered checks

- PASS — `all_nine_registered_shards_completed`
- PASS — `all_shard_provenance_consistent`
- PASS — `all_shards_train_validation_only`
- PASS — `all_metrics_finite`
- PASS — `all_registered_validation_baselines_completed`
- PASS — `selected_family_uses_validation_only`

## Model-family validation aggregate

| family | coverage mean/min | assignment accuracy | conditional score | median full-pipeline runtime s | best epoch / epochs run | failures |
|---|---:|---:|---:|---:|---:|---:|
| edge_mlp | 0.958/0.958 | 0.602 | 86.5677 | 0.012760 | 18.7/28.7 | 6 |
| hetero_gnn | 0.896/0.875 | 0.611 | 87.8063 | 0.014451 | 5.3/15.3 | 15 |
| graph_transformer | 0.889/0.875 | 0.605 | 87.1710 | 0.014651 | 6.0/16.0 | 16 |

## Validation-only non-learning baselines

| method | coverage | conditional score | median full-pipeline runtime s |
|---|---:|---:|---:|
| greedy | 0.833 | 91.6468 | 0.012396 |
| load_balanced | 0.833 | 89.3068 | 0.012024 |
| hungarian | 0.854 | 87.1378 | 0.012520 |
| assignment_milp | 0.854 | 91.6515 | 0.110134 |
| deterministic_lns | 0.833 | 76.9108 | 0.277414 |
| hybrid_load_balanced | 1.000 | 106.7769 | 0.014202 |
| hybrid_assignment_milp | 0.979 | 103.8930 | 0.114440 |
| order_aware_lns | 1.000 | 85.6637 | 0.462494 |

## Interpretation boundary

Selection is based only on validation and authorizes a separate frozen-evaluation preregistration; it is not a frozen-test result. Conditional scores exclude failed candidates and must be read together with coverage. The witness is a feasible A1-proxy teacher, not an optimum or real expert. No A4 repair, motion planning, collision guarantee, physical model or RL was used.
