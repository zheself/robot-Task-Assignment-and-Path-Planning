# A3.5 Feasible-Pair Pointer development pilot results

Evidence: **SIM_GEOMETRIC, development only**. No frozen-test/stress was generated or accessed.

- Pilot decision: **CONTINUE_TO_NEW_PREREGISTRATION**.
- Selected pointer: `hetero_gnn_pair_pointer`; matched static decoder: `hetero_gnn_static`.
- Mean coverage: 0.785 pointer vs 0.507 static (difference +0.278).

## Integrity and continuation checks

- PASS — `all_fifteen_shards_complete`
- PASS — `all_provenance_consistent`
- PASS — `train_validation_only`
- PASS — `zero_hard_mask_violations`
- PASS — `zero_atomicity_violations`
- PASS — `manifest_train_validation_only`
- PASS — `all_metrics_finite`
- PASS — `at_least_two_seed_wins`
- PASS — `one_overall_group_equivalent_improvement`
- PASS — `two_constraint_cells_improve`
- PASS — `no_cell_regresses_over_one_group`
- PASS — `zero_pointer_dead_ends`
- PASS — `no_repair_used`
- PASS — `final_verifier_coverage_is_decision_metric`

## Model variants

| variant | coverage mean±std | pair accuracy | completion | conditional score | runtime s | dead ends |
|---|---:|---:|---:|---:|---:|---:|
| edge_mlp_static | 0.493±0.010 | n/a | 1.000 | 80.4629 | 0.01542 | 0 |
| hetero_gnn_static | 0.507±0.039 | n/a | 1.000 | 88.6065 | 0.01747 | 0 |
| graph_transformer_static | 0.479±0.017 | n/a | 1.000 | 87.5151 | 0.01856 | 0 |
| hetero_gnn_pair_pointer | 0.785±0.010 | 0.2180 | 1.000 | 90.5247 | 0.40024 | 0 |
| graph_transformer_pair_pointer | 0.715±0.039 | 0.2097 | 1.000 | 90.4554 | 0.40559 | 0 |

## Strong validation baselines

| method | coverage | conditional score | runtime s |
|---|---:|---:|---:|
| hybrid_assignment_milp | 0.792 | 107.8312 | 0.83393 |
| order_aware_lns | 0.812 | 91.2818 | 0.48600 |
| hybrid_load_balanced | 0.771 | 112.8866 | 0.01785 |

## Registered conclusion

development-only evidence supports preregistering a new untouched final protocol; no frozen benchmark is generated

Conditional scores exclude failed candidates and cannot replace coverage. This pilot neither changes the A3 v4 failure nor establishes GNN/Pointer superiority, motion safety, real execution or physical-process improvement.
