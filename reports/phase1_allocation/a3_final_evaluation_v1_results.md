# A3 one-time frozen evaluation result

Evidence: **SIM_GEOMETRIC**. This is the single preregistered A3 frozen evaluation.

- Result class: **A3_FINAL_FAILED_BASELINE_FLOOR**.
- A3 minimum final gate: **FAILED**.
- Learned overall verified coverage: 0.5671296296296297.
- Best context/weak overall coverage: 0.5555555555555556.
- Required wording: The fixed learned model failed the absolute or weak-baseline feasibility floor; stop the learned A3 branch without retuning on v4. 

## Integrity and gate checks

- PASS — `all_source_and_checkpoint_hashes_match`
- PASS — `all_144_frozen_instances_evaluated_by_all_three_checkpoints_and_all_eight_baselines`
- PASS — `complete_stress_matrix`
- PASS — `zero_schema_or_manifest_failures`
- PASS — `zero_nan_or_unexpected_exceptions`
- PASS — `hard_mask_and_atomic_units_never_violated`
- PASS — `designed_edge_infeasible_detection_rate_equals_one`
- PASS — `zero_witness_hash_or_verifier_failures`
- PASS — `absolute_iid_small`
- PASS — `absolute_iid_medium`
- FAIL — `absolute_ood_dense_precedence`
- PASS — `absolute_ood_resource_bottleneck`
- FAIL — `absolute_ood_tight_windows`
- FAIL — `absolute_ood_scale`
- PASS — `overall_not_below_best_context`
- PASS — `relative_context_iid_small`
- PASS — `relative_context_iid_medium`
- FAIL — `relative_context_ood_dense_precedence`
- PASS — `relative_context_ood_resource_bottleneck`
- PASS — `relative_context_ood_tight_windows`
- PASS — `relative_context_ood_scale`

## Learned per-difficulty coverage

| cell | coverage |
|---|---:|
| iid_small | 0.9722 |
| iid_medium | 0.9028 |
| ood_dense_precedence | 0.4028 |
| ood_resource_bottleneck | 0.5972 |
| ood_tight_windows | 0.3750 |
| ood_scale | 0.1528 |

## Strong-baseline paired coverage

| reference | difference [95% CI] | Holm p |
|---|---:|---:|
| hybrid_load_balanced | -0.1204 [-0.1875, -0.0556] | 1 |
| hybrid_assignment_milp | -0.1690 [-0.2384, -0.0995] | 1 |
| order_aware_lns | -0.2245 [-0.2963, -0.1551] | 1 |

## Boundary

The selected model is an edge-MLP, not a GNN. Results concern only the A1 geometric/timing proxy. They do not establish motion-level collision safety, robot execution, real production, physical quality or sim-to-real success.
