# A4a Pair-Pointer warm-start development-pilot result

Decision: **`STOP_A4_LEARNING_WARM_START_BRANCH`**

Integrity override: **`A4A_PRIMARY_EVALUATION_INVALID_STOP`**

Evidence: **SIM_GEOMETRIC development-only**

The preregistered 1.0 s primary view is invalid because of two implementation defects documented in the integrity audit. The following table is the unaffected, descriptive 50-iteration view; it is not a replacement confirmatory endpoint.

## Descriptive fixed-50-iteration view

| Initializer + identical repair | Coverage | Init median (s) | Repair median (s) | E2E median (s) | Assignment retention |
|---|---:|---:|---:|---:|---:|
| cold_start | 0.625 | 0.0000 | 0.7502 | 0.7502 | 1.000 |
| hybrid_assignment_milp | 0.729 | 0.4570 | 0.7491 | 1.2164 | 0.808 |
| hybrid_load_balanced | 0.771 | 0.0238 | 0.7548 | 0.7796 | 0.826 |
| matched_static | 0.677 | 0.0060 | 0.7504 | 0.7564 | 0.699 |
| pair_pointer | 0.774 | 0.4132 | 0.7508 | 1.1286 | 0.753 |

## Gate

- `at_least_two_seed_wins_vs_static`: False
- `initializer_retention_at_least_half`: True
- `integrity`: True
- `mean_pair_vs_static_at_least_one_group`: False
- `pair_repair_above_raw`: False
- `practical_advantage_vs_load_balanced`: False

## Mechanical integrity

- `actual_rows`: 5616
- `complete_matrix`: True
- `expected_rows`: 5616
- `group_count`: 24
- `group_count_ok`: True
- `three_seed_pairing`: True
- `zero_forbidden_access`: True
- `zero_mask_atomicity_failure`: True

## Semantic integrity override

- MILP adapter used a load-balanced state on 28 primary rows whose MILP assignment had a schedule-infeasible result; those rows are not a valid MILP-incumbent warm start.
- The 1.0 s evaluator marked an incumbent unsuccessful when repair ended slightly after the wall-clock budget, even when first_feasible_time_s was already within budget; the incumbent-at-budget should have been retained.

## Boundary

This is a development-only geometric proxy experiment. It does not alter A3/A3.5, establish real-robot or collision safety, use RL/path planning/physical modelling, or authorise a new frozen benchmark.
