# A4a Pair-Pointer warm-start development-pilot closure

Date: 2026-08-11  
Evidence: **SIM_GEOMETRIC — DEVELOPMENT ONLY**  
Formal decision: **`STOP_A4_LEARNING_WARM_START_BRANCH`**  
Integrity classification: **`A4A_PRIMARY_EVALUATION_INVALID_STOP`**

## Execution and isolation

The preregistration was frozen before generation with configuration SHA-256
`a194a4483c5abd7090f330459fdd608111ff289c1f039ac879623c90ff1a393f`.
The new `a4wsp1` corpus contains 96 train and 48 validation instances in 48
and 24 independent task groups, balanced over six cells. It contains only
train/validation, has no frozen/stress directory and shares no namespace with
v2/v3/v4/A3.5. The manifest internal hash is
`cd7d451094f00be2852aec52dbb64c37277725ba5557a6c2908af93bd49c822e`.

The missing train-fitted vocabulary/normalizer metadata was recovered once
from the 96 A3.5 **train** instances as part of the immutable checkpoint
pipeline. Its vocabulary and normalizer hashes exactly match the A3.5 locks;
no A3/A3.5 validation/frozen/stress instance was used. The evaluator seal is
`4f76c31a63b4e0173c8365709a5ba639881fc49f91028065ab78bb7790f0c214`.

Targeted tests passed 21/21. The non-frozen regression passed 147 tests with
one upstream Gymnasium warning. Slurm preflight `941032` passed on
`sist-cpu-01`. The sole six-cell validation array `941035` completed all tasks
with exit code 0 and produced exactly 5,616 rows; aggregation `941120` exited
0. Every validation cell has exactly one start and one completion marker.

## Registered result and semantic integrity override

The immutable aggregation mechanically passed row-count, group-count, split,
seed-pairing, mask and atomicity checks and returned
`STOP_A4_LEARNING_WARM_START_BRANCH`. Its method gates all failed: repair did
not improve raw Pair-Pointer in the reported primary view; fewer than two
Pair seeds beat matched static; the mean one-group margin failed; and no
practical advantage over load-balanced+repair was established.

Post-run semantic audit found two evaluator defects:

1. when the assignment MILP returned a schedule-infeasible plan, its adapter
   substituted the load-balanced state. This affected 28 primary MILP rows, so
   they cannot be called MILP-incumbent warm starts;
2. fixed-time rows were marked unsuccessful if the repair call ended slightly
   after the wall-clock budget, even when a feasible incumbent had already
   been found within budget. A correct anytime evaluator must retain that
   incumbent at the cutoff.

The second defect depressed all reported 1.0 s coverage values; for example,
Pair-Pointer was reported as 62.85% although 68.06% of its rows recorded a
first feasible incumbent by 1.0 s. Load-balanced was reported as 69.79%
although its by-budget first-feasible rate was 77.08%. These post-hoc values
diagnose the defect and are not replacement confirmatory results.

Because the preregistration permits zero integrity failures, the primary
evaluation is invalid and the formal action is stop. This is not evidence that
Pair-Pointer warm-start is inferior; it is evidence that this protocol cannot
answer the question. V1 validation must not be rerun or patched.

## Unaffected descriptive fixed-work evidence

The fixed-50-iteration view is not affected by the wall-clock cutoff defect
and remains useful descriptively. The invalid MILP adapter is excluded from
method interpretation.

| Initializer + identical repair | Coverage | Median init | Median repair | Assignment retention | Mean assignment edits |
|---|---:|---:|---:|---:|---:|
| Pair-Pointer, three-seed mean | 77.43% | 0.413 s | 0.751 s | 75.34% | 7.84 units |
| matched static, three-seed mean | 67.71% | 0.006 s | 0.750 s | 69.87% | 10.04 units |
| hybrid load-balanced | 77.08% | 0.024 s | 0.755 s | 82.60% | 5.80 units |
| cold start | 62.50% | 0.000 s | 0.750 s | not meaningful | 30.79 units |

Pair-Pointer is descriptively better than matched static under equal work, but
is effectively tied with load-balanced coverage while paying about 0.39 s
more initializer cost. By cell, Pair/load coverage is 72.9/87.5% for dense
precedence, 87.5/62.5% for resource bottleneck, 47.9/50.0% for scale and
56.2/62.5% for tight windows; both reach 100% on the two IID cells. This is
heterogeneous and does not satisfy a stable engineering-advantage narrative.

Repair retained about 75.3% of Pair-Pointer assignments and changed 7.84
atomic units on average, so it did not completely erase the learned
initializer. That fact alone is insufficient: the simple load-balanced start
retained more of its assignment, matched coverage and was faster.

## Closure and next action

Do not generate an A4 frozen benchmark and do not change A3/A3.5 conclusions.
Under the registered integrity rule, stop the A4 learning warm-start branch and
return the first-paper engineering line to solver/LNS/heuristic methods, with
Pair-Pointer retained as the validated decoder ablation. A corrected warm-start
study would require explicit future authorization, a new protocol/version,
new IDs/data and two mandatory fixes: expose the true MILP assignment incumbent
and evaluate the best incumbent exactly at each end-to-end cutoff.

All findings remain geometric-proxy evidence. They establish no real robot,
factory, collision-safety, path-planning, RL, sim-to-real or process-quality
claim.
