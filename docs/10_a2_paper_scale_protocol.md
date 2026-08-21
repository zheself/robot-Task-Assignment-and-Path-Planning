# A2 paper-scale v2 preregistration

Status: protocol `PREREGISTERED_BEFORE_V2_GENERATION`; outcome `FROZEN_FAILED`  
Evidence: `SIM_GEOMETRIC`  
Configuration: `configs/allocation/benchmark_v2.json`

## 1. Experimental unit and sample size

The independent unit is `task_group_id`, not an individual constraint variant.
Variants from one group share workpiece geometry, robot layout and parent curves;
they always remain in one split and are averaged within group for inferential
statistics.

The frozen design contains:

| split | cells | independent groups | variants | instances | allowed use |
|---|---:|---:|---:|---:|---|
| train | 4 | 96 | 2 | 192 | fitting only |
| validation | 4 | 24 | 2 | 48 | selection only |
| frozen test | 6 | 72 | 2 | 144 | final evaluation only |
| stress | 4 | 24 | 1 | 24 | diagnostic evaluation only |
| total | 18 | 216 | — | 408 | — |

No sample may move between splits. Pilot-v1 instances are not merged into v2.

## 2. Difficulty cells

Train/validation cover small sparse, medium balanced, precedence-dense and
resource-dense in-domain cells. Frozen test contains separate IID-small and
IID-medium cells plus unseen dense-precedence, resource-bottleneck, tight-window
and scale-OOD cells. Stress contains extreme scale, tight-window and resource
cells plus an explicit unavailable-capability negative control.

Robot count, segment count, precedence density, shared-resource density,
time-window tightness and variants/group are fixed in `benchmark_v2.json`.
Changing any range, seed, count, objective weight or solver budget creates v3.

## 3. Feasible/infeasible policy

“Proxy admissible” has a deliberately narrow definition: the instance passes
the A0 schema and each atomic assignment unit has at least one robot allowed for
all its member segments by the frozen A1 edge oracle. It does **not** prove that
a joint assignment and schedule exists.

- Every non-negative-control cell is `admissible_required`; generation aborts
  if proxy admissibility fails.
- `designed_edge_infeasible` injects one unavailable required capability and
  must fail proxy admissibility by construction.
- `schedule_infeasible` means a particular assignment-first method followed by
  the deterministic scheduler produced no valid schedule. It is retained as a
  method failure and never relabelled as global instance infeasibility.
- Schema failure, hash mismatch, split leakage, unexpected solver error,
  unbounded result or verifier failure is an experiment-integrity failure.

Quality metrics are never imputed for failed plans. Feasibility is reported on
all preregistered groups, while conditional quality comparisons use only
pairwise jointly verified outputs and disclose their reduced sample size.

## 4. Solver protocol

Every instance runs greedy, load-balanced greedy, Hungarian plus ordering,
assignment MILP and deterministic assignment-LNS. MILP receives 3 seconds with
requested relative gap 0. LNS receives 100 iterations and seed 0. All plans pass
through one frozen verifier. `assignment_mip_gap` only describes the assignment
MILP; it is not a joint scheduling/path optimality gap.

## 5. Statistical protocol

Primary reporting is on frozen test, stratified by difficulty cell.

1. Verified-plan rate uses all groups and reports group-cluster bootstrap 95%
   confidence intervals.
2. Makespan, load variance, travel/setup and the preregistered weighted proxy
   score are compared only on pairwise jointly verified variants, averaged to
   task-group level.
3. Paired score differences use 5,000 group bootstrap resamples, seed 260811,
   and Wilcoxon signed-rank tests when at least five nonzero group pairs exist.
4. Holm correction is applied across the preregistered comparisons against
   assignment MILP and deterministic LNS.
5. Runtime uses every run and reports median and interquartile range. No failed
   plan is awarded an artificial quality score.
6. Stress is descriptive only and excluded from claims of average superiority.

`best_observed_relative_gap` remains descriptive and is not called an
optimality gap.

## 6. A2 acceptance gate

A2 v2 passes only if schema, leakage and hash audits have zero failures; no
unexpected solver/verifier status occurs; candidate-plan coverage is at least
90% for train and 80% for validation; every non-negative frozen cell has at
least 50% coverage; and the designed-infeasible cell is detected at 100%.

Failure of a threshold is reported as an A2 negative result and continues to
block A3. Thresholds are not changed after observing v2 results.

## 7. Evidence boundaries

All results remain `SIM_GEOMETRIC`. The benchmark provides no verified IK,
continuous collision safety, CAD, real controller timing, physical quality,
factory deployment or sim-to-real evidence. It contains no RL transitions and
does not use the UR5 static positioning CSVs.

## 8. Recorded outcome

The manifest was frozen as
`c9f532e72db33324e8f84677001cab028eb9a8e5634857998375ae7ed99f0842`.
The gate failed only the minimum frozen-cell candidate-coverage check; all
integrity, train/validation coverage and negative-control checks passed. Exact
results are in `reports/phase1_allocation/a2_paper_v2_results.md` and diagnosis
is in `docs/11_a2_v2_failure_analysis.md`. No threshold was revised.
