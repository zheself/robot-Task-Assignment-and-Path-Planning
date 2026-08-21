# A2 paper-scale v3 preregistration

Status: `PREREGISTERED_BEFORE_V3_GENERATION_OR_EVALUATION`; outcome `FROZEN_FAILED`  
Evidence: `SIM_GEOMETRIC`  
Configuration: `configs/allocation/benchmark_v3.json`

## Decision basis

Frozen v2 failed its OOD candidate-coverage gate. Development v1 and v2 then
used only the 192 v2 train and 48 v2 validation instances. The selected revision
combines the unchanged fixed-topological scheduler with a deterministic
precedence/resource-aware minimum-slack scheduler and chooses the lower frozen
proxy score when both are feasible. Order-aware LNS reconstructs both orders and
does not require a feasible greedy schedule.

No v2 frozen-test or stress instance may select the v3 method, budget or
threshold. V2 frozen remains disclosed diagnostic evidence only.

## Frozen v3 design

- 408 instances, 216 independent task groups and 18 difficulty cells.
- Train 192/96 groups; validation 48/24; frozen test 144/72; stress 24/24.
- The factor ranges and acceptance thresholds remain identical to v2.
- Master seed is `260821`; task-group identifiers are v3-prefixed. Geometry,
  layouts, curves and instance identifiers are therefore new and cannot overlap
  v2 groups.
- Six designed edge-infeasible stress instances remain negative controls.

## Methods and budgets

The five frozen v1 baselines are retained. Three disclosed A2 revisions are
added: hybrid load-balanced scheduling, hybrid assignment-MILP scheduling and
order-aware assignment-LNS. MILP budget remains 3 seconds with requested gap 0;
both LNS methods receive 100 iterations and seed 0. No method receives a
different verifier or acceptance rule.

The assignment MILP is still assignment-only. Neither its bound nor gap proves
joint scheduling optimality. The new schedulers are deterministic proxy timing,
not motion planning or collision-safety guarantees.

## Statistics and acceptance

The independent unit remains `task_group_id`. Verified-plan coverage uses all
groups with clustered bootstrap intervals. Quality is compared only on jointly
verified group pairs, with 5,000 bootstrap resamples, Wilcoxon tests and Holm
correction. Stress remains descriptive.

The gate is unchanged: zero integrity failure, train coverage at least 90%,
validation at least 80%, every non-negative frozen cell at least 50%, and 100%
negative-control detection. Thresholds, cells or budgets cannot change after
the manifest is generated. A failure remains an A2 negative result and blocks
A3.

## Recorded outcome

The frozen manifest SHA-256 is
`a039238d50ac10a6ecc44d6001f07aa75cbc3169c9ac5c8c1800a1794742ae12`.
All integrity, train/validation, five frozen-cell and negative-control checks
passed. OOD-scale candidate coverage was 11/24 (45.8%) and therefore failed the
unchanged 50% minimum. The gate remains failed and A3 remains blocked. Exact
results are recorded in `reports/phase1_allocation/a2_paper_v3_results.md`.
