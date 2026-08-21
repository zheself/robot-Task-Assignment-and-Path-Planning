# A2 paper-scale v2 failure analysis

## Decision

The frozen v2 experiment is complete and the preregistered A2 gate **failed**.
Do not lower the threshold, delete difficult instances, extend only favourable
budgets or reuse the observed v2 frozen split as a future unseen test.

## Evidence

All integrity checks passed: 408/408 instances conformed to schema, hashes
matched, no workpiece/layout/task-group/parent-curve crossed splits, statuses
were anticipated, negative controls were detected, and 99 repository tests
passed.

The only failed gate was per-frozen-cell candidate coverage. Coverage was 100%
on IID-small and 87.5% on IID-medium, but 37.5% on dense precedence, 54.2% on
resource bottleneck, and 12.5% on both scale and tight windows.

Of 600 failed method runs, 570 ended as `schedule_infeasible`; 30 correctly
rejected six designed edge-infeasible instances. Diagnostics are dominated by
`TIME_OR_RESOURCE_WINDOW`. Assignment MILP reached SciPy status 0 on 367/408
instances and status 1 on 35; an assignment incumbent therefore often failed
under the common downstream order/scheduler.

## Root-cause interpretation

All five baselines primarily optimize assignment, then derive robot order from
one stable global topological order. They do not jointly optimize deadline
slack, resource order and assignment. Assignment-LNS also requires its
load-balanced initial schedule to be feasible. With more curves, a tight task
is often placed behind unrelated work and misses its window. Methods therefore
fail together on many OOD instances.

This does not establish global infeasibility. `proxy_admissible` intentionally
never claimed joint schedule feasibility.

## Required work before a new A2 gate

1. Add deterministic earliest-deadline/minimum-slack precedence-aware ordering.
2. Add a small-instance joint assignment/scheduling CP-SAT or MILP baseline, or
   explicitly narrow the oracle claim if unavailable.
3. Let LNS change order/resource sequencing and remove its dependence on a
   feasible greedy initial plan.
4. Develop only on hand fixtures and v2 train/validation.
5. Preregister v3 with a new master seed and entirely new frozen groups, then
   rerun the same integrity, coverage and clustered-statistics protocol.

V2 frozen results are disclosed diagnostic evidence only. A3 remains blocked.
