# A4b evaluator and ordinary LNS/ALNS preregistration

Status: **FROZEN BEFORE `a4b_neural_lns_dev_v1` GENERATION**  
Date: 2026-08-13  
Configuration: `configs/allocation/a4b_neural_lns_dev_v1.json`  
Evidence: **SIM_GEOMETRIC — development only**

## A4b-0 semantic specification

All timing uses `time.monotonic_ns`. Search start precedes initializer start.
Each initializer completion, candidate evaluation and verified incumbent has
an absolute monotonic timestamp and a relative end-to-end timestamp. The
snapshot at budget `b` is the lowest-objective verified incumbent whose own
timestamp is no later than `b`. Function return time never deletes an earlier
incumbent. An incumbent first observed after `b` is not backdated.

Initializer time is charged. If initializer completion is later than `b`, the
snapshot is `initializer_timeout` even if the initializer later returns a
plan. All failed snapshots remain in coverage denominators. One maximum-budget
trace supplies consistent snapshots at 0.5, 1.0 and 3.0 seconds.

Every initializer record contains `requested_initializer`,
`actual_initializer`, `solver_status`, `has_true_incumbent`, `fallback_used`,
`fallback_reason`, `initializer_plan_hash`, verifier feasibility/failure, and
start/completion timestamps. A MILP assignment incumbent is preserved when
scheduling rejects it. A no-incumbent MILP remains no-incumbent. Any explicitly
allowed fallback carries its actual method label; A4b main search allows no
initializer fallback.

The old A4a outputs, closure and invalid primary statistics are immutable and
will not be recalculated or overwritten.

## Shared controlled search

All controlled methods start from `hybrid_load_balanced`. They share:

- complete atomic allocation units from the A1 handoff contract;
- destroy ratios 0.10/0.25/0.40 cycled by iteration;
- hard-feasible regret/load reinsertion, capped at 256 candidate evaluations
  per neighborhood;
- unchanged A1 `build_schedule` and `verify_plan`;
- verified-first simulated-annealing acceptance;
- the same seeds 1201/1907, 30-iteration limit and 0.5/1.0/3.0 s end-to-end
  cutoffs;
- infeasible current states allowed, but never retained as best feasible;
- restart to the best incumbent after 12 non-improving iterations;
- one CPU thread and identical Slurm resources;
- every failure in the denominator.

The controlled methods are fixed random-destroy LNS, round-robin handcrafted
LNS, train-selected best single handcrafted operator and adaptive LNS/ALNS.
The existing `order_aware_lns` is retained as a fixed-iteration historical
reference and explicitly not inserted into controlled fixed-time conclusions,
because its older internal repair/trace interface is different. Small-instance
oracle destroy evaluates all registered neighborhoods under the same micro
repair budget and is only an upper bound/label audit, never a scalable baseline.

## Destroy operators

1. `random_destroy`: uniform without replacement;
2. `worst_cost_destroy`: current assigned edge cost plus completion proxy;
3. `load_imbalance_destroy`: units on overloaded robots, then unit cost;
4. `precedence_chain_destroy`: high-degree and small-gap predecessor endpoints;
5. `critical_slack_destroy`: smallest current or static time-window slack;
6. `shared_resource_conflict_destroy`: high shared-resource occupancy/duration;
7. `relatedness_destroy`: Shaw-style geometry, window, resource, robot and
   precedence relatedness around a seeded unit;
8. `compound_destroy`: round-robin union of precedence, slack, resource and
   relatedness rankings.

Every operator returns distinct atomic-unit indices. Segment-level splitting
is impossible by construction.

## ALNS rule

All eight weights start at 1. Seeded roulette selects an operator. After each
application:

```text
w_next = (1 - rho) * w + rho * max(reward, 1e-6), rho = 0.20
```

Rewards are 8 for a global best, 4 for an accepted improvement, 2 for first
feasibility, 1 for another accepted state and 0 for rejection. Acceptance is
identical outside operator selection: feasible beats infeasible; a verified
current state cannot be replaced by infeasible; improving score/surrogate is
accepted; other candidates use seeded simulated annealing with initial
temperature 5% of current scale and cooling 0.97.

## Data protocol and selection

The new seed is 842137 and prefix is `a4bnlsd1`. Six cells each contain four
train groups and two development groups, with two variants per group: 24/12
independent groups and 48/24 instances. Only `train` and `development` exist.
`validation`, `frozen_test` and `stress` are forbidden. IDs are structurally
disjoint from v2/v3/v4, A3.5 pilot/final and A4a. Static UR5 CSV is prohibited.

Implementation note: the inherited geometric generator names its non-frozen
held-out geometry family `validation`. The A4b adapter may use that string only
as an internal geometry-family selector when generating the public
`development` split. A4b IDs, paths, records, manifest, loader and aggregation
remain `development`; no prior validation instance/path is read and no A4b
validation split is emitted.

One train group per cell selects the best single handcrafted operator by
coverage, conditional objective with failures ranked last, runtime and ID.
Development cannot change any operator, ratio, repair, seed or budget. The
independent reporting unit is `task_group_id`; variants and seeds stay within
group aggregation.

## A4b-2 record and label contract

Every step stores instance/group/cell/split; current assignment/order;
verifier/failure; current and best objective; static unit features; robot load,
completion and location; predecessor/slack/resource state; operator and destroy
set/ratio; repaired state; feasibility and objective/violation delta; repair
runtime and candidate evaluations; absolute/relative timestamp; acceptance and
incumbent update; plan, verifier, state, config and trace hashes.

Candidate label sets evaluate all eight destroy sets from the same state with
the same repair budget. They store feasible/objective improvement,
time-to-feasible, repair cost, edit distance and Pareto dominance. Their only
name is **search-generated neighborhood improvement labels**; they are not true
expert actions. Future learning should use pairwise/listwise ranking or
multi-candidate ordering rather than destroy-classification accuracy alone.

## Metrics and stop rule

Primary future metrics are verifier coverage at fixed end-to-end time,
time-to-first-feasible, objective at fixed time, normalized anytime regret and
time-to-target. Descriptive metrics include fixed-iteration coverage,
makespan/load/travel proxy score, repair success, evaluated neighborhoods,
destroy size, edits, runtime decomposition, difficulty/scale and median/IQR.

Before any learned destroy pilot, the evaluator fixtures, replay and complete
matrix must pass. Adaptive ALNS 1.0-second development coverage must not be
below fixed random LNS. If it is lower, the ordinary search foundation is not
accepted; no extra budget or neural training may conceal the failure.
