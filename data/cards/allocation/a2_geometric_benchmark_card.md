# A2 leakage-safe geometric benchmark pilot card

Version: `a2-geometric-benchmark-v1` (`PILOT_FROZEN`)  
Generator: `a2-continuous-generator-v1`  
Evidence: `SIM_GEOMETRIC`  
Master seed: `260809`

## Purpose

This engineering pilot supports the first-paper allocation/scheduling pipeline before
verified CAD or production curves are available. It supplies continuous curve
segments, abstract robot layouts, declared capabilities/tools, precedence,
handoff, time windows and shared-resource proxies. It is not execution data and
is too small to support paper-level GNN fitting or statistical claims.

## Geometry and instances

- Curve families: line, circular arc, uniform cubic B-spline and closed loop.
- Every curve is a sampled continuous polyline in SI metres and is split only
  at shared endpoints into ordered process segments.
- Robot bases are programmatically placed around a synthetic workcell. The A1
  analytical reach oracle remains the only reachability evidence.
- Two variants in each non-stress task group share the same workpiece geometry,
  layout and robots but receive independently seeded declared constraints.
- Frozen v1 contains 19 instances and 95 method runs:
  - train: 4 groups / 8 instances, 2–4 robots, 8–24 segments;
  - validation: 2 groups / 4 instances, 2–5 robots, 10–28 segments;
  - frozen test: 3 groups / 6 instances, 3–6 robots, 16–40 segments;
  - stress: 1 group / 1 instance, 8 robots, 64–80 segments.

## Split and leakage policy

The split unit is a task group. Workpiece ID, layout ID, task-group ID and every
parent-curve ID are audited to occur in exactly one split. Sibling variants are
therefore never divided across train, validation or frozen test. Manifest and
canonical instance SHA-256 values are verified on every benchmark run.

Allowed access is explicit:

- train: `fit_only`;
- validation: `selection_only`;
- frozen test and stress: `evaluation_only`.

Changing generator assumptions, seeds, split membership, objective weights or
budgets requires a new benchmark/manifest version and a complete rerun.
The paper-scale benchmark must therefore be `v2` or later; this pilot must not
be overwritten or silently expanded.

## Frozen baseline protocol

Methods are greedy, load-balanced greedy, Hungarian plus deterministic order,
assignment MILP plus deterministic schedule, and deterministic assignment LNS.
MILP receives 10 seconds and requested relative gap 0. LNS receives 100
iterations and seed 0. Every returned plan is rechecked by the same verifier.

`assignment_mip_gap` concerns only the assignment formulation. It does not
certify joint assignment/scheduling/path optimality. `best_observed_relative_gap`
is a within-instance descriptive comparison, not a certified optimality gap.

## Known failures and exclusions

The frozen benchmark deliberately retains schedule-infeasible outputs caused by
the current assignment-first methods under tight time/resource constraints.
They remain in the failure library; they must not be deleted or relabelled after
seeing a future GNN result.

There is no verified IK, continuous collision check, CAD, controller timing,
factory cycle time, stress/contact/plasticity model or physical-quality label.
No result from this benchmark supports a real deployment or sim-to-real claim.

## Remaining A2 gate

Only eight train instances, four validation instances and six frozen-test
instances exist in pilot v1. Before A3, a paper-scale protocol must preregister
more independent task groups per split and per difficulty cell, declare how
generator-infeasible versus solver-failed cases are handled, and repeat all
baseline/failure reporting. Pilot v1 is for engineering regression only.
