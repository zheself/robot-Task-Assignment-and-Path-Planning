# A1 solver/verifier evidence card

Version: `a1-foundation-v1`  
Evidence label: `SIM_GEOMETRIC`  
Gate date: 2026-08-09

## Included

- Stable reason-coded analytical robot–segment edge mask.
- Deterministic list scheduling for fixed assignment and robot order.
- Post-hoc verification of coverage, declared edge feasibility, process duration,
  robot/segment/resource windows, robot non-overlap, proxy transition time, precedence, parent-curve
  order, same-robot handoff and shared-resource capacity.
- Greedy edge-cost, load-balanced greedy, repeated-slot Hungarian plus stable
  topological ordering, and small-instance assignment MILP baselines.
- Uniform result fields: status, runtime, objective, best bound, MIP gap,
  diagnostics and optional plan.

## Solver semantics

The MILP minimizes a maximum assigned proxy-load variable with a deterministic
tie breaker. Its output is then passed to the same list scheduler and verifier
as every other baseline. `optimal` therefore means optimal for this assignment
MILP formulation under the configured tolerance—not optimal continuous
multi-robot motion, not a complete resource-constrained project schedule, and
not a physical process optimum.

The configured default is 30 seconds and zero requested relative gap. Raw
SciPy status/message, incumbent objective, dual bound and returned MIP gap are
preserved when available. A limit without an incumbent remains `limit`; it is
not silently reported as feasible.

## Foundation matrix

The automated smoke matrix uses three deterministic fixture-derived cases:

1. four independent continuous segments and two robots;
2. two segments/two robots sharing a capacity-one resource;
3. an intentionally infeasible capability mask.

Each case runs all four methods. This matrix validates plumbing and failure
reporting only. A2 must create the larger versioned benchmark, immutable grouped
splits, difficulty families and paper-level comparison protocol.

## Explicit exclusions

- no verified IK or joint-space reachability;
- no continuous robot–robot, robot–fixture or swept-volume collision guarantee;
- no controller timing or real factory cycle-time evidence;
- no stress, contact, plasticity or process-quality model;
- no GNN/RL result and no method-superiority claim.
