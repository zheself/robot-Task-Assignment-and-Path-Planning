# A3.5 Feasible-Pair Pointer pilot preregistration

Status: **FROZEN BEFORE A3.5 DATA GENERATION OR TRAINING**  
Date: 2026-08-10  
Evidence: **SIM_GEOMETRIC, DEVELOPMENT ONLY**

This pilot is a new question after the immutable A3 failure. It does not alter,
repair or rerun v4. It asks whether an autoregressive action
`(atomic_allocation_unit, robot)` can improve the final un-repaired A1-verifier
coverage over the matched static decoder by conditioning on preceding choices.

## Locked data protocol

The protocol ID is `a3_5_pointer_pilot_v1`, master seed `351907`, ID prefix
`a35p1`. It generates train and validation only. The six cells are IID-small,
IID-medium, dense precedence, resource bottleneck, tight windows and scale.
Each cell contains 8 independent train groups and 4 independent validation
groups, with two variants per group: 96 train and 48 validation instances in
total. Workpiece, layout, task-group, parent-curve and instance IDs must be new
and disjoint both across splits and from every v2/v3/v4 manifest.

No `frozen_test` or `stress` directory may be generated or opened. A3.5 loaders
must reject those split names and every v2/v3/v4 instance/witness path. Historical
v4 reports are contextual evidence only and cannot select this protocol.

## Teacher and serialization

Teacher candidates are the verified hybrid assignment-MILP and order-aware LNS
incumbents under fixed budgets, with the constructive feasible witness retained
as a declared fallback. The lowest weighted proxy score among verified plans
that can be represented as atomic-unit blocks is selected. MILP receives 1 s
and zero relative gap; LNS receives 50 iterations and seed 0. Status, objective,
bound/gap, runtime, verifier result and fallback use are recorded. These are
proxy incumbents, not guaranteed optima or real expert demonstrations.

The canonical action sequence is ordered by planned start time, robot ID and
unit ID. Every prefix must satisfy the hard pair mask. Replaying the sequence
must restore its assignment and robot-local atomic-unit order exactly.

## Locked methods and budgets

Five neural variants use hidden dimension 64, two encoder layers, four heads,
dropout 0.1, AdamW, learning rate 0.001, at most 30 epochs, patience 6 and seeds
101/211/307:

1. edge-MLP + static decoder;
2. heterogeneous GNN + static decoder;
3. graph Transformer + static decoder;
4. heterogeneous GNN + Feasible-Pair Autoregressive Decoder;
5. graph Transformer + Feasible-Pair Autoregressive Decoder.

Static methods retain assignment CE plus pairwise-order loss. Pointer methods
use teacher-forced pair-selection CE with an exact `-inf` hard mask. Greedy
rollout is deterministic. No repair, fallback or best-seed reporting is allowed.
Validation also reruns hybrid assignment-MILP, order-aware LNS and hybrid
load-balanced under the fixed config budgets.

## Metrics and failure policy

The primary metric is validation verified-candidate coverage; every failure
remains in its denominator. Report overall/cell/seed coverage, teacher-forced
pair accuracy, rollout completeness, mask and atomicity violations, makespan,
load imbalance, conditional weighted proxy score, runtime and stable failure
reasons. Conditional quality cannot replace coverage.

Failures distinguish decoder dead-end, incomplete assignment,
schedule-infeasible, precedence, time-window, shared-resource and
mask/integrity failures. All raw candidates and failures are retained.

## Frozen continuation gate

Select the Pointer family using mean validation coverage, then conditional
score, pair accuracy and runtime. It passes only if all conditions hold:

- zero schema, hash, leakage, forbidden-access, hard-mask or atomicity failure;
- at least two of three seeds beat the matched static decoder seed on overall
  validation coverage;
- the three-seed mean improves by at least `1/24 = 0.041667`, one independent
  validation group equivalent;
- at least two of dense precedence, resource bottleneck and tight windows
  improve;
- no cell regresses by more than `1/4 = 0.25`, one cell-group equivalent;
- improvement appears in final verifier coverage, without repair, and complete
  runtime/dead-end evidence is reported.

Failure closes the Pointer branch and returns the first-paper work to strong
solver/LNS plus path planning. Passing means only that a new untouched final
protocol may be proposed; it does not authorise generation of a frozen
benchmark.

## Claim boundary

This is not RL, A4 repair, motion planning, collision certification, physical
modelling or real-robot evidence. The decoder is named Feasible-Pair
Autoregressive Decoder or Atomic-Unit–Robot Pair-Pointer; it is not described as
a classical Pointer Network. No outcome changes the A3 v4 failure.

Machine-readable protocol: `configs/allocation/a3_5_pointer_pilot_v1.json`  
Protocol SHA-256: `876933101dc5ed2e56984d8666dae36509481e93d6d6422ec4eed5380a7bea17`
