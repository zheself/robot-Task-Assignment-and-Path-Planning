# A4b ordinary-LNS recovery v2 preregistration

Status: **FROZEN BEFORE `a4b_ordinary_lns_dev_v2` DATA GENERATION**  
Date: 2026-08-13  
Evidence: **SIM_GEOMETRIC development-only**

## Purpose

This follow-up protocol repairs ordinary-search evidence defects discovered by
a read-only audit of `a4b_neural_lns_dev_v1`. It does not reopen A3, A3.5 or
A4a, does not overwrite A4b v1, and does not train a neural model.

The v1 fixed-time snapshots remain descriptive. Its fixed-iteration rows and
train-selected single-operator result are not used as evidence because every
controlled trace stopped on the three-second time limit before completing 30
iterations.

## Independent data namespace

- protocol: `a4b_ordinary_lns_dev_v2`;
- master seed: `915731`;
- ID prefix: `a4blnsd2`;
- train: 48 task groups, two variants per group, 96 instances;
- development: 24 task groups, two variants per group, 48 instances;
- eight/four groups per difficulty cell respectively;
- only `train` and `development`; `validation`, `frozen_test` and `stress` are
  forbidden;
- all evidence is `SIM_GEOMETRIC`; static UR5 CSV is forbidden.

## Budget semantics

Fixed-time and fixed-iteration are separate runs.

Fixed-time runs use a monotonic absolute deadline and emit snapshots at 0.5,
1.0 and 3.0 seconds. Initializer, guidance, repair, scheduling and verification
are charged to end-to-end time. A neighborhood may return after the deadline,
but no candidate first verified after the deadline is credited. Its overrun is
recorded.

Fixed-iteration runs must complete exactly 10, 20 or 30 neighborhoods. A
safety watchdog is not an experimental time budget: a watchdog interruption
marks the run incomplete and forbids an iteration snapshot.

## Shared repair v2

Every controlled method uses the same deterministic atomic-unit repair:

- all robot-local insertion positions are candidates;
- partial states are ranked by a lexicographic violation vector containing
  missing/invalid units, precedence/order deadlock, time-window lateness,
  shared-resource overlap and load;
- the same 256 candidate-evaluation cap and absolute deadline apply;
- when the cap/deadline is reached, remaining units use the same deterministic
  best structural insertion, not blind append;
- final feasibility is still decided only by the unchanged A1 scheduler and
  verifier;
- assignment and robot-local order edits are recorded separately.

## Operators and ALNS

The eight v1 operator families remain: random, worst-cost, load-imbalance,
precedence-chain, critical-slack, shared-resource-conflict, relatedness and
compound. Failure-aware operators condition on the segment/unit reported by
the scheduler diagnostic and its predecessor/resource/local-order context.

ALNS uses seeded roulette. Train-only calibration compares online and
usage-normalized segmented weight updates. Rewards are limited to a global
best, first feasibility, strict objective/violation improvement, or an accepted
previously unseen diversification state. Repeated or non-improving infeasible
states receive zero. The chosen update scheme and best single operator are
frozen before development.

## Metrics

Primary development metrics are fixed end-to-end budget verifier coverage,
time-to-first-feasible, best objective at fixed time, normalized primal
integral and time-to-target. For minimization, the normalized gap uses a
train-frozen per-cell target and reference. Before the first incumbent and for
failed runs, gap is exactly 1; values are clipped to [0, 1]. The normalized
primal integral is the time average of that step function over the cutoff.

Fixed-iteration coverage/objective is secondary and is emitted only for a
complete exact-iteration trace. The independent statistical unit is
`task_group_id`; variants and seeds remain clustered within groups, and every
failure stays in the denominator.

## Search-generated labels

States are stratified across initializer, failed-search and near-boundary
feasible search states. Candidate destroy sets are canonicalized and deduplicated.
Every candidate uses the same destroy ratio, repair cap and micro-deadline and
stores the full violation vector, assignment/order/total atomic-unit edits,
first-feasible event, timing decomposition and Pareto relations. These remain
`search-generated neighborhood improvement labels`, never expert actions.

## Stop gates

Development may run only if all of the following pass on fixtures and train:

1. exact-K runs complete exactly K neighborhoods and truncated traces cannot
   emit fixed-iteration evidence;
2. precedence, window and resource counterexamples are recoverable by shared
   repair;
3. ALNS weight movement is deterministic and favors a known-success operator;
4. ALNS is not below random LNS on train one-second and exact-30 coverage;
5. train initializer-failure recovery is nonzero;
6. label records contain either infeasible-to-feasible or strict violation
   improvement and have no duplicate destroy sets within a state;
7. replay, split isolation, hashes and forbidden-path tests pass.

Failure of a gate stops before the development array. No additional budget or
neural training may conceal the failure.

