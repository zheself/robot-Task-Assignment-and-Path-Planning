# A4b evaluator and ordinary LNS/ALNS development results

Date: 2026-08-13  
Protocol: `a4b_neural_lns_dev_v1`  
Evidence: **SIM_GEOMETRIC development-only**  
Result status: **HOLD_A4B_LEARNED_DESTROY_TRAINING**

> **Post-result semantic erratum (2026-08-18).** The v1 fixed-time rows remain
> descriptive and retain the corrected anytime-cutoff meaning. The v1
> fixed-iteration rows do not constitute fixed-iteration evidence: all 192
> controlled traces stopped at the three-second time limit, zero completed 30
> neighborhoods, and the median trace completed only two. The runner then
> called `best_at_iteration` on those truncated traces. The train-selected
> single-operator result used the same hybrid 10-iteration/one-second pattern
> and is likewise not valid selection evidence. These rows are preserved, not
> recomputed or deleted. Recovery protocol `a4b_ordinary_lns_dev_v2` is frozen
> in `docs/31_a4b_ordinary_lns_recovery_preregistration.md`; no v1 result is
> used to relax its gates.

## 1. State audit and immutable boundaries

The prompt's A3, A3.5 and A4a history agrees with the sealed project files.
The only pre-execution status difference was that `README.md`, `ROADMAP.md` and
the execution-status documents still described A4 as unauthorised/not started.
This request explicitly authorised a separate A4b protocol, so those status
statements are superseded only for A4b.  The following closures are unchanged:

- A3: `A3_FINAL_FAILED_BASELINE_FLOOR`; v4 was not accessed or rerun;
- A3.5: `A3_5_DECODER_HYPOTHESIS_SUPPORTED`; the sealed final was not accessed
  or rerun and no checkpoint was changed;
- A4a: `A4A_PRIMARY_EVALUATION_INVALID_STOP` and
  `STOP_A4_LEARNING_WARM_START_BRANCH`; validation was not patched or rerun.

The repository has an empty `.git/` directory rather than usable Git metadata.
Consequently this run records SHA-256 source/config/test/manifest hashes, but
cannot honestly provide a commit ID or clean-worktree assertion.

## 2. Evaluator defects, reproductions and fixes

The A4a MILP-label defect originated in
`src/safe_residual_rl/allocation/warm_start.py`: the
`hybrid_assignment_milp` branch substituted `_load_balanced_state` when the
solver returned no scheduled plan, while retaining the MILP dictionary key.
The minimal semantic reproduction is a `SolverResult` with no plan/incumbent
plus an explicit fallback; the corrected adapter reports requested MILP,
actual load-balanced, `fallback_used=true`, a reason and the actual verifier
state.  A MILP assignment incumbent rejected by scheduling is retained as
`hybrid_assignment_milp_assignment_incumbent`, not silently replaced.

The A4a cutoff defect originated in `scripts/run_a4_warm_start_pilot.py`, where
an already verified result was changed to failure when total function return
time exceeded the budget.  The corrected evaluator records monotonic absolute
and relative timestamps for each incumbent and takes the best incumbent whose
own timestamp is at or before the cutoff.  A late function return does not
erase it; a first incumbent after the cutoff is not backdated.  Initializer
time is included, and initializer completion after a cutoff is
`initializer_timeout`.

A post-smoke audit also found a new ordinary-search provenance defect:
heuristic assignments were not retained when the scheduler rejected them.
That produced an all-unassigned state and made partial neighborhood repair
incapable of recovering units outside the destroy set.  The affected process-
successful jobs `942267` and `942271_[0-5]` are retained under
`outputs/.../invalid_runs/pre_heuristic_assignment_incumbent_fix/` and are not
method evidence.  `finalize_assignment` and the heuristic callers now retain
the true assignment incumbent.  The minimal regression constructs an
edge-feasible two-unit assignment whose precedence/window combination is
unschedulable and verifies that the complete assignment survives adaptation.

The A4b initializer record contains requested/actual initializer, solver
status, true-incumbent flag, fallback flag/reason, initializer plan/state hash,
verifier feasibility/failure and monotonic start/completion times.  A4b main
search allows no fallback.  Nothing in this repair recalculates A4a.

## 3. Ordinary LNS/ALNS implementation

Every controlled arm starts from `hybrid_load_balanced` and uses the same
complete atomic units, 0.10/0.25/0.40 destroy-ratio cycle, capped 256-candidate
regret/load reinsertion, A1 scheduler/verifier, verified-first simulated-
annealing acceptance, seeds 1201/1907, 30-iteration ceiling and 0.5/1.0/3.0 s
end-to-end cutoffs.  One maximum trace supplies all cutoffs.  Failures remain
in the denominator.

The eight operators are:

1. uniform random without replacement;
2. worst assigned edge/completion cost;
3. units on the most overloaded robot;
4. precedence-chain/critical-predecessor endpoints;
5. smallest time-window slack;
6. high shared-resource occupancy/duration;
7. Shaw-style geometry/window/resource/robot/precedence relatedness;
8. a compound union of precedence, slack, resource and relatedness rankings.

All outputs are distinct atomic-unit indices, so an indivisible curve unit
cannot be split.  The implemented arms are fixed random LNS, handcrafted
round-robin, train-selected single handcrafted operator, adaptive ALNS and a
small-instance oracle destroy upper bound.  Existing `order_aware_lns` is
reported only as a fixed-iteration historical reference because it does not
share the new repair/trace interface.

ALNS uses seeded roulette over weights initially equal to 1 and
`w_next=(1-rho)w+rho*max(reward,1e-6)`, with `rho=0.20`.  Rewards are 8/4/2/1/0
for global best, accepted improvement, new feasibility, other accepted state
and rejection.  Restart is to best incumbent after 12 non-improving
iterations.  A verified current state is never replaced by an infeasible one.

## 4. Data and selection

The new master seed is 842137 and every generated identity contains prefix
`a4bnlsd1`.  The sealed manifest contains:

| split | task groups | variants/group | instances |
|---|---:|---:|---:|
| train | 24 | 2 | 48 |
| development | 12 | 2 | 24 |

Each split covers `iid_small`, `iid_medium`, `dense_precedence`,
`resource_bottleneck`, `tight_windows` and `scale`.  There is no A4b
`validation`, `frozen_test` or `stress` split.  IDs are checked against
v2/v3/v4, A3.5 pilot/final and A4a prefixes; group, workpiece, layout and parent
curve sets do not cross train/development.  All records are `SIM_GEOMETRIC`;
static UR5 CSVs are prohibited.

The inherited generator internally calls its non-frozen held-out geometry
family `validation`; the A4b adapter used that name only as a geometry-family
alias while emitting new public `development` IDs, paths and manifest records.
It did not read a prior validation corpus and produced no A4b validation split.

Only one train group per cell selected the single handcrafted operator.
`relatedness_destroy` was the only operator with 12/12 train-selection
coverage; each other single operator had 11/12.  Development did not change
that selection.

## 5. Development results

The independent reporting unit is `task_group_id`; two variants and two search
seeds remain inside each of the 12 development groups.  The controlled results
are descriptive, not paper/frozen evidence.

| E2E budget | random coverage | round-robin | train-selected | ALNS |
|---:|---:|---:|---:|---:|
| 0.5 s | 62.5% | 62.5% | 62.5% | 62.5% |
| 1.0 s | 62.5% | 62.5% | 62.5% | 62.5% |
| 3.0 s | 62.5% | 66.7% | 62.5% | 62.5% |

At 1 s, conditional mean group objective was 118.633 for random, 118.394 for
round-robin, 118.505 for train-selected relatedness and 118.633 for ALNS.
These conditional means must not be compared as unconditional quality when
coverage differs.  Median incumbent timestamps were 0.0244, 0.0242, 0.0281
and 0.0282 s respectively; they mostly reflect the shared feasible initializer.

The 1 s cell coverages were identical across the four controlled methods:
100% iid-small, 100% iid-medium, 50% dense-precedence, 75% resource-
bottleneck, 25% tight-windows and 25% scale. The historically reported “fixed
30 iterations” values (66.7% for round-robin and 62.5% for random,
train-selected and ALNS) came from truncated three-second traces and are
invalid as iteration-controlled evidence under the erratum above. The valid
one-second fixed-time gate remains a tie between ALNS and random; it has not
demonstrated an improvement.

The legacy `order_aware_lns` fixed-30 reference reached 66.7% instance-row
coverage, including 50% on scale, but is not an anytime-controlled comparison:
it uses its own initializer/repair loop and has no incumbent-event trace.  The
small-instance oracle destroy arm ran only on ten eligible instance-seed rows
and had 100% cutoff coverage; it is a micro-neighborhood upper bound/label
check, not a scalable baseline or an ALNS superiority result.

The result matrix has 1,260 metric rows, including 1,212 anytime-compliant
controlled/oracle rows, 202 traces and 12 independent development groups.
There are 424 repeated cutoff/iteration snapshots with no feasible incumbent.
Across 1,057 controlled neighborhood steps, 625 repair outcomes were verified;
failed candidates comprise 399 precedence and 33 time-window failures.  No
post-fix `initializer_incomplete` failure remains.  All traces terminated at
the 3 s end-to-end cap.

Normalized primal integral/anytime regret and time-to-target were named in the
preregistration but their target/reference and failure penalty were not fixed
in the configuration before generation.  They are therefore not invented
post hoc or promoted to primary results.  Likewise, v1 records initializer and
repair time plus end-to-end timestamps, but verifier work inside candidate
repair is not separately timed from repair.  These are protocol gaps to close
before a learned method is compared.

## 6. A4b-2 records and replay

Traces store static unit features; current assignment/order; verifier and
failure state; current/best objective; robot load/completion/location;
precedence/window/resource state; operator, unique destroy set and ratio;
candidate state; feasibility and objective/violation change; repair runtime
and candidate count; acceptance/incumbent update; monotonic timestamps; and
plan, verifier, state, config and trace hashes.

Twelve train-only state records evaluate all eight operators, producing 96
candidate outcomes under the identical repair cap of 256.  They contain
feasible/objective improvement, time-to-feasible, repair cost, edit distance
and Pareto dominance.  The label is exactly **search-generated neighborhood
improvement labels**, never “expert action”.  Of 96 candidates, 21 are
verified; the failures are 67 precedence and 8 time-window cases.

Re-execution audits reproduced 24 controlled traces (six cells by four arms),
148 recorded repair transitions, and all 96 label candidates from recorded
before-state, destroy set, seed and
repair cap, matching candidate state hashes, candidate counts and feasibility.
This interface supports future pairwise/listwise unit ranking and
autoregressive no-replacement selection.  No neural checkpoint exists.

## 7. Tests and Slurm evidence

- A4b targeted matrix: 34 passed; A4b plus affected A1 solver regressions:
  52 passed;
- full non-frozen regression: 169 passed, one pre-existing Gymnasium warning;
- excluded from that regression: A2 benchmark generation, A2 v2/v4, A3 final
  and A3.5 sealed-final test files, to avoid old frozen paths;
- corrected CPU smoke `942372`: `COMPLETED`, exit `0:0`, `sist-cpu-16`, 9 s,
  one CPU and 8 GiB;
- corrected development array `942373_[0-5]`: every task `COMPLETED`, exit
  `0:0`, `sist-cpu-16`, one CPU/8 GiB; elapsed 2:06--4:20.

Config SHA-256 is
`4e6a52a3bb4374a895131af5d35028a98bb3a985563c419d2dec1694995766fe`;
manifest semantic hash is
`b08c5d693ce99323757289e026aea311956e4531e2722b5df268910ba8a186ed`;
result hash is
`4b3616237a98062e819108bcd58678cb321993444cb2c952168f10820808f23d`.
The complete per-file source/test/Slurm hashes are embedded in
`outputs/phase1_allocation/a4b_neural_lns_dev_v1/summary.json`.

## 8. Gate decision

A4b-0 passes: the two A4a defects and the additional heuristic-assignment
defect have reproductions; provenance and cutoff semantics pass; A4a is
untouched.  A4b-2's schema, split guards, equal-budget candidate outcomes and
replay pass.

A4b-1 is a reproducible ordinary-search implementation and its minimum ALNS
coverage rule passes, but it is **not yet a reliable empirical foundation for
starting learned-destroy training**.  ALNS merely ties random coverage,
recovers none of the shared initializer failures, and is not better on
conditional 1 s objective; round-robin alone recovers a small additional share
by 3 s/10 iterations.  In addition, the normalized-anytime target/penalty and
fine timing decomposition were not fully frozen.

Therefore the action is **HOLD_A4B_LEARNED_DESTROY_TRAINING**.  Do not train a
GNN scorer or autoregressive destroy pointer yet.  The next authorised step
should first freeze an A4b follow-up development protocol that (1) defines the
primal-integral reference, failure penalty and time-to-target; (2) separates
guidance, repair, scheduling and verifier time; and (3) requires ordinary ALNS
to show non-degenerate operator adaptation or feasibility recovery on train-
only counterexamples before any learned comparison.  It must still use new
development IDs and may not generate a frozen benchmark without a later,
separate decision.
