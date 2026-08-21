# Industrial Robot Process Planning

This repository supports a staged master's-thesis programme:

```text
continuous process segments
  -> multi-robot allocation and scheduling
  -> per-robot geometric-kinematic path planning
  -> execution compensation and optional learning
```

The first paper focuses only on the first arrow:

> **Feasibility-Aware Graph Learning for Continuous-Process Multi-Robot Task
> Allocation and Scheduling**

It studies curve/segment decomposition, assignment, robot-local ordering,
load balance, precedence/time windows and shared-workspace scheduling. It uses
strong optimisation and heuristic baselines, a feasibility-masked heterogeneous
GNN/graph Transformer, and deterministic repair. PointNet is only an optional
curve/CAD geometry encoder.

Path planning provides first-paper edge costs and feasibility diagnostics; it is
expanded in the second stage. The existing UR5 calibration and residual-RL
module is retained for the third stage. Current SAC/TD3 results do **not** beat
the projected supervised prior, so RL is not assumed to be a thesis or paper
contribution.

The definitive plan is [the master's work-package roadmap](docs/09_masters_thesis_work_packages.md).
For first-paper interfaces and experiments, see
[the continuous-process allocation plan](docs/08_continuous_process_multi_robot_plan.md).

As of 2026-08-10, A2 is passed and frozen at paper benchmark v4. The benchmark
contains 408 `SIM_GEOMETRIC` instances in 216 independent groups; all 402
ordinary instances have audit-only constructive A1-proxy witnesses. This
unblocks A3 graph-learning work. W9 established train-only preprocessing,
hard-mask decoding and deterministic model foundations. W10 then completed the
registered 3-family × 3-seed train/validation comparison without accessing
frozen-test/stress. Validation selected the `edge_mlp` ablation at 95.8%
verified-candidate coverage; the heterogeneous GNN and graph Transformer were
lower, and strong hybrid/LNS baselines reached 97.9–100%. This is a negative
result for a GNN-superiority narrative. The required separate preregistration
was frozen before any A3 final access.

See the [W10 development result](docs/18_a3_w10_development_results.md) and its
[machine-generated report](reports/phase1_allocation/a3_w10_development_v1_results.md).
The [one-time final-evaluation preregistration](docs/19_a3_one_time_frozen_evaluation_preregistration.md)
was followed exactly: the evaluator was developed using fixtures and
train/validation only, sealed with its source hashes, and then invoked once on
frozen-test/stress. It completed with `A3_FINAL_FAILED_BASELINE_FLOOR`: mean
three-seed coverage was 56.7%, with failures on dense precedence, tight windows
and scale. V4 is now observed and closed to retuning or repeat evaluation. See
the [A3 final closure](docs/20_a3_final_evaluation_closure.md).

A strictly separate A3.5 development pilot then tested whether an
autoregressive atomic-unit–robot Feasible-Pair Pointer addresses a limitation
of the static decoder. It used a newly generated train/validation-only corpus
with zero v2/v3/v4 ID overlap and never generated or accessed frozen-test or
stress data. The selected heterogeneous-GNN Pair-Pointer reached 78.5% mean
verified validation coverage versus 50.7% for the matched static decoder over
the same three seeds, with zero mask, atomicity or decoder-dead-end failures.
However, it remained below order-aware LNS (81.3%), was much slower than the
static decoder, and is only development evidence. See the
[A3.5 preregistration](docs/21_a3_5_feasible_pair_pointer_pilot_preregistration.md),
[closure](docs/22_a3_5_feasible_pair_pointer_pilot_results.md), and
[machine-generated report](reports/phase1_allocation/a3_5_pointer_pilot_v1_results.md).
No new frozen benchmark exists. At pilot closure, creating one required
explicit authorisation and a new preregistration.

That authorisation has now been used to freeze the
[A3.5 sealed-final preregistration](docs/23_a3_5_sealed_final_evaluation_preregistration.md).
It fixes six existing checkpoints, a new 72-group/144-instance namespace, the
group-paired significance test, strong-baseline secondary analysis and result
wording. The protocol has now completed exactly once. The fixed Pair-Pointer
reached 65.05% frozen coverage versus 40.28% for matched static decoding; the
group-paired +24.77-point difference was significant and robust across all
three seeds. Strong baselines remained higher (68.75–79.86%), so the supported
claim is decoder improvement, not solver superiority or non-inferiority. See
the [sealed-final closure](docs/24_a3_5_sealed_final_evaluation_closure.md) and
[compact result](reports/phase1_allocation/a3_5_sealed_final_v1_results.md).

The project is assembling the first-paper evidence and has completed a
development-only A4b evaluator/ordinary-LNS foundation; it is not training new
models. The [paper evidence skeleton](docs/25_a3_5_first_paper_evidence_skeleton.md)
links the main table, difficulty/seed plots, coverage–runtime Pareto, retained
failures, development-to-final comparison, claim boundary table and method
architecture. The plots explicitly show that hybrid load-balanced dominates
Pair-Pointer overall; they are descriptive renderings of immutable outputs.

## Evidence boundaries

Until verified industrial CAD, curves, layouts and logs arrive, allocation
instances are `SYNTHETIC` or `SIM_GEOMETRIC`. Static UR5 positioning CSVs are
not allocation data or offline-RL transitions. This project does not model
stress, contact force, sheet plasticity or real hemming quality; those can only
enter later through documented external constraints/costs.

See [research scope](docs/00_research_scope.md), [claims boundaries](docs/05_claims_and_boundaries.md), and [execution status](docs/07_execution_status.md).

## A4a warm-start pilot closure

The separately authorised `a4_warm_start_pilot_v1` development protocol is
closed without a frozen benchmark. Its sole validation run exposed two
evaluator-integrity defects: an infeasible MILP initializer could be replaced
by a load-balanced state while retaining the MILP label, and the fixed-time
path discarded incumbents that had become feasible before the cutoff when the
call returned slightly after it. Under the preregistered zero-integrity-failure
rule, the confirmatory timing comparison is invalid and the formal outcome is
`STOP_A4_LEARNING_WARM_START_BRANCH`. It is not evidence that Pair-Pointer is
inferior. Fixed-50-iteration results are retained only as descriptive debugging
evidence. See the [A4a closure](docs/27_a4_warm_start_pilot_results.md).

## A4b evaluator and ordinary-search foundation

The separately authorised `a4b_neural_lns_dev_v1` protocol repaired the
initializer-provenance and cutoff semantics without rerunning A4a.  It added
eight atomic-unit destroy operators, fixed random/round-robin/single-operator
LNS, adaptive ALNS, a small-instance oracle, replayable anytime traces and
train-only search-generated neighborhood-improvement labels.  Its independent
corpus contains 48 train and 24 development `SIM_GEOMETRIC` instances and no
validation/frozen/stress split.

Corrected CPU jobs `942372` and `942373_[0-5]` completed successfully.  At one
second, random and adaptive LNS both obtained 62.5% group-level coverage; ALNS
did not recover shared initializer failures or establish an objective
advantage.  The infrastructure is reusable, but the action is
`HOLD_A4B_LEARNED_DESTROY_TRAINING` until the normalized-anytime target/penalty,
fine timing split and a non-degenerate ordinary-search gate are frozen.  See
the [A4b result](docs/30_a4b_evaluator_and_alns_results.md).

A semantic audit later established that v1's fixed-iteration and
train-selected-operator rows came from time-truncated traces and are not valid
iteration-controlled evidence; the fixed-time rows remain descriptive. The
separate preregistered recovery `a4b_ordinary_lns_dev_v2` now has distinct
fixed-time/exact-iteration runs, structured travel-aware repair diagnostics,
fine timing and train-frozen anytime metrics. Its independent 96-train/48-
development corpus has been generated and audited. The first executed train
job `981071` failed closed when the 60-second operational watchdog truncated
180/240 fixed-iteration traces; development did not run. A watchdog-only
recovery amendment preserves the failed attempt. Serial recovery `983899` was
cancelled during calibration before any complete output after a six-cell CPU-
parallel execution amendment passed its tests. Packed calibration `984111`
finished all 312 trace computations but failed only at final metadata recording.
A frozen hash-bound metadata recovery performed no search; recovery/merge/train
and label gates `984885` plus smoke `984886` passed. Six-worker packed
development `984887` and fail-closed replay/aggregation `984888` then completed
successfully. All 384 fixed-iteration traces reached exact 30 and the
12,484-step replay audit passed. At 0.5/1.0/3.0 seconds all four methods tied at
72.92% coverage on the same identities; exact-30 coverage was 72.92% random,
75.00% round-robin/single-operator and 76.04% ALNS. The fixed-time searches
usually completed only zero or one neighborhood because repair dominated
runtime, so the result does not support learned destroy training. The action
remains `HOLD_A4B_LEARNED_DESTROY_TRAINING`. See the
[v2 execution record](docs/32_a4b_ordinary_lns_recovery_results.md) and
[development closure](docs/37_a4b_v2_development_closure.md).

The future A4b direction is now explicitly tied back to A3.5: after a separate
ordinary-search semantic-parity recovery, a still-unapproved pilot may test
whether a Pair-Pointer-derived atomic-unit destroy policy improves identical-
budget LNS. It must record whether A3.5 representations are frozen and reused,
transferred, distilled or only structurally inherited; a generic GNN cannot be
renamed Pair-Pointer-derived. The primary comparison keeps the hybrid
initializer and every component except destroy selection identical. See the
[corrected research plan](docs/28_a4b_neural_lns_research_plan.md) and
[unapproved staged draft](docs/38_a4b_pair_pointer_guided_search_protocol_draft.md).
The HOLD remains in force.
