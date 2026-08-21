# Project Instructions

These instructions apply to the entire repository.

## Mission and long-term architecture

This is the clean workspace for a modular master's-thesis programme in
industrial robotic process planning. Its planned 18-month architecture is:

```text
continuous process curves
  -> task segmentation / allocation / scheduling       [WP-A, current priority]
  -> multi-arm geometric-kinematic route planning      [WP-B]
  -> execution compensation / optional learning         [WP-C]
  <- external process constraints and costs             [physical-team interface]
```

The current first-paper line is **feasibility-aware graph learning for
continuous-process multi-robot task allocation and scheduling**. The atomic
object is an ordered continuous process segment (glue, hemming or weld curve
section), not a discrete point. Another team member owns discrete-point task
allocation; do not mix its instances, claims or baselines with this work.

Current gate (2026-08-11): A0/A1 are frozen foundations, A2 passed and is
frozen at v4, and A3 W9/W10 train-validation development passed. W10 selected
the `edge_mlp` ablation—not the heterogeneous GNN or graph Transformer—using
validation only. Its 95.8% validation coverage remains below the 100% strong
hybrid/LNS baselines, so no graph-learning superiority claim is supported.
The one-time A3 final evaluation is complete and immutable. It returned
`A3_FINAL_FAILED_BASELINE_FLOOR`: the fixed `edge_mlp` reached 56.7% overall
frozen coverage but failed dense-precedence, tight-window and scale absolute
gates and trailed every strong baseline. Do not rerun, tune, repair or select
against v4 frozen-test/stress. Preserve the sealed evaluator and all failed
cases. The learned A3 branch is stopped on v4; A4 must remain blocked until a
documented decision chooses either a solver-only continuation or a genuinely
new untouched benchmark. Path planning and RL remain out of scope for this
decision.

A separate development-only `a3_5_pointer_pilot_v1` has now tested the narrowly
defined decoder hypothesis without touching v4. It uses a new train/validation-
only `SIM_GEOMETRIC` corpus and an Atomic-Unit–Robot Feasible-Pair Pointer. The
selected `hetero_gnn_pair_pointer` improved mean validation verified-candidate
coverage from 50.7% for its matched static decoder to 78.5%, passed every
preregistered continuation check, and produced zero mask, atomicity or dead-end
failures. It did **not** beat `order_aware_lns` (81.3%) or establish a paper
result; median inference was about 23 times slower than the matched static
decoder. No frozen/stress benchmark was generated. The user has now authorised
and the project has frozen `a3_5_sealed_final_v1`. Its sole primary hypothesis
is Pair-Pointer versus the matched static decoder under the same hetero-GNN;
strong MILP/LNS comparisons are secondary quality–time evidence. The sealed
final has now run exactly once and returned
`A3_5_DECODER_HYPOTHESIS_SUPPORTED`: Pair-Pointer coverage 65.05% versus
matched static 40.28%, paired difference +24.77 points, 95% CI [+18.52,
+31.48], p=0.0000099999, with all integrity checks passing. It remained below
MILP 72.22%, LNS 79.86% and hybrid load-balanced 68.75%, and failed secondary
non-inferiority. Preserve the result and all failures. Do not rerun, retune,
repair, change checkpoints or reinterpret it as strong-solver superiority.
A4a was subsequently authorised as a separate development-only warm-start
pilot. Protocol `a4_warm_start_pilot_v1` is now closed with formal action
`STOP_A4_LEARNING_WARM_START_BRANCH` and integrity classification
`A4A_PRIMARY_EVALUATION_INVALID_STOP`. Do not rerun its validation. Two
semantic defects invalidate its primary time-budget evidence: the MILP adapter
substituted a load-balanced state when no scheduled MILP plan survived, and
the cutoff logic discarded already-feasible incumbents after a repair overrun.
Fixed-iteration results are descriptive only. No A4 frozen benchmark exists.

Current priority is first-paper evidence organisation around the immutable
A3.5 decoder result and the new development-only A4b solver/heuristic search
foundation, not new learning-model training. Any future A4b learning question
must be Pair-Pointer-derived guided LNS: it must explicitly reuse, transfer,
distil or structurally inherit the A3.5 atomic-unit/robot-compatibility
representation and change only destroy-set selection in the primary causal
comparison. It must not be a generic GNN destroy selector renamed after the
fact. A4b-0 evaluator semantics and
A4b-2 replayable label interface pass. The v1 ordinary ALNS development run
only tied random LNS at 62.5% one-second coverage and did not recover shared
initializer failures. Its formal action is
`HOLD_A4B_LEARNED_DESTROY_TRAINING`.  Do not train Neural LNS, generate an A4b
frozen benchmark or interpret development results as learned-method evidence.
The v1 fixed-iteration and train-selected-operator rows have a permanent
semantic erratum: their traces were fixed-time truncated and they are not
iteration-controlled evidence. A separate recovery protocol
`a4b_ordinary_lns_dev_v2` has generated a new 96-train/48-development
`a4blnsd2` corpus after non-frozen tests. The first executed train job `981071`
failed closed because the 60-second operational watchdog truncated 180 of 240
fixed-iteration traces; no development ran. A frozen watchdog-only recovery
amendment raised that stall guard to 1,800 seconds without changing any search
budget or gate. Serial replacement `983899` was then cancelled during
calibration with no complete output after a tested CPU-parallel amendment was
frozen. Packed calibration `984111` completed all 312 trace computations but
failed closed while writing shard metadata because the NumPy version attribute
was misspelled; no old downstream job ran. A separately frozen metadata-only
amendment recovered provenance from the exact hashed JSONL/log artifacts
without rerunning search. Recovery/merge/train-gate/labels `984885` and smoke
`984886` passed. Packed development `984887` then completed all six pinned,
single-threaded cell workers, and replay/aggregation `984888` passed. The final
matrix has 768 traces, 2,304 metric rows, 384/384 exact-30 completion and a
12,484-step replay audit. All four methods tied at 72.92% coverage on the same
identities at 0.5/1.0/3.0 seconds. Exact-30 coverage was 72.92% random, 75.00%
round-robin, 75.00% train-selected single operator and 76.04% ALNS; ALNS's
three additional recoveries first appeared only after 21.99--155.53 seconds.
Median fixed-time search completed zero or one neighborhood because shared
repair dominated runtime. This closes v2 as
`A4B_V2_DEVELOPMENT_COMPLETE` but does not supply a discriminative learned-
destroy evaluation foundation; the formal HOLD is unchanged. See
`docs/37_a4b_v2_development_closure.md`.
Use `docs/25_a3_5_first_paper_evidence_skeleton.md` and the descriptive assets
under `reports/phase1_allocation/figures/a3_5_sealed_final_v1/`. The fixed claim
is improvement over the matched static decoder. Always disclose that
`hybrid_load_balanced` dominates Pair-Pointer in overall coverage–runtime and
that MILP/LNS coverage is higher. Do not rerun A3/A3.5 or reopen A4a. Any
continuation beyond the completed A4b evaluator/ordinary-search development
foundation first requires a new-ID semantic-parity ordinary-search recovery
protocol and explicit authorisation. If A4b is continued, first preregister
that recovery with new IDs, semantic-parity repair acceleration and
non-degenerate fixed-time recovery gates; do not tune it on the observed v2
development split. Passing recovery would still require a second explicit
authorisation before any Pair-Pointer-derived model training.

Protocol-R implementation and non-frozen testing are now complete under an
implementation-only authorisation. The candidate remains unsealed and
unexecuted: prepared repair/search, guarded data/challenge interfaces,
conjunctive gates and the CPU-only Slurm chain exist, but no `a4blnsr3` corpus,
calibration, development output or job was created. Fixture speed diagnostics
are not a passed recovery gate. See `docs/40`; freeze and execute each still
require separate explicit authorisation, and the HOLD remains.

The A3.5 seal grants no implicit checkpoint-use permission. Protocol R must not
load any A3.5 checkpoint. A later Protocol P may use sealed weights only after
explicit new authorisation, read-only by fixed file/state-dictionary hash on
entirely new data, with no reevaluation, selection, ensembling, replacement,
fine-tuning or adaptation and no access to A3.5 frozen instances, witnesses or
final traces. Without that permission, call the method Pair-Pointer
architecture-derived, not frozen-representation reuse. The future primary
method, if authorised, is the frozen encoder/compatibility representation plus
a newly trained search-state destroy head and unchanged repair; other lineage
routes are ablations, not post-hoc alternatives.

The existing UR5 data-calibrated trajectory-compensation implementation is a
reusable WP-C foundation, not the first paper's claimed method. Its SAC/TD3
policies currently fail to beat the projected supervised prior on the primary
frozen test; preserve that result.

Read `docs/09_masters_thesis_work_packages.md`,
`docs/00_research_scope.md`, and `docs/05_claims_and_boundaries.md` before
changing any model or experiment.

## Scientific boundaries

- PointNet is neither a GNN nor a task-allocation algorithm. It may only be an
  optional local encoder for dense sampled curves/CAD point sets. Robot–task,
  task–task and resource relationships belong in an heterogeneous GNN or graph
  Transformer plus a constrained decoder.
- Existing UR5 CSVs are static absolute-positioning measurements, not offline
  RL transitions and not multi-robot allocation data.
- `SYNTHETIC` and `SIM_GEOMETRIC` continuous workcells are not real factories;
  conservative shared-zone or swept-envelope checks are not collision-safety
  guarantees.
- Stress, contact force, sheet plasticity and hemming-quality models belong to
  other team members. This project may consume their validated limits/costs as
  external inputs, but must not claim to model or predict them.
- RL is optional. Introduce it only for a defined dynamic re-planning/execution
  problem with simulator-generated transitions and a predeclared advantage over
  solver, heuristic, supervised and non-learning baselines.
- Never claim real-line deployment, real hemming validation, sim-to-real
  success, physical-quality improvement, or formal safety without matching
  evidence.

## Mandatory implementation order

Do not train a GNN or tune residual RL before completing, in order:

`A0 schema/constraint dictionary -> A1 oracle/verifier/non-learning baselines
-> A2 leakage-safe geometric benchmark -> A3 masked GNN solver imitation or
warm-start -> A4 repair/scaling/dynamic re-planning -> A5 path/execution
interfaces -> A6 held-out real-geometry case`.

CP-SAT/MILP is a required small-instance oracle; greedy, load-balanced greedy,
Hungarian-plus-sequencing, and LNS/ALNS are required comparison families when
applicable. Every learned allocation must be passed through a deterministic
feasibility/conflict verifier and marked infeasible when repair fails.

## Module contract

- `allocation`: continuous segment schema, graph construction, masks,
  assignment/order decoder, scheduling, solver adapters and repair.
- `planning`: per-robot initial route, pose/direction constraints, time
  parameterisation, reachability and shared-space checks.
- `execution`: existing FK/Jacobian, safety projection, error prior and optional
  learned execution policy.
- `process_interface`: typed external limits/costs (speed/pose bounds,
  continuity/hand-off, priority, fixture/no-go zones, risk/quality cost). It
  must not contain an undocumented physical surrogate.

The upper layer consumes only a stable planning edge-cost interface:
`feasible, travel_time, process_time, path_length, kinematic_risk,
conflict_proxy, confidence`. The lower layer must not silently rewrite an
allocation decision; it returns diagnostics and repair constraints.

## Data, evidence and reproducibility

- Split allocation data by workpiece/layout/task-instance before feature fitting
  or graph construction; do not leak curve siblings or layouts across splits.
- Fit all normalisation and learned cost models on training groups only.
- Save configuration, seed, split-manifest hash, dependency versions, metrics,
  solver status, time limit and failure cases under ignored `outputs/`.
- Use SI internally; preserve unverified source units/frames as `unverified`.
- Raw data, models, checkpoints and large arrays remain out of Git.
- Use `pytest`; add regression tests for schemas, masks, precedence, scheduling,
  repair, leakage and deterministic rollouts.

## Repository hygiene

The legacy repository `Mechanism-Guided-Residual-Reinforcement-Learning-for-
Robotic-Hemming` is read-only prior work. Do not rebuild its old ManiSkill/SB3
environment. Do not install ManiSkill/SAPIEN for this programme unless a future,
documented path-planning requirement makes that decision separately.

Use the directory contract in `docs/09_masters_thesis_work_packages.md`.
