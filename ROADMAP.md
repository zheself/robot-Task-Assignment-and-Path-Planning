# Implementation Roadmap

The programme follows a **first paper first, then modular expansion** strategy.
Detailed work packages and 18-month gates are in
`docs/09_masters_thesis_work_packages.md`.

## First-paper critical path

| Gate | Deliverable | Start condition | Exit condition |
|---|---|---|---|
| A0 | segment/robot/resource schema and constraint dictionary | now | auditable fixtures and typed validation pass |
| A1 | feasibility/cost oracle, verifier, heuristics and CP-SAT/MILP | A0 | solver-valid allocations and reported failures |
| A2 | `SYNTHETIC`/`SIM_GEOMETRIC` benchmark and frozen splits | A1 | leakage tests and automatic reports pass |
| A3 | masked heterogeneous GNN/graph Transformer | A2 | validation-selected, frozen-test evaluation against strong baselines |
| A4 | repair, LNS/ALNS, scale and dynamic re-planning | A3 | equal-budget Pareto and failure analysis |
| A6 | held-out real-geometry case, if data are verified | external data | evidence-labelled case study |

No GNN or RL training precedes A0–A2. A5 is intentionally not required for
first-paper acceptance: it is the later path/execution interface stage.

Current gate state (2026-08-13): A0 and A1 are frozen foundations; A2 passed
and is frozen at paper benchmark v4 (manifest
`0c98f30e92697ce8b5eca724df0f7d1b7053293df1e792707487ecb6c71b5398`).
A3 W9 foundations and W10 full train/validation development have passed. W10
selected `edge_mlp` across three seeds at 95.8% validation coverage; the two
graph models were weaker and the strongest non-learning baselines remained at
97.9–100%. A3 is therefore not yet a passed paper gate. The next gate is a
one-time final evaluation using the separately frozen protocol, selected family
and all three registered checkpoints. That evaluation is now complete and
failed the preregistered A3 baseline floor. V4 frozen-test/stress are observed
and permanently evaluation-only; they cannot revise architecture, seed, budget,
threshold or claims. The next step is a research-direction decision, not A4
implementation: retain strong solver methods as the current result, or define a
new learned question with a genuinely untouched benchmark.

That newly posed decoder question has now completed a development-only A3.5
pilot, not a new A3 final evaluation. On an independent train/validation-only
`a3_5_pointer_pilot_v1` corpus, `hetero_gnn_pair_pointer` improved matched
static-decoder coverage by 27.8 percentage points and passed the preregistered
continuation rules across all three seeds. It still did not exceed the strongest
LNS/MILP baselines, and its inference cost increased substantially. Therefore
the project has frozen the user-authorised `a3_5_sealed_final_v1` protocol. Its
primary comparison is Pair-Pointer versus the matched hetero-GNN static decoder;
MILP/LNS are secondary quality–time references. The one-time final evaluation
is now closed as **`A3_5_DECODER_HYPOTHESIS_SUPPORTED`**: +24.77-point paired
coverage improvement with CI excluding zero, but lower coverage than every
registered strong baseline and no secondary non-inferiority. The first-paper
learned contribution may therefore be framed as dynamic decoder improvement
with explicit optimisation limits.  The separately authorised A4b-0/A4b-1/
A4b-2 development foundation is now implemented; it cannot alter this final
result and has not started Neural LNS training.

Immediate work is **FIRST_PAPER_EVIDENCE_AND_WRITING plus an explicit decision
on whether to preregister stronger A4b ordinary-search recovery**. No learned
destroy method is run.
The evidence package must foreground the matched-decoder confirmatory result
and the negative engineering result that hybrid load-balanced is both faster
and more feasible. A4, if later authorised, becomes a new equal-budget
warm-start-value hypothesis; it is not “add repair to rescue Pair-Pointer.”

## Post-paper expansion

1. **WP-B:** use allocation outputs to generate per-arm geometry/kinematics
   routes, pose/direction constraints, timing and shared-zone re-planning.
2. **WP-C:** integrate the existing positioning-error prior, FK/Jacobian safety
   projection and, only if justified, supervised/deep/RL execution improvement.
3. **WP-D:** connect validated process-team limits/costs and perform held-out
   real-geometry/log case studies without taking ownership of physical models.

The compensation experiments already provide reusable infrastructure but are
not a gate for allocation paper progress. Their failed RL comparison remains
recorded in `reports/pre_advisor/real_static_multi_prior_sequence_rl_results.md`.

## A4a closure and next gate

The development-only identical-repair warm-start pilot was executed once and
is closed as `A4A_PRIMARY_EVALUATION_INVALID_STOP`. Two evaluator-integrity
defects invalidate its registered timing evidence, so it neither supports nor
refutes learned warm-start value. No A4 frozen set was generated. The first
paper therefore remains centred on the sealed A3.5 matched-decoder result with
solver/LNS and hybrid methods presented as stronger engineering references.
Any corrected warm-start study requires explicit authorisation, a new protocol,
new IDs and new validation; it may not overwrite or rerun `a4_warm_start_pilot_v1`.

## A4b development gate

`a4b_neural_lns_dev_v1` is separate from A4a and uses only new train/
development IDs.  Truthful initializer assignment provenance, monotonic
cutoffs, eight destroy operators, shared repair/verifier, adaptive weights,
trace replay and candidate-label replay are implemented.  At the registered
one-second group-level gate, ALNS and random LNS both achieved 62.5% coverage;
round-robin alone reached 66.7% at three seconds.  This passes the minimal
non-inferiority check by equality but does not supply a useful learned-destroy
training signal.  Status: `HOLD_A4B_LEARNED_DESTROY_TRAINING`; no A4b frozen
benchmark or neural checkpoint exists.

Post-result audit invalidated the v1 fixed-iteration and operator-selection
rows because their traces were truncated by the fixed-time cap. Recovery
protocol `a4b_ordinary_lns_dev_v2` is now frozen and its independent
96-train/48-development corpus is generated. It separates time and iteration
budgets and adds structured repair, train-frozen anytime metrics and stricter
train/label gates. Executed train job `981071` failed closed because its
60-second operational watchdog truncated 180/240 fixed-iteration traces; no
development ran. A frozen watchdog-only amendment now uses a 1,800-second
stall guard without changing experimental budgets. Serial replacement
`983899` was cancelled during calibration with no complete output after a
tested six-cell CPU-parallel amendment was frozen. Packed calibration `984111`
completed all 312 traces but failed at final metadata recording. A frozen
metadata-only recovery consumed the exact hashed artifacts without search;
recovery/merge/train/label job `984885` and smoke `984886` passed. Packed
development `984887` and replay/aggregation `984888` completed exit 0. The
matrix contains 768 traces and 2,304 metric rows with 384/384 exact-30
completion and a passing 12,484-step replay audit. All four methods tied at
72.92% coverage on identical identities at 0.5/1.0/3.0 seconds. Exact-30
coverage was 72.92% random, 75.00% round-robin/single-operator and 76.04% ALNS,
but ALNS's three additional recoveries appeared only after 21.99--155.53
seconds. Fixed-time traces usually completed zero or one neighborhood because
repair dominated runtime. V2 is closed as `A4B_V2_DEVELOPMENT_COMPLETE`, while
`HOLD_A4B_LEARNED_DESTROY_TRAINING` remains unchanged. A continuation should
first use a separately authorised, new-ID ordinary-search recovery protocol;
otherwise A4b should stop at this engineering foundation.

If recovery later passes, the only planned learning continuation is
Pair-Pointer-derived guided LNS, not a generic learned destroy network. Its
primary causal comparison fixes `hybrid_load_balanced` and all repair,
verification, acceptance and budget settings and changes only atomic-unit
destroy selection. Recovery authorisation does not authorise model training;
the latter requires a second protocol and approval. See `docs/28` and the
unfrozen `docs/38` draft.
