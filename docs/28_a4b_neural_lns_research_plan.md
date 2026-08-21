# A4b Pair-Pointer-derived guided LNS research plan

Status: **DIRECTION CORRECTED; `HOLD_A4B_LEARNED_DESTROY_TRAINING`**  
Direction correction: 2026-08-20  
Evidence boundary: **SIM_GEOMETRIC development-only**

This is a living research plan. It does not amend the frozen A4b v1/v2
protocols or closures. `a4b_neural_lns_dev_v1` and
`a4b_ordinary_lns_dev_v2` retain their original definitions and results.

## Historical closure and corrected question

A3 remains `A3_FINAL_FAILED_BASELINE_FLOOR`; A3.5 remains
`A3_5_DECODER_HYPOTHESIS_SUPPORTED`; A4a remains
`A4A_PRIMARY_EVALUATION_INVALID_STOP` with formal decision
`STOP_A4_LEARNING_WARM_START_BRANCH`; A4b v2 remains
`A4B_V2_DEVELOPMENT_COMPLETE` with action
`HOLD_A4B_LEARNED_DESTROY_TRAINING`. No old frozen instance, witness,
checkpoint selection or evaluator is reopened, and no old result is repaired
or reinterpreted.

### A3.5 sealed-checkpoint permission boundary

The A3.5 seal does not itself grant A4b permission to load a sealed checkpoint.
A4b may never read A3.5 frozen instances, witnesses, final-evaluation traces or
predictions. It may not reevaluate, select, ensemble, replace, fine-tune or
adapt a sealed checkpoint. A future Protocol P may load named Pair-Pointer
checkpoint weights only after explicit new authorisation, read-only, by fixed
file and state-dictionary hash, and only on entirely new Protocol-P data. The
protocol must freeze the checkpoint-to-model-seed mapping before generation;
no best-seed selection is allowed.

If that permission is not explicitly granted, the method must use newly
trained weights and be called **Pair-Pointer architecture-derived guidance**.
It may not claim reuse of the A3.5 learned representation. Protocol R neither
requires nor permits any checkpoint access.

The complete Protocol-R review candidate is `docs/39`; it proposes the fresh
`a4b_ordinary_search_recovery_v3`/`a4blnsr3` namespace, deterministic train-only
challenge generation, semantic-parity repair acceleration and fail-closed
opportunity gates. It remains an unsealed, unimplemented proposal: neither its
presence nor review authorises data generation, code implementation, Slurm
submission or any change to `HOLD_A4B_LEARNED_DESTROY_TRAINING`.

The corrected A4b research question is:

> Can a Pair-Pointer-derived, feasibility-aware neighborhood policy improve
> the anytime performance of LNS over ordinary LNS/ALNS under identical
> initializer, repair, verification, acceptance, iteration and end-to-end
> time budgets?

Pair-Pointer is neither a replacement solver nor merely an initializer in this
question. Its A3.5 atomic-unit, robot-compatibility and heterogeneous
constraint-graph representation is extended to choose which complete atomic
units an unchanged solver-based search should destroy. Repair, scheduler,
verifier and acceptance remain non-learning and identical across methods.

## Reusable completed foundation

A4b-0/A4b-2 provenance, monotonic cutoff, replay and label semantics remain
reusable. A4b v2 also retains eight ordinary destroy operators, shared repair,
fixed-time/fixed-iteration traces and failure-denominator accounting. No
ordinary-search code or evidence is deleted.

V2 nevertheless completed a median of only zero or one neighborhood in the
0.5/1.0/3.0-second regime; one shared repair commonly required several
seconds. All methods therefore had identity-wise identical 72.92% fixed-time
coverage. This regime normally exposes zero or one destroy decision and cannot
identify a learned selector effect. The HOLD remains mandatory.

## Meaning of Pair-Pointer-derived

A future method may use the name **Pair-Pointer-derived guidance** only when
its recorded provenance fits one of these routes:

| route | inherited A3.5 content | trainable content | permitted attribution |
|---|---|---|---|
| frozen representation reuse | frozen heterogeneous encoder weights, segment/unit and robot embeddings, unit-robot pair features or compatibility logits | search-state encoder and destroy head only | direct reuse of the frozen A3.5 representation |
| weight transfer | named A3.5 encoder/pair modules initialise the new policy under separate permission | transferred modules plus state/destroy head, according to a fixed fine-tuning rule | Pair-Pointer weight transfer, not a frozen model |
| distillation | frozen Pair-Pointer representations or compatibility rankings act only as teachers on new train instances | a new search policy | Pair-Pointer-derived by distillation, not weight reuse |
| structural inheritance | atomic-unit–robot pointer factorisation, state update and hard-mask structure are retained but weights are newly trained | all model weights | architectural lineage only; no claim of A3.5 knowledge transfer |

A generic GNN over units with none of these declared links must be called a
generic learned destroy selector, not Pair-Pointer-derived. Protocol P may not
compare these routes and choose the best after observing outcomes. Subject to
the explicit permission above, its preregistered **primary method** is:

```text
fixed-hash frozen Pair-Pointer heterogeneous encoder/compatibility representation
  + newly trained current-search-state destroy head
  + unchanged shared repair
```

All inherited Pair-Pointer parameters are frozen; primary-method fine-tuning
is prohibited. Weight transfer with a fixed fine-tuning rule, distillation and
structural inheritance are preregistered ablations only and cannot replace the
primary method after train or development results are seen. If checkpoint
permission is denied, Protocol P must be rewritten and frozen around the
architecture-derived method before any data or training; it is not an
automatic fallback.

## Exact future model interface

Inputs:

- the static A3.5 heterogeneous segment/robot/resource constraint graph;
- atomic-unit embeddings pooled from segment embeddings, robot embeddings,
  unit-robot pair features/compatibility and the allowed-pair mask;
- current assignment and robot-local order, robot load/completion/last
  location, precedence readiness, time-window slack and shared-resource state;
- current verifier status/failure reason, current and best objective, search
  progress, previous operator/destroy set and its outcome.

### Unit-robot compatibility to unit-destroy adapter

For atomic unit `u` and robot `r`, the fixed compatibility value `c[u,r]` comes
from the frozen Pair-Pointer pair-scoring head, not from the encoder's unused
static assignment head. For each `u`, the adapter deterministically constructs
a leave-one-unit counterfactual pointer state from the current solution:

- every unit except `u` is marked assigned and `u` is the only query unit;
- robot load, completion, last unit/location and resource use are recomputed
  from current robot-local orders after removing `u`, using the frozen A3.5
  proxy state-update semantics;
- `step = number_of_units - 1`, while previous global unit/robot action uses a
  fixed zero/none sentinel so no arbitrary inter-robot ordering is invented;
- the original Pair-Pointer compatibility and predecessor masks are applied,
  and only the allowed row for `u` is extracted from the pre-destroy scores.

This yields `c[u,r | current solution without u]`. It is a declared
counterfactual use of the frozen scorer, not a claim that A3.5 was trained on
LNS states. The adapter supplies the destroy head with:

- compatibility of `u` with its currently assigned robot;
- the best allowed alternative-robot compatibility;
- current-versus-best-alternative margin/regret and current-robot rank;
- allowed-robot fraction and allowed compatibility log-partition;
- entropy of compatibility softmax over allowed robots;
- whether the current assignment disagrees with the frozen compatibility
  ranking;
- current precedence, time-window, shared-resource, load and local-order risk.

The exact aggregation, temperature, missing-alternative sentinel and feature
normalisation, as well as counterfactual state reconstruction and extraction
code hashes, must be train-frozen. They cannot use Protocol-P development.
The new state/destroy head may combine these diagnostics with `u`, robot and
global search embeddings, but cannot silently replace `c[u,r]` with an
unrelated learned compatibility model. A score-only ablation that removes the
compatibility diagnostics must be named as such.

Outputs, in the registered first stage, are either one score per complete
atomic unit followed by deterministic top-k selection, or an autoregressive
without-replacement distribution over complete atomic units. A hard mask
forbids split-unit selection, duplicates, pinned/protocol-prohibited units and
selection beyond the registered destroy quota. The selector returns only a
canonical ordered destroy-set identity and its scores/probabilities. It cannot
change assignment, order, repair candidates, acceptance or budgets.

The selected set flows through the unchanged shared repair, A1 scheduler and
verifier. Only after this destroy-only method passes a separate preregistered
gate may unit-robot reinsertion ranking be proposed. Full learned repair,
learned acceptance, a complete solution constructor and RL are outside this
plan.

Training targets remain **search-generated neighborhood improvement labels**,
never expert truth. Equal-micro-budget candidate outcomes store feasibility
gain, objective gain, time-to-feasible, repair cost, edit distance and
Pareto/dominance relations. Pairwise/listwise ranking is primary; standalone
destroy classification accuracy is insufficient.

### End-to-end timing and cache boundary

The trace clock starts before method-specific data preparation. It includes
graph construction, frozen Pair-Pointer encoder inference, current-state
encoding, compatibility adaptation, destroy scoring or autoregressive
decoding, hard masking, canonicalisation, initializer, repair, scheduling and
final verification. Model/process deserialisation may be outside the trace
only as a declared environment setup cost; data-dependent warm-up is forbidden
and setup time is reported separately.

Static embeddings may be cached only after they are computed inside the timed
trace, and only for later iterations of that same trace. No cache may cross
instances, methods, seeds or independent trace repetitions. Ordinary methods
receive the same boundary: their graph/static operator features are also
constructed and cached after the common timer starts. Precomputed learned
embeddings are forbidden in the primary comparison. Cache hit/miss, graph,
encoder, state, selector, repair and verifier time are recorded separately.

## Experiments without initializer confounding

### Primary causal experiment

Every method starts from the same `hybrid_load_balanced` state. The comparison
is random-destroy LNS, handcrafted LNS, ALNS and Pair-Pointer-derived guided
LNS. They share atomic-unit semantics, destroy ratio, repair backend and
candidate cap, scheduler/verifier, acceptance, seed, iteration count,
end-to-end time cutoffs, failure denominator, hardware and thread settings.
The sole controlled difference is neighborhood/destroy-set selection.

### Later secondary factorial experiment

This experiment is not a substitute for the primary comparison:

| Initializer | Ordinary guidance | Pair-Pointer-derived guidance |
|---|---:|---:|
| `hybrid_load_balanced` | yes | yes |
| matched static | yes | yes |
| frozen Pair-Pointer | yes | yes |

It estimates initializer and guidance effects separately. Comparing
`Pair-Pointer initializer + learned guidance` only against
`hybrid initializer + ordinary LNS` is prohibited because it confounds both
factors.

### Success hierarchy and strong references

Claims advance only in this order:

1. demonstrate an advantage over random, handcrafted and ALNS inside the
   identical primary search framework;
2. compare with the strongest ordinary `order_aware_lns` as a separately
   labelled strong reference, matching backend/budgets where possible and
   disclosing any repair/interface difference;
3. report the coverage-runtime envelope against MILP and
   `hybrid_load_balanced` without presenting different backends as the same
   causal comparison;
4. claim superiority over traditional methods only if the guided method
   actually exceeds the preregistered strong baselines under a valid matched
   protocol and uncertainty analysis.

Passing level 1 supports only a neighborhood-guidance claim. It does not imply
levels 2--4.

## Literature position

Hottung and Tierney's Neural LNS supports embedding a learned component inside
an anytime LNS instead of replacing optimization with one-shot construction.
Its learned component is a CVRP/SDVRP repair operator trained with
policy-gradient reinforcement learning, so it is not a direct implementation
template for this destroy selector.

Zhou et al.'s airport-ground-handling method is closer: a GCN/imitation model
selects MILP variables to destroy and an off-the-shelf solver repairs the
subproblem. A4b may borrow current-solution conditioning, controlled destroy
degree, solver-preserving repair and incumbent-curve evaluation. It may not
transfer CVRP route/customer semantics, airport MILP variable groups,
airport-specific demonstrations, RL objectives, off-the-shelf sub-MILP
assumptions, real-airport evidence or either paper's performance claims.

Primary sources: [Hottung and Tierney](https://arxiv.org/abs/1911.09539) and
[Zhou et al.](https://arxiv.org/abs/2302.13797).

## Mandatory ordinary-search recovery before learning

The next admissible execution is a separately authorised ordinary-search
recovery, not model training. It must use a new ID prefix, master seed and
train/development namespace with no tuning on observed v2 development. Repair
acceleration must first prove semantic parity: identical candidate set and
order, selected state, verifier outcome and random-transition signature. Any
repair-method change is a new shared method used by every ordinary and future
guided selector and requires its own preregistration.

Before development access, the recovery must freeze and pass a minimum
completed-neighborhood/search-opportunity gate and show non-degenerate,
operator-dependent fixed-time ALNS behavior. Exact proposed gates and
fail-closed stages are in
`docs/38_a4b_pair_pointer_guided_search_protocol_draft.md`. Passing recovery
does not automatically authorise neural training; it only permits a separate
request to freeze a Pair-Pointer-derived destroy pilot.

## Evidence boundary

All future data remain `SIM_GEOMETRIC development-only` until separately
authorised. This direction cannot support claims about learned-search
superiority before a valid controlled experiment, nor real robots, factories,
collision safety, sim-to-real, force, stress, plasticity or process quality.
