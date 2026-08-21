# A4b Pair-Pointer-derived guided-search staged protocol draft

Status: **DRAFT — NOT FROZEN, NOT AUTHORISED FOR EXECUTION**  
Date: 2026-08-20  
Current action: **`HOLD_A4B_LEARNED_DESTROY_TRAINING`**  
Evidence boundary: **SIM_GEOMETRIC development-only**

This draft plans two separately authorised protocols. It submits no job,
generates no data, trains no model and does not alter any A3/A3.5/A4a/A4b
closure. Protocol R must close successfully before Protocol P may even be
requested.

## Protocol R: ordinary-search semantic-parity recovery

### Purpose and isolation

Protocol R exists only to create enough completed neighborhoods inside the
unchanged 0.5/1.0/3.0-second end-to-end cutoffs for selector comparisons.
Before execution it must freeze a new protocol ID, ID prefix, master seed,
train/development namespace, group counts and hashes. There is no validation,
frozen-test or stress split. Old v2 development may justify this engineering
question but cannot select an implementation or parameter.

Before generation, Protocol R must also preregister an initializer-failure
challenge stratum. The current draft target is at least eight independent
eligible train groups spanning at least four difficulty cells, in addition to
human-readable fixtures. If generation cannot satisfy the frozen eligible
count without post-generation selection, the corpus gate fails and no search
gate is evaluated.

The shared `hybrid_load_balanced` initializer, atomic units, destroy ratios,
256 repair candidate cap, scheduler/verifier, acceptance, seeds within each
paired comparison, exact-iteration count, end-to-end cutoffs, denominator and
single-thread trace semantics remain fixed. CPU parallelism may only separate
independent traces.

### Recovery order

1. Profile fixtures and fresh Protocol-R train cases without changing search.
2. Accelerate the existing repair implementation only through
   semantic-preserving changes such as cached static features, incremental
   evaluation and eliminated duplicate computation.
3. On every parity case compare the canonical candidate set and its order,
   selected successor state, plan/verifier hashes, acceptance input and seeded
   random-transition signature.
4. Reject the acceleration on any mismatch; do not average mismatches away.
5. If parity-preserving acceleration cannot pass the opportunity gate, stop.
   A different repair algorithm requires a separate amendment and becomes the
   common repair for every ordinary and later learned method.

### Expected acceleration target for detailed freeze

V2 recorded 5.76--6.34 s median repair time per neighborhood at the unchanged
256-candidate cap. To leave time for initializer, destroy selection,
scheduling and verification while completing a median of three neighborhoods
by 3.0 s, the draft engineering target is at most 0.80 s median wall time for
one complete ordinary neighborhood on the frozen CPU node family. This implies
approximately 7.2--7.9x median repair-path acceleration relative to v2. The
90th-percentile first-neighborhood completion must also remain within 3.0 s,
matching the 90% opportunity gate below.

This is a wall-clock recovery target, not a new search budget. Protocol R may
not reduce the 256-candidate cap, omit candidates or verifier calls, reorder
candidates, change selected states, increase the time cutoff or reuse work
across independent traces to reach it. The detailed freeze must include a
fresh-train profile by component, same-node serial/accelerated paired timing,
CPU affinity and the measured rather than theoretical speedup. Failure to
reach the opportunity gate stops the protocol even if a microbenchmark reports
a favorable speedup.

### Proposed train gates to freeze before data generation

These thresholds are identifiability requirements, not performance values
selected from v2 development:

- parity: 100% equality of candidate identities/order, selected state,
  verifier result and random-transition signature on all fixtures and audited
  fresh-train traces;
- opportunity: for every controlled method, median completed neighborhoods is
  at least one by 1.0 s and at least three by 3.0 s; at least 90% of 3.0-second
  traces complete one neighborhood;
- exact iteration: every required fixed-iteration trace completes the exact
  registered count; interrupted traces stay failed in the denominator;
- non-degeneracy: at least two handcrafted operators produce distinct
  canonical destroy sets and distinct successor-state or outcome hashes on
  eligible cases in every nontrivial difficulty cell;
- ALNS operation: at least two operators are selected, deterministic weight
  updates move away from uniform when rewards differ, and fixed-time outcomes
  are not identity-wise identical across all operators;
- eligible recovery: the preregistered challenge stratum contains at least
  eight initializer-failed independent groups across four cells; at least two
  groups and at least 25% are recovered within 3.0 s by an ordinary method;
- paired non-systematic-weakness gate: at both 1.0 and 3.0 s, ALNS may trail
  random by no more than one independent group in paired verifier coverage;
  its paired mean normalized primal integral may be worse by no more than
  0.02, its median completed-neighborhood count must be at least 90% of
  random, and its eligible recovery count may trail random by no more than one
  group;
- integrity: all failures remain in denominators; replay, source/config/
  manifest hashes, forbidden paths and split/group isolation pass.

Coverage, normalized primal integral/anytime regret, completed-neighborhood
count and eligible recovery are reported together. No single coverage cutoff
alone can pass the ordinary-search foundation. The numerical margins above are
draft values and must be frozen before Protocol-R generation, not adjusted
after train or development outcomes.

Any failed gate stops before Protocol-R development. Development is used only
once for the frozen recovery assessment and cannot tune Protocol P.

## Protocol P: Pair-Pointer-derived destroy pilot

Protocol P remains unavailable until Protocol R has closed successfully and
the user separately authorises training. It must then receive another new ID,
seed and train/development namespace; Protocol-R development cannot select its
model or hyperparameters.

### Research question

> Can a Pair-Pointer-derived, feasibility-aware neighborhood policy improve
> the anytime performance of LNS over ordinary LNS/ALNS under identical
> initializer, repair, verification, acceptance, iteration and end-to-end
> time budgets?

### Derivation audit and model contract

Protocol P receives no checkpoint permission from the A3.5 closure. It may not
read A3.5 frozen instances, witnesses, final traces or predictions, and may not
reevaluate, select, ensemble, replace, fine-tune or adapt sealed checkpoints.
Only a new explicit authorisation may permit read-only loading of named,
fixed-hash Pair-Pointer weights on new Protocol-P data. Without that
authorisation, the protocol must be rewritten as Pair-Pointer
architecture-derived guidance and cannot claim learned-representation reuse.

Subject to that authorisation, the primary method is fixed now as a frozen
Pair-Pointer heterogeneous encoder/compatibility representation plus a newly
trained current-search-state destroy head and unchanged repair. All inherited
parameters remain frozen; fine-tuning is not allowed in the primary method.
Exact checkpoint hashes and deterministic checkpoint-to-model-seed mapping
must be frozen before data generation. Weight-transfer/fine-tuning,
distillation and structural inheritance may appear only as named,
preregistered ablations; their results cannot replace the primary method.

The model consumes static heterogeneous graph embeddings, unit-robot
compatibility/pair features and current search state. It returns either
atomic-unit scores plus deterministic top-k or autoregressive probabilities
without replacement. Hard masks enforce whole units, uniqueness, destroy
quota and protocol exclusions. Its only action is a canonical destroy set.
Repair, candidate cap, scheduler/verifier and acceptance are common,
non-learning components.

For each unit `u`, compatibility is queried directly from the frozen
Pair-Pointer pair-scoring head under a deterministic leave-one-unit
counterfactual current state. All other units are marked assigned; load,
completion, robot-local tail/location and resource use are recomputed after
removing `u`; step is `n_units - 1`; previous global action is the fixed
zero/none sentinel; and the original compatibility/predecessor mask exposes
only allowed robots for `u`. The encoder's unsupervised static assignment head
is not used as Pair-Pointer compatibility.

The adapter then exposes compatibility with the current robot, best allowed
alternative compatibility, their margin/regret, current-robot rank,
allowed-robot fraction and compatibility log-partition, allowed-robot entropy,
disagreement with the frozen ranking and dynamic precedence/window/resource/
load/order risk. Aggregation, temperature, normalisation, missing-alternative
sentinels and counterfactual reconstruction code are frozen using Protocol-P
train only. The destroy head cannot substitute an unrelated compatibility
model under the Pair-Pointer-derived name. Protocol P must test deterministic
reconstruction, mask equality, finite allowed logits and identical repeated
compatibility hashes before training.

Supervision is pairwise/listwise ranking over equal-micro-budget
`search-generated neighborhood improvement labels`. It is not expert truth.
No RL, learned repair, learned acceptance, reinsertion learning or complete
solution construction is permitted in Protocol P.

### End-to-end time boundary

Every fixed-time trace starts its monotonic clock before method-specific data
preparation. Charged time includes graph construction, frozen encoder
inference, current-state encoding, pair-to-destroy adaptation, scoring or
autoregressive decoding, hard mask, canonicalisation, initializer, repair,
scheduler and verifier. Static embeddings may be cached only after their timed
construction and only within the same trace. Ordinary methods construct and
cache their static graph/operator features inside the same boundary. No cache
crosses instance, method, seed or trace; no learned-only offline embedding or
data-dependent warm-up is permitted. Component and cache timings are retained.

### Primary causal matrix

All rows start from the identical `hybrid_load_balanced` solution:

| method | initializer | controlled difference |
|---|---|---|
| random-destroy LNS | shared hybrid | destroy-set selection |
| handcrafted LNS | shared hybrid | destroy-set selection |
| ALNS | shared hybrid | destroy-set selection |
| Pair-Pointer-derived guided LNS | shared hybrid | destroy-set selection |

Destroy ratio, repair/candidate cap, verifier, acceptance, seed, iterations,
time cutoffs, failure denominator, hardware and threads are identical.

### Secondary factorial, later and separately gated

| Initializer | Ordinary guidance | Pair-Pointer-derived guidance |
|---|---:|---:|
| `hybrid_load_balanced` | yes | yes |
| matched static | yes | yes |
| frozen Pair-Pointer | yes | yes |

The factorial cannot replace the primary matrix. A cross-cell comparison that
changes initializer and guidance simultaneously cannot establish guidance
value.

### Success hierarchy

1. Primary evidence first compares Pair-Pointer-derived guidance with random,
   handcrafted and ALNS under the identical framework above.
2. The strongest ordinary `order_aware_lns` is then reported as a strong
   reference, with any repair/backend difference explicit.
3. MILP and `hybrid_load_balanced` define a coverage-runtime envelope; they do
   not become rows in the destroy-selection causal contrast when their
   backends or roles differ.
4. “Better than traditional methods” is permitted only after a separately
   preregistered matched comparison actually exceeds the strong baselines with
   the required uncertainty criterion.

### Stop conditions and later work

Protocol P fails closed on derivation-provenance, hard-mask, atomicity,
opportunity, replay, split, hash or denominator failure. It must report fixed
time coverage, time-to-first-feasible, objective at time, normalized primal
integral and time-to-target at independent `task_group_id` level.

Passing a destroy-only pilot would still not authorise reinsertion ranking.
That extension requires another protocol showing a valid destroy-only gain.
RL, full repair learning and acceptance learning remain outside this sequence.

## Required next authorisation

The complete pre-implementation freeze candidate is now in
`docs/39_a4b_protocol_r_freeze_package_draft.md` with machine-readable config
`configs/allocation/a4b_protocol_r_freeze_candidate_v1.json`. It remains
unfrozen and unexecuted. The next possible request is only authorization to
implement the reviewed Protocol-R code/tests and return final source/test/
Slurm hashes for a second freeze review. Implementation authorization does not
freeze or execute Protocol R, and no authorization for Protocol P is implied.
