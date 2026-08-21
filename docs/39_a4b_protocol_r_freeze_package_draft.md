# A4b Protocol-R complete freeze package candidate

Status: **DRAFT FOR SUBMISSION REVIEW — NOT FROZEN, NOT AUTHORISED**  
Date: 2026-08-20  
Proposed protocol: `a4b_ordinary_search_recovery_v3`  
Proposed ID prefix: `a4blnsr3`  
Formal action: **`HOLD_A4B_LEARNED_DESTROY_TRAINING`**  
Evidence: **SIM_GEOMETRIC development-only**

Implementation status (2026-08-20): implementation and testing only were
authorized. The code, guarded runner, tests and CPU-only Slurm chain now exist;
see `docs/40_a4b_protocol_r_implementation_review.md`. This document and its
JSON remain unsealed. No Protocol-R corpus, search output or job exists.

## Material Passport

- artifact type: code-experiment protocol/freeze candidate;
- stage: pre-implementation and pre-submission review;
- authoritative machine-readable candidate:
  `configs/allocation/a4b_protocol_r_freeze_candidate_v1.json`;
- profile evidence:
  `reports/phase1_allocation/a4b_protocol_r_v2_profile_evidence.json`;
- source data used for planning: A4b v2 development traces only;
- verification status: `ANALYZED_NOT_EXECUTED`;
- next gate: explicit user approval to implement and return a final hash-bound
  freeze package; approval to execute is a later, separate decision.

This package is deliberately unusable as an active protocol: its status is
`DRAFT_FOR_REVIEW_NOT_FROZEN_NOT_AUTHORIZED`, its seal-time hashes are null,
and no Protocol-R runner or Slurm file exists. It creates no corpus, performs
no search and grants no checkpoint access.

## 1. Purpose and non-purpose

Protocol R asks only whether the existing ordinary search can be implemented
with exact semantic parity but enough fixed-time search opportunity to support
a later selector comparison. It does not test Pair-Pointer guidance, create
learning labels, train a model, load A3.5 weights or authorize Protocol P.

The unchanged experimental semantics are:

- `hybrid_load_balanced` initializer without fallback;
- complete atomic units;
- the same eight destroy operators and ratios 0.10/0.25/0.40;
- the same candidate sequence and 256 candidate cap;
- the same violation ranking, selected state, scheduler/verifier, acceptance,
  ALNS rewards/update and restart;
- exact 10/20/30 iteration snapshots and 0.5/1.0/3.0-second end-to-end cutoffs;
- all failures in the independent `task_group_id` denominator.

The only admissible scientific difference from v2 is a new independent data
namespace. The only implementation difference is removal of repeated
computation while preserving the reference transition semantics.

## 2. Full proposed configuration

The complete proposed values, counts, gates, hardware and boundaries are in
[the review-only JSON config](../configs/allocation/a4b_protocol_r_freeze_candidate_v1.json).
There are no omitted parameter defaults. The null `seal_time_required_fields`
are not experimental choices: they are hashes/versions that can exist only
after the proposed code and tests have been implemented and reviewed.

Implementation audit correction (2026-08-20): the first review candidate
omitted the generator geometry and per-cell ranges even though this section
claimed there were no omitted defaults. The draft JSON now copies those values
verbatim and binds their source config by SHA-256. This is a draft correction,
not a freeze: Protocol R reuses only the programmatic generator and reads no
A3.5 instance, witness, evaluation trace, prediction or checkpoint.

Key immutable candidates for review are:

| item | proposed value |
|---|---|
| protocol / prefix | `a4b_ordinary_search_recovery_v3` / `a4blnsr3` |
| master seed | `1047299` |
| paired search seeds | `3253`, `4099` |
| profile/parity seed | `5843` |
| regular train | 8 groups/cell × 6 cells = 48 groups |
| train challenge | 2 groups/cell × 4 cells = 8 groups |
| total train | 56 groups, 112 instances |
| development | 4 groups/cell × 6 cells = 24 groups, 48 instances |
| variants | 2 per group |
| splits | train/development only |
| calibration subset | 40 train instances |
| development matrix | 768 traces, 2,304 metric rows |
| neural/checkpoint stage | none |

All identifiers must contain `a4blnsr3` and be disjoint from A2 v2/v3/v4,
A3.5 pilot/final, A4a, A4b v1 and `a4blnsd2`. Development IDs are generated
once but cannot be opened before all train gates pass.

The exact ID templates are freeze-candidate fields, not implementation
conventions: regular train groups are
`a4blnsr3-train-{cell}-regular-group-{index:03d}`, challenge train groups are
`a4blnsr3-train-{cell}-challenge-group-{index:03d}`, development groups are
`a4blnsr3-development-{cell}-group-{index:03d}`, and instances append
`-v{variant:02d}`. Workpiece, layout and parent-curve IDs are derived from the
group ID by the templates in the JSON. The manifest gate requires zero overlap
between train and development for group, instance, workpiece, layout and
parent-curve IDs. Development is never used for parameter selection or
challenge construction.

## 3. Existing profile evidence

The read-only profile used all 384 v2 fixed-iteration development traces and
11,520 completed neighborhoods. Every source trace hash is recorded in
[the profile evidence JSON](../reports/phase1_allocation/a4b_protocol_r_v2_profile_evidence.json).

| component | q25 | median | q75 |
|---|---:|---:|---:|
| guidance | 0.0115 s | 0.0148 s | 0.0263 s |
| complete repair | 2.0789 s | 5.9961 s | 11.8185 s |
| repair selection/structural path | 2.0630 s | 5.9808 s | 11.7864 s |
| final scheduler | 0.000178 s | 0.000294 s | 0.000441 s |
| candidate evaluations | 153 | 256 | 256 |

Repair selection accounts for approximately 99.74% of median repair time;
7,612/11,520 steps (66.08%) reach the 256 cap. Cell medians range from 0.387 s
for `iid_small` to 21.448 s for `scale`, whose cap saturation is 96.67%.
Scheduler/verifier optimization cannot plausibly recover the required wall
time. The target must be reached by eliminating repeated deterministic
candidate-state work while evaluating the same candidates.

## 4. Proposed implementation and code paths

Historical v2 files remain as the reference backend and are not rewritten.
The proposed implementation creates:

| path | proposed responsibility |
|---|---|
| `src/safe_residual_rl/allocation/search/prepared_repair.py` | immutable per-instance units, robot index, costs, predecessor, window, resource and geometry tables |
| `src/safe_residual_rl/allocation/search/repair_protocol_r.py` | accelerated candidate evaluation, trace-local cache and shadow-reference parity hooks |
| `src/safe_residual_rl/allocation/search/alns_protocol_r.py` | v2-equivalent search loop with injected reference/accelerated repair and expanded timing/signatures |
| `src/safe_residual_rl/allocation/search/data_protocol_r.py` | new namespace generator, challenge construction, split/path guards and manifest audit |
| `scripts/run_a4b_protocol_r.py` | generate/profile/parity/calibrate/gate/smoke/development/replay/aggregate commands |
| `scripts/run_a4b_protocol_r_worker.sh` | one-thread environment, affinity and CPU-signature audit |
| `scripts/submit_a4b_protocol_r_chain.sh` | duplicate-job guard, node selection and `afterok` submission only after a later execution authorization |
| `tests/allocation/test_a4b_protocol_r_repair.py` | repair and transition parity tests |
| `tests/allocation/test_a4b_protocol_r_data.py` | namespace/challenge/split tests |
| `tests/allocation/test_a4b_protocol_r_pipeline.py` | matrices, gates, dependencies and failure propagation |
| `slurm/a4b_protocol_r_*.sbatch` | nine proposed fail-closed stages |

Permitted acceleration is ordered and auditable:

1. prepare immutable values currently rebuilt by each `analyze_state` call:
   allocation units, segment/unit maps, robot indices, edge costs, precedence
   edges, windows, durations and resource membership;
2. compute canonical candidate state and its hash once, then reuse it for
   ranking, tie-break, trace and memo lookup;
3. memoize only identical state diagnostics inside one trace;
4. optionally maintain exact incremental load/order/precedence/window/resource
   bookkeeping, but only behind a shadow-reference equality check.

Candidate pruning, candidate parallelism, approximate diagnostics, changed
floating comparison order, changed state hashing, reduced cap and reordered
robot/position/unit loops are forbidden. If stages 1--3 do not reach the speed
gate and stage 4 cannot pass exact parity, Protocol R stops; it does not change
repair semantics to meet the target.

Implementation audit correction (2026-08-20): the inherited
`repair_micro_deadline_s` field was removed. In v2 it applied only to label
generation, not ordinary search; Protocol R has no label stage. Search repair
is governed only by the unchanged 256-candidate cap, the fixed end-to-end
cutoff, or the 1,800-second operational watchdog in exact-iteration mode.

## 5. Deterministic challenge generation

Challenge groups are train-only eligibility cases, not development evaluation
or performance-selected hard examples. There are two groups in each of
`dense_precedence`, `resource_bottleneck`, `tight_windows` and `scale`; both
variants of all eight groups must have a verified constructive witness and a
`hybrid_load_balanced` `time_window_failure`.

For `(cell, group_slot, attempt)`, the generator derives a fresh group-attempt
seed from the proposed master seed; the unchanged generator derives variant
constraint seeds internally, so both variants retain the same group geometry.
It accepts an already eligible base instance.
Otherwise it compares the verified witness finish and hybrid finish for each
atomic unit, sorts positive finish gaps by descending gap then unit ID, and
progressively tightens only end windows to
`min(original_end, midpoint(witness_finish, hybrid_finish))`. After every
prefix it verifies the unchanged witness and reruns the initializer. The first
witness-feasible/time-window-failing prefix is retained with all hashes.

A group is accepted only if both variants pass. There are at most 64 base
attempts per slot. Exhaustion fails corpus generation. No LNS outcome,
operator, recovery result or development data participates in generation.
Manual replacement is forbidden.

## 6. Semantic-parity fixtures and signatures

Twelve named human-readable fixtures are proposed:

| fixture | required invariant |
|---|---|
| `r01_atomic_unit` | no segment-level split; destroy/reinsert whole units only |
| `r02_duplicate_destroy` | duplicate/out-of-range destroy IDs rejected identically |
| `r03_candidate_order` | exact unit→robot→position candidate identity/order hash |
| `r04_rank_tie` | equal violation/regret ties retain state-hash/unit tie-break |
| `r05_precedence` | precedence vector, deadlock and selected successor match |
| `r06_tight_window` | lateness vector and time-window failure match bitwise |
| `r07_shared_resource` | overlap vector and resource failure match bitwise |
| `r08_cap_fallback` | 256-cap flag, remaining-unit order and fallback state match |
| `r09_fake_deadline` | identical fake-clock cutoff gives identical prefix/fallback |
| `r10_feasible_to_infeasible` | verifier protection and acceptance input match |
| `r11_initializer_recovery` | infeasible start, first feasible event and plan hash match |
| `r12_alns_transition` | operator, destroy set, RNG, acceptance, reward and weight update match |

Fresh-train parity additionally uses all 40 calibration instances:

- 8 operators × 3 destroy ratios = 24 repair scenarios per instance;
- reference and accelerated repair run sequentially on the same pinned CPU;
- semantic runs disable the real deadline or use the same fake clock;
- candidate sequence/order, every violation-vector field, selected state,
  evaluation count, cap/fallback flags, scheduler diagnostic, plan and verifier
  hashes must match 100%;
- random, online-ALNS and segmented-ALNS exact-five prefixes must have identical
  RNG and transition signatures;
- real-clock speed runs are separate. Their common completed transition prefix
  must match; the accelerated trace may contain an extra suffix because it
  completes more work before the same cutoff.

One mismatch stops before calibration. No tolerance or aggregate pass rate is
allowed for deterministic fields.

## 7. Test matrix before any Slurm submission

| test family | minimum checks |
|---|---|
| static cache | prepared and reference units/robots/costs/windows/resources/precedence identical |
| candidate parity | identity, order, rank/scalar, state hash and chosen insertion exact |
| fallback/deadline | cap=256, fake-clock prefix, fallback order and overrun semantics |
| search transition | fixed-iteration signature, RNG state, acceptance, reward, ALNS weight |
| anytime | initializer charged; 0.5/1.0/3.0 snapshots unchanged; extra accelerated suffix not backdated |
| challenge | deterministic repeat, two variants, verified witness, initializer failure, 64-attempt fail-close |
| data isolation | 56/24 groups, 112/48 instances, zero old-prefix/group/layout/curve overlap |
| path guards | reject validation/frozen_test/stress and all A3/A3.5/A4a/A4b-old tokens |
| calibration matrix | 400 fixed-iteration + 120 fixed-time traces; 400/400 exact-30 |
| opportunity gates | global/per-cell completed-neighborhood thresholds and timing decomposition |
| ALNS gate | paired coverage, primal integral, opportunity and challenge recovery subgates |
| development matrix | 384 fixed-iteration + 384 fixed-time traces; 2,304 rows |
| replay | every transition/state/plan/verifier hash; all failures retained |
| execution | unique CPU affinity, one-thread environment, empty CUDA visibility, same selected node |
| dependency | every stage uses `afterok`; missing/failed shard blocks every descendant |
| provenance | config/source/test/dependency/CPU/node/command hashes and dependency versions |

The pre-submission suite must include targeted Protocol-R tests, affected
non-frozen allocation regression and the explicitly guarded full non-frozen
project regression. No test may discover or traverse a forbidden frozen path.

## 8. Train and development matrices

Calibration uses 24 regular train instances plus all 16 challenge instances.
Per instance it runs seven handcrafted operators, two ALNS update schemes and
random LNS in fixed-iteration mode, plus the two ALNS schemes and random LNS at
the one-second fixed-time calibration point. Expected totals are:

- 400 fixed-iteration traces, all exact 30;
- 120 fixed-time traces;
- 520 traces and 1,320 metric rows.

Only the complete merged calibration may select the best single operator,
ALNS update scheme and metric references. Challenge and regular rows remain
identified separately; challenge rows cannot redefine development difficulty
or be reported as ordinary distribution performance.

Development retains the v2-sized paired matrix on entirely new natural
development groups:

- 48 instances × 2 seeds × 4 methods × 2 budget modes = 768 traces;
- 384/384 exact-30 traces required;
- 2,304 metric rows across three time and three iteration snapshots;
- 24 independent group-level statistical units.

## 9. Gates and stop logic

All gates in the JSON are conjunctive.

### Profile and opportunity

- global median complete neighborhood ≤0.80 s;
- global median repair-path speedup ≥7.0x on paired same-node cases;
- `scale` median complete neighborhood ≤2.40 s;
- for every method, global median completed neighborhoods ≥1 at 1 s and ≥3 at
  3 s;
- at least 90% of 3-second traces complete one neighborhood;
- every method/cell has median ≥1 completed neighborhood at 3 s;
- `iid_small` and `iid_medium` have median ≥3 at 3 s.

### Challenge and ordinary-search behavior

- all eight eligible groups across four cells exist before calibration;
- at least two groups and at least 25% are recovered by an ordinary method by
  3 s;
- at 1 s and 3 s ALNS paired coverage trails random by no more than one group;
- paired mean normalized primal integral is no more than 0.02 worse;
- ALNS median completed-neighborhood count is at least 90% of random;
- ALNS eligible recovery trails random by no more than one group;
- operator weights respond deterministically to unequal rewards and fixed-time
  operator outcomes are not identity-wise all equal.

No single coverage point can override another failed subgate. Failure stops
before development; no extra time, cap reduction or parameter adjustment is
allowed.

## 10. CPU and Slurm plan

No GPU or GRES is requested. Submission-time node selection is restricted to
`sist_gpu58`, `sist_gpu59`, `sist_gpu60`: choose the lexicographically first
available node with at least six free CPUs and 32 GiB unallocated memory, then
pin every timed stage in the chain to that exact node. This avoids permanent
binding now while preventing cross-node fixed-time comparisons. Node state,
CPU model/signature, affinity, load and memory are recorded before each stage;
a mismatch fails.

Packed stages allocate six CPUs and 32 GiB. Six single-threaded cell workers
receive distinct cpuset CPUs and 4 GiB process limits. All workers set
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`
and `CUDA_VISIBLE_DEVICES=""`.

| proposed job name | CPUs / memory | limit | dependency |
|---|---:|---:|---|
| `a4b-r3-preflight` | 1 / 8G | 00:30 | none |
| `a4b-r3-generate` | 1 / 8G | 01:00 | preflight `afterok` |
| `a4b-r3-profile` | 6 / 32G | 04:00 | generate `afterok` |
| `a4b-r3-profile-gate` | 1 / 8G | 00:30 | profile `afterok` |
| `a4b-r3-calibrate` | 6 / 32G | 06:00 | profile-gate `afterok` |
| `a4b-r3-train-gate` | 1 / 8G | 01:00 | calibrate `afterok` |
| `a4b-r3-smoke` | 1 / 8G | 00:30 | train-gate `afterok` |
| `a4b-r3-development` | 6 / 32G | 08:00 | smoke `afterok` |
| `a4b-r3-finalize` | 1 / 8G | 02:00 | development `afterok` |

The eventual submission wrapper must reject any live job named
`a4b-r3-*`, require an explicit execution token, save the exact `sbatch`
commands and returned IDs, and submit only the linear `afterok` chain. It must
never auto-resubmit a failed stage.

## 11. Expected wall clock

These are planning estimates, not evidence that acceleration has succeeded:

- paired reference/candidate profile and parity: approximately 1.5--2.5 h
  packed wall, dominated by reference `scale` evaluations;
- accelerated 520-trace calibration: approximately 0.8--1.5 h if the speed gate
  passes;
- merge/gates/smoke: approximately 0.3--0.6 h;
- accelerated 768-trace development: approximately 1.0--2.0 h, with `scale` as
  the expected straggler;
- replay/aggregation: approximately 0.1--0.3 h.

Expected wall after the first allocation is 4--6 h; queue time is excluded.
The longer requested Slurm limits are failure guards, not added experimental
budgets. Actual CPU-hours, worker imbalance and speedup must replace estimates
in the closure.

## 12. Fail-closed dependency chain

```text
preflight
  -> generate + challenge audit
  -> six-cell profile/parity
  -> deterministic profile merge + parity/speed/opportunity gate
  -> six-cell train calibration
  -> deterministic merge + complete train/ALNS/challenge gate
  -> one-instance smoke
  -> six-cell development
  -> replay + aggregate + closure candidate
```

Each merge requires all six expected cell shards, exact identities/counts,
unique trace keys, complete hashes, identical source/config/manifest/node/CPU
signatures and no incomplete exact-iteration trace. A missing, failed, partial,
duplicate or foreign shard returns nonzero, so all descendants remain blocked
by `afterok`.

## 13. Immediate stop conditions

Stop and preserve all outputs/logs without retry when any of the following
occurs:

1. forbidden old/frozen path access, namespace overlap or split/group leakage;
2. challenge witness failure, insufficient eligible groups or attempt-cap
   exhaustion;
3. any candidate/order/vector/state/plan/verifier/RNG/transition parity
   mismatch;
4. speed or completed-neighborhood opportunity gate failure;
5. reduced/reordered candidate set, altered cap, repair or acceptance semantic
   drift;
6. CPU affinity, thread, CUDA, selected-node or CPU-signature mismatch;
7. incomplete/duplicate/hash-inconsistent shard or exact-iteration trace;
8. ALNS multi-metric non-systematic-weakness or operator-dependence failure;
9. any development access before every train gate passes;
10. any attempt to produce labels, a neural checkpoint or Protocol-P output.

Failure does not authorize an amendment, extra budget or automatic rerun. The
failed attempt is hashed and reported for a new user decision.

## 14. Seal checklist after review, before any later submission

If the user approves implementation, a second review must receive:

- final source diff and every new source hash;
- exact config and protocol-document hashes with no null seal fields;
- profile-evidence hash and extraction test;
- targeted/affected/full-non-frozen test commands, counts and report hashes;
- challenge dry-fixture evidence without Protocol-R corpus generation;
- Slurm/submit script hashes and `sbatch --test-only` records;
- dependency versions, CPU allowlist and node-selection audit command;
- expected matrices recomputed mechanically from the config;
- explicit statement that implementation approval is not execution approval.

Only after that package is approved may the protocol status become frozen.
Freezing still does not submit jobs. A third explicit authorization is required
to execute the unique chain. Every outcome retains
`HOLD_A4B_LEARNED_DESTROY_TRAINING`.

## 15. Review decisions requested

Before implementation, the user should accept or revise:

1. namespace `a4b_ordinary_search_recovery_v3` / `a4blnsr3` and three seeds;
2. 56/24 group structure and train-only 8-group challenge stratum;
3. deterministic time-window challenge construction and 64-attempt stop;
4. allowed cache/incremental acceleration scope;
5. exact parity plus 0.80 s global / 2.40 s scale targets;
6. multi-metric ALNS and opportunity thresholds;
7. six-worker same-node Slurm plan and 4--6 h expected post-allocation wall;
8. three-step authority boundary: implement → freeze → execute.
