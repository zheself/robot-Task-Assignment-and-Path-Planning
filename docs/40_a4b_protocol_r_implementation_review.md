# A4b Protocol-R implementation review package

Status: **IMPLEMENTED AND TESTED; NOT FROZEN; NOT EXECUTED**  
Formal action: **`HOLD_A4B_LEARNED_DESTROY_TRAINING`**  
Evidence: **fixture and in-memory fresh-generation diagnostics only**

## Material Passport

- material type: implementation and reproducibility review;
- protocol candidate: `a4b_ordinary_search_recovery_v3` / `a4blnsr3`;
- governing draft: `docs/39_a4b_protocol_r_freeze_package_draft.md`;
- authorization used: implementation and testing only;
- not performed: freeze, corpus materialization, calibration, development,
  Slurm submission, label generation or neural training;
- protected history: A3/A3.5/A4a/A4b v1/v2 results and outputs unchanged.

## 1. Implemented capability

Protocol R now has an independent prepared repair backend. It builds immutable
unit, robot, compatibility-cost, precedence, window and resource tables once
per search trace. Candidate enumeration remains unit → robot → insertion
position, the 256 cap and regret/tie-break order are unchanged, and the A1
scheduler/verifier remain the feasibility authorities.

Every repair records the complete candidate identity/order hash, diagnostic
vector hash, selected-state hash, plan/verifier outcome, cache hits/misses and
timing. Cache scope is one instance/method/seed/trace. No candidate pruning,
parallel candidate evaluation, approximate diagnostic or cross-trace cache is
implemented.

The Protocol-R search loop retains the v2 operator, acceptance, reward,
restart, ALNS update, RNG seed and anytime-incumbent semantics. Preparation,
state encoding, repair and final verification are inside the end-to-end clock.

## 2. Data and challenge implementation

The draft now contains the complete generator geometry and all six cell specs,
copied from the named source config and bound by SHA-256. No A3.5 instance,
witness, trace, prediction or checkpoint is read.

Regular and challenge group IDs follow the exact `a4blnsr3` templates.
Challenge attempts use one group-attempt seed; the two variants share geometry
and retain separate constraint seeds. The constructor verifies the witness,
evaluates the unchanged initializer, ranks positive hybrid-versus-witness unit
finish gaps, and tightens only end windows. It retains only verified-witness
`time_window_failure` pairs and fails after 64 attempts without manual or
LNS-outcome replacement.

The data writer is one-shot and has two independent locks: the config must be
frozen with all seal fields populated, and an explicit execution authorization
must be supplied. With the current draft config it fails before creating an
output directory.

## 3. Fail-closed runner and gates

The runner implements draft audit, preflight, generation, six-cell
profile/parity, profile gate, six-cell calibration, train gate, smoke,
six-cell development and final replay/aggregation commands.

The train gate requires all configured subgates conjunctively: exact matrix,
per-method/global/per-cell search opportunity, paired 1/3-second coverage,
paired normalized primal integral, completed-neighborhood ratio, eight-group
challenge recovery, ALNS-versus-random recovery deficit, operator count,
reward-sensitive non-uniform weights, per-cell outcome dependence and at least
one identity-wise difference from random. Failures remain in all denominators.

## 4. CPU and Slurm implementation

Nine CPU-only sbatch scripts implement the reviewed linear `afterok` chain.
The submit wrapper requires confirmation plus the exact execution token,
refuses a live `a4b-r3-*` job, verifies frozen config/seal readiness, selects
the first eligible node from `sist_gpu58/59/60`, passes that node to every
stage, and saves stage IDs/dependencies/node in JSON. It never uses `afterany`,
GPU/GRES or automatic resubmission.

Packed stages bind six workers to six distinct cpuset CPUs. Every worker sets
OMP, MKL, OpenBLAS and NumExpr threads to one and clears CUDA visibility.

## 5. Test evidence

Targeted Protocol-R suite: **35 passed**. It covers reference/prepared
diagnostic equality, candidate order/vector hashes, atomicity, cap/fallback,
cache transparency, online/segmented transitions, cutoff semantics,
incompatible edges, deterministic IDs and sibling geometry, challenge
tightening, path/split guards, 400/120 matrix completeness, deterministic
merge, worker environment, dependencies and draft execution refusal.

Affected A4b regression excluding two state-inapplicable historical dry-runs:
**72 passed, 2 deselected**. A full invocation additionally reports two
pre-existing precondition failures: those tests require “no recovered shard
JSON exists”, whereas the completed v2 workspace must preserve those shards.
The old test source was deliberately left unchanged to preserve the historical
source hash; no v2 artifact was removed or modified.

Additional A0/A1/non-frozen allocation regression: **63 passed**.

All nine sbatch files passed `sbatch --test-only` on `sist_gpu60`. The reported
test-only identifiers were `985324`–`985332`; neither `squeue` nor `sacct`
contained jobs for them afterwards. The worker wrapper was executed on the
login-node cpuset and reduced its process affinity to one CPU with all thread
variables and CUDA visibility matching the contract.

## 6. Diagnostic acceleration evidence

These are non-gating implementation diagnostics; neither generated instance
was saved as Protocol-R data:

| fixture | candidates | reference | prepared | speedup | exact parity |
|---|---:|---:|---:|---:|---:|
| A0 explicit-boundary fixture, median of 7 | small | 0.000969 s | 0.000483 s | 2.01x | yes |
| fresh in-memory `iid_medium` | 134 | 0.8228 s | 0.0204 s | 40.29x | yes |
| fresh in-memory `scale` | 256 | 12.1889 s | 0.2183 s | 55.84x | yes |

The large-instance diagnostics support proceeding to the preregistered
profile gate, but do not satisfy it. Only the future complete 40-instance,
960-repair-scenario train profile can pass that gate.

## 7. Draft corrections found during implementation

Three pre-freeze ambiguities were corrected:

1. missing generator geometry/cell parameters are now copied into the draft
   and source-hashed;
2. challenge seeding separates the group-attempt seed from variant constraint
   seeds so siblings share geometry;
3. unused `repair_micro_deadline_s` inherited from v2 label generation was
   removed because Protocol R has no label stage and v2 ordinary search never
   used it.

These are draft corrections, not amendments to an executed/frozen protocol.

## 8. Remaining freeze blockers

Before freeze, a separate review must inspect the final source diff, freeze
dependencies and the CPU-family/runtime-node rule, resolve the seal manifest
without a self-referential config hash, recompute all hashes, replace all eight
null seal fields, rerun tests against those exact hashes, and authorize freeze
separately from execution.

Even after freeze, execution remains blocked until another explicit
authorization. Every Protocol-R outcome retains
`HOLD_A4B_LEARNED_DESTROY_TRAINING`.
