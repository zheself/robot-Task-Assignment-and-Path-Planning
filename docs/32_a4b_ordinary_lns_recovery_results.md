# A4b ordinary-LNS recovery v2 execution record

Date: 2026-08-18 through 2026-08-20  
Protocol: `a4b_ordinary_lns_dev_v2`  
Evidence: **SIM_GEOMETRIC development-only**  
Current action: **HOLD_A4B_LEARNED_DESTROY_TRAINING**

## Scope and immutable boundaries

This recovery repairs ordinary-search evidence only. It does not reopen A3,
A3.5 or A4a; does not access their frozen instances, witnesses or results;
does not overwrite `a4b_neural_lns_dev_v1`; and contains no neural training,
RL, reinsertion learning, frozen benchmark, path planning or physical model.

The v1 fixed-time snapshots remain descriptive. Its fixed-iteration and
train-selected operator rows are invalid for their claimed iteration and
selection semantics and are preserved with an erratum in
`docs/30_a4b_evaluator_and_alns_results.md`.

## Implemented recovery

- fixed-time and exact-iteration searches are separate invocations;
- a monotonic absolute deadline includes initializer, destroy guidance,
  repair, scheduling and verification;
- candidates verified after a cutoff are retained in the trace but never
  credited at that cutoff;
- exact-iteration rows are emitted only after a complete exact-K trace;
- partial repair ranks every robot-local insertion position by structured
  missing/invalid, precedence/order, time-window, shared-resource and load
  diagnostics, with travel/setup-aware unit duration;
- the unchanged A1 scheduler and verifier remain the only feasibility
  authorities;
- ALNS supports train-selected online or usage-normalized segmented updates;
- normalized primal integral and time-to-target use train-frozen per-cell
  target/reference values and gap 1 before first feasibility;
- search-generated neighborhood labels deduplicate canonical destroy sets and
  record violation, assignment/order edit and timing decompositions.

## Tests completed before data generation

- A4b v2 targeted recovery: 22 passed;
- final A4b evaluator/ordinary search plus affected A1 scheduler/verifier
  matrix: 89 passed;
- full explicitly non-frozen regression: 155 passed, with one unrelated
  Gymnasium deprecation warning;
- recoverable artificial counterexamples pass for precedence, tight windows
  and capacity-one shared-resource/window interaction.

No A2 v4 materialization, A3/A3.5 final or A4a suite was run.

## Independent corpus

The one-shot corpus was generated after the tests above:

| split | groups | variants/group | instances |
|---|---:|---:|---:|
| train | 48 | 2 | 96 |
| development | 24 | 2 | 48 |

All six cells have 16 train and eight development instances. Every record is
`SIM_GEOMETRIC`; IDs contain `a4blnsd2`; no validation, frozen-test or stress
directory exists. All 144 instances and 72 groups are unique, split identities
and parent curves are disjoint, and every generated witness passes the A1
verifier. Manifest semantic SHA-256:
`dddb0db22febad8f10a8d15192ea8019f586f80035a882308a2794ca4737530c`.

## Current Slurm state

The original CPU chain `978039`, `978084`, `978085_[0-5]`, `978086` was
cancelled after more than six hours pending, with elapsed zero and no output.
Two attempted GPU-node CPU-only placements also exposed `QOSMinGRES` before
execution and were cancelled at elapsed zero; their IDs and rationale are
recorded in `docs/33_a4b_v2_runtime_hardware_amendment.md`.

The amended train-only calibration/gating/label job `981071` started on
`normal/account=v-chengwy`, fixed node `sist_gpu59`, at
2026-08-18 16:25:55 UTC. It requests no GPU GRES and sets
`CUDA_VISIBLE_DEVICES=""`. Slurm allocated 2 CPUs because of the node's
allocation/billing granularity, while OMP and MKL remain fixed at one thread;
memory is 8 GiB.

The job is fail-closed: calibration is followed by the one-second/exact-30
ALNS-versus-random gate and then the label gate; a failed gate exits nonzero.
The fail-closed continuation is queued entirely through `afterok`:

- smoke `981080`, dependency `afterok:981071`;
- six-cell array `981081_[0-5]`, dependency `afterok:981080`;
- replay and aggregation `981082`, dependency `afterok:981081_*`.

The train job is currently running; downstream jobs are
`PENDING (Dependency)`. A failed gate, smoke or array task
prevents every downstream step; the chain does not retry failed work.

## Decision pending

No ordinary-LNS/ALNS v2 performance result exists yet. The action remains
`HOLD_A4B_LEARNED_DESTROY_TRAINING`. Passing process tests or generating the
corpus is not sufficient to authorise learned destroy training.

## Train-gate failure and controlled restart

Job `981071` subsequently failed the train gate after 3:31:02. Its sole failed
gate was exact-iteration completeness: 180 of 240 fixed-iteration traces hit
the 60-second operational watchdog. Labels, smoke and development did not run.
The exact-30 100% rows in that failed output contain only six complete rows per
method and are not evidence.

The root-cause analysis and pre-restart operational amendment are frozen in
`docs/34_a4b_v2_watchdog_recovery_amendment.md`. The restart reruns the whole
train calibration with a 1,800-second stall watchdog and 24-hour Slurm wall
time while retaining the original iterations, repair budget, methods, seeds,
fixed-time deadlines and fail-closed gates. The formal action remains
`HOLD_A4B_LEARNED_DESTROY_TRAINING`.

Targeted A4b tests passed 59/59 and the affected A0/A1 scheduler/verifier plus
A4b matrix passed 111/111 before restart. The failed calibration and gate files
were preserved under `failed_attempts/job_981071/` with their original hashes.
The old never-satisfied jobs `981080`, `981081_[0-5]` and `981082` were
cancelled at elapsed zero.

The replacement fail-closed chain is:

- train/gate/labels `983899`, started 2026-08-18 22:37:24 UTC on `sist_gpu59`;
- smoke `983900`, dependency `afterok:983899`;
- development array `983901_[0-5]`, dependency `afterok:983900`;
- replay/aggregate `983902`, dependency `afterok:983901_*`.

The running job requests no GPU/GRES, uses 8 GiB, and limits OMP/MKL to one
thread. Slurm allocates two CPU billing units on this node. No performance
evidence exists until all gates and aggregation complete.

## CPU-parallel recovery amendment

Serial job `983899` remained in calibration after 33:35 with no complete
calibration JSON/JSONL and had not entered gate or labels. After the CPU-
parallel implementation passed 69 targeted and 121 affected non-frozen tests,
the job and its zero-elapsed downstream chain were cancelled. Its stdout,
stderr, Slurm state and hashes are preserved under
`cancelled_attempts/job_983899/`; none is reused.

The frozen amendment in `docs/35_a4b_v2_cpu_parallel_execution_amendment.md`
defines six cell shards. Each trace is still deterministic and single-threaded.
Merge requires exactly 4 instances and 40 exact-30 plus 12 fixed-time traces
per cell, 312 unique traces overall, 240/240 exact-30 completion, and identical
manifest/config/amendment/source hashes. Operator/update selection and metric
references are recomputed only after complete deterministic merge.

An array submission `984040_[0-5]` demonstrated six distinct pinned CPUs but
could not queue the full downstream chain because the QOS counts array tasks
toward its submit limit. It was cancelled after about one minute with zero
complete shards. A first packed allocation `984061` then showed that `srun`
steps inherited the full job memory and only one could start; it was cancelled
after 16 seconds with zero shards and repaired by setting 4 GiB per step. Both
attempts and their diagnostics are preserved and excluded.

A second packed diagnostic `984072` started all six 4-GiB steps, but live
worker logs showed that this cluster reused core pairs across concurrent
`srun --cpu-bind=cores` steps. It was cancelled after 2:08 with zero shards.
The final implementation avoids step-level binding: it expands the six-CPU job
cpuset and deterministically binds worker index 0--5 to distinct logical CPUs.
This failure and its affinity map are also preserved and excluded.

Indexed packed diagnostic `984091` then demonstrated six distinct runtime
affinities (`0,1,2,28,29,30`). It was stopped before any shard completed to add
the same uniqueness requirement to the merge gate itself, rather than relying
only on launch logs. Its 2:49 run and zero-shard state are preserved and
excluded.

The unique active fail-closed chain is now:

- packed six-worker calibration `984111`, running on `sist_gpu59`;
- deterministic merge/train gate/labels `984112`, `afterok:984111`;
- smoke `984113`, `afterok:984112`;
- packed six-worker development `984114`, `afterok:984113`;
- replay/aggregate `984115`, `afterok:984114`.

Calibration and development each allocate 6 CPU and 24 GiB with no GPU. Every
worker has a 4-GiB process limit and is bound by deterministic index to one of
the allocation CPUs `0,1,2,28,29,30`. Logs confirm six distinct final
affinities and aggregate CPU consumption confirms simultaneous execution. The projected serial
fixed-iteration time is 13.5 hours versus about 7.3 hours for the slowest cell,
an estimated 1.86x train-calibration wall-clock speedup. This is execution
evidence only; `HOLD_A4B_LEARNED_DESTROY_TRAINING` remains unchanged.

## Metadata-only recovery after job 984111

Job `984111` completed all six JSONL trace shards but exited 1 after 8:32:12
when every worker reached the final `_record()` call: the NumPy version
attribute was misspelled. All 312 identities, 240/240 exact-30 traces, 72
one-second calibration traces, internal hashes and cutoff semantics passed a
read-only audit, but no cell JSON metadata existed and the original
`984112`--`984115` chain correctly remained closed.

The amendment in `docs/36_a4b_v2_calibration_metadata_recovery_amendment.md`
froze job, source, manifest, JSONL and log hashes before implementation. The
recovery command validates those exact artifacts, reconstructs 792 calibration
rows, records unavailable original worker fields as unavailable, separates
original execution hashes from recovery-tool hashes, and never calls search or
rewrites a trace. Targeted A4b tests passed 74/74 and the affected non-frozen
A0/A1 scheduler/verifier plus A4b matrix passed 108/108.

Recovery/merge/train-gate/labels job `984885` completed exit 0 in 26:15. The
train gate passed with 312 unique traces, 240/240 exact-30 completion, three
initializer-failure recoveries, and ALNS coverage equal to random at both
exact-30 (75.0%) and one second (70.83%). Train-only selection chose
`relatedness_destroy` and the `online` ALNS update. The label gate passed with
24 state records, 151 improving candidates and zero duplicate destroy sets.
Smoke `984886` completed exit 0 with eight traces.

At the post-smoke checkpoint, packed development `984887` was running on
`sist_gpu59` with six workers bound to logical CPUs `0,1,2,28,29,30`, while
finalize `984888` remained dependent on its success. At that checkpoint these
were still train/smoke and execution-integrity facts, not a development
performance conclusion. The action remained
`HOLD_A4B_LEARNED_DESTROY_TRAINING` until development, replay and aggregation
finish and are audited.

## Development completion and final closure

Packed development `984887` subsequently completed exit 0 in 13:18:30, and
the fail-closed replay/aggregation job `984888` completed exit 0 in 00:05:14.
All six cells produced 128 traces and 384 metric rows, for 768 traces and 2,304
metric rows overall. All 384 fixed-iteration traces completed exact 30; replay
checked 12,484 steps and passed. The aggregated result SHA-256 is
`9fbac9e69ff7bc5c897871c79873acd4536a862ea5c2bb318535307d06cf05d7`.

At every 0.5/1.0/3.0-second cutoff all four controlled methods had identical
72.92% coverage on the same 70/96 instance/seed identities. At exact 30,
coverage was 72.92% random, 75.00% round-robin, 75.00% train-selected
`relatedness_destroy`, and 76.04% ALNS. ALNS recovered three identities that
random did not and lost none, but those recoveries first appeared after 21.99,
123.63 and 155.53 seconds. The fixed-time traces completed a median zero or one
neighborhood, while exact-30 traces required roughly 193--200 seconds median.

The 26 shared initializer failures were all `time_window_failure` and were
concentrated in dense precedence (4), resource bottleneck (2), tight windows
(8) and scale (12). Across method/snapshot rows the retained failure library
contains 276 precedence and 332 time-window failures. Candidate events contain
9,173 precedence and 512 time-window failures; these are event counts, not
independent samples.

The execution is classified `A4B_V2_DEVELOPMENT_COMPLETE`, but the scientific
action remains `HOLD_A4B_LEARNED_DESTROY_TRAINING`: the current fixed-time
regime is repair-dominated and does not discriminate neighborhood selection.
The complete result, failure analysis, hashes and conditional recommendation
for a new ordinary-search recovery protocol are recorded in
`docs/37_a4b_v2_development_closure.md`.
