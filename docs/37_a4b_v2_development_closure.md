# A4b ordinary-LNS recovery v2 development closure

Date: 2026-08-20  
Protocol: `a4b_ordinary_lns_dev_v2`  
Execution classification: **`A4B_V2_DEVELOPMENT_COMPLETE`**  
Scientific action: **`HOLD_A4B_LEARNED_DESTROY_TRAINING`**  
Evidence: **SIM_GEOMETRIC development-only**

## Closure scope

This document closes the v2 ordinary-search recovery execution. It does not
change the immutable A3, A3.5 or A4a conclusions, does not repair or rerun A4b
v1, and does not authorize Neural LNS, RL, reinsertion learning, a frozen A4b
benchmark, path planning or physical-model claims.

The v2 result establishes an auditable evaluator, exact-iteration ordinary
search and replayable data foundation. It does not establish that ALNS is
better than random or handcrafted LNS and does not establish that a learned
destroy selector would be identifiable under the present fixed-time regime.

## Completed fail-closed chain

The exact-hash metadata recovery and all downstream dependencies completed:

| stage | Slurm job | status | elapsed | result |
|---|---:|---|---:|---|
| recovery, deterministic merge, train gate and labels | `984885` | `COMPLETED` | 00:26:15 | exit `0:0` |
| development smoke | `984886` | `COMPLETED` | 00:00:13 | exit `0:0` |
| six-cell packed development | `984887` | `COMPLETED` | 13:18:30 | exit `0:0` |
| replay and aggregation | `984888` | `COMPLETED` | 00:05:14 | exit `0:0` |

Jobs `984887` and `984888` ran on `sist_gpu59`. The development allocation used
six CPUs and 24 GiB with no GPU. The six single-threaded workers were pinned to
logical CPUs `0,1,2,28,29,30`; OMP, MKL, OpenBLAS and NumExpr were all one
thread and `CUDA_VISIBLE_DEVICES` was empty. Finalization started only after
`afterok:984887` succeeded.

The six worker wall times were 0.38, 2.42, 3.18, 3.59, 6.53 and 13.31 hours.
Their sum was 29.41 CPU-hours versus 13.31 hours of development wall time, an
observed 2.21x speedup over serial execution. The 37% six-worker efficiency is
explained by the `scale` straggler; it is not evidence of oversubscription.

## Integrity and provenance

- six cells each produced 128 traces and 384 metric rows;
- total development matrix: 768 traces and 2,304 metric rows;
- 384 fixed-time and 384 fixed-iteration traces;
- all fixed-iteration traces completed exact 30; incomplete count zero;
- replay checked 12,484 search steps and passed;
- 48 instances belong to 24 independent `task_group_id` units;
- all failures remain in the denominator;
- no validation, frozen-test or stress split was generated or accessed;
- all records retain the `a4blnsd2` namespace and
  `SIM_GEOMETRIC_DEVELOPMENT_ONLY` evidence label.

Key hashes:

- corpus manifest:
  `dddb0db22febad8f10a8d15192ea8019f586f80035a882308a2794ca4737530c`;
- v2 config:
  `90b20a1335d48bac28252615284509d5993e9b21811f80175a5c53fe703c7cbc`;
- train gate:
  `530ac1bc5d4028ab512222e9e761978c6817bc2fe800918cef45b45a7c662eee`;
- label gate:
  `cd58fe27508613dcddf5d2d958e19da08d80aeca856c9cf8ae16e9249944c45a`;
- replay audit:
  `4794d12a5b74f9238a906c28830fc16f92a72cd3e611b765d5ab9c0705e676e1`;
- aggregated result:
  `9fbac9e69ff7bc5c897871c79873acd4536a862ea5c2bb318535307d06cf05d7`.

The aggregate retains a 32-entry source-hash map together with the train-gate,
label-gate and amendment hashes. Before corpus generation, 22 targeted, 89
affected and 155 explicitly non-frozen tests passed; before metadata recovery,
74 targeted and 108 affected non-frozen tests passed. No search or code change
was made after the final development result to alter these metrics.

The metadata-only recovery did not call initializer, destroy, repair,
scheduler, verifier or a search runner and did not rewrite the six preserved
calibration JSONL files. The earlier failed/cancelled attempts and logs remain
preserved and excluded.

## Train-only selections and label gate

The complete 312-trace train matrix contained 240/240 exact-30 traces and 72
fixed-time traces. ALNS tied random LNS at both the train exact-30 gate (75.0%)
and one-second gate (70.83%). Train-only selection fixed
`relatedness_destroy` as the best single operator and `online` as the ALNS
update scheme before development access.

The label gate passed with 24 state records, 151 improving candidates and zero
duplicate destroy sets. These records remain `search-generated neighborhood
improvement labels`, not expert actions.

## Development fixed-time result

All four methods returned the same verifier coverage at every registered
cutoff:

| method | 0.5 s | 1.0 s | 3.0 s |
|---|---:|---:|---:|
| random LNS | 72.92% | 72.92% | 72.92% |
| handcrafted round-robin | 72.92% | 72.92% | 72.92% |
| train-selected `relatedness_destroy` | 72.92% | 72.92% | 72.92% |
| adaptive ALNS | 72.92% | 72.92% | 72.92% |

The equality is identity-wise, not an aggregate cancellation: the same 70 of
96 instance/seed identities were feasible and the same 26 failed for every
method at 0.5, 1.0 and 3.0 seconds.

At one second the shared per-cell coverage was:

| cell | verified identities | coverage |
|---|---:|---:|
| `iid_small` | 16/16 | 100.0% |
| `iid_medium` | 16/16 | 100.0% |
| `dense_precedence` | 12/16 | 75.0% |
| `resource_bottleneck` | 14/16 | 87.5% |
| `tight_windows` | 8/16 | 50.0% |
| `scale` | 4/16 | 25.0% |

The fixed-time objective and normalized-primal-integral summaries do not
reverse this result. At three seconds random LNS had conditional objective
98.452 and normalized primal integral 0.4717, versus 98.804 and 0.4806 for
ALNS; lower is better. These are development-only descriptive differences and
were not accompanied by a preregistered superiority test.

## Development exact-iteration result

Every method completed exactly 30 neighborhoods:

| method | verified identities | coverage |
|---|---:|---:|
| random LNS | 70/96 | 72.92% |
| handcrafted round-robin | 72/96 | 75.00% |
| train-selected `relatedness_destroy` | 72/96 | 75.00% |
| adaptive ALNS | 73/96 | 76.04% |

Relative to random LNS, ALNS recovered three identities and lost none. The
three recoveries occurred in three independent groups, each changing one of
four variant/seed rows:

- `a4blnsd2-development-dense_precedence-group-002-v01`, seed 2203,
  first feasible at iteration 2 and 21.99 s;
- `a4blnsd2-development-dense_precedence-group-003-v00`, seed 2203,
  first feasible at iteration 11 and 123.63 s;
- `a4blnsd2-development-resource_bottleneck-group-000-v01`, seed 2909,
  first feasible at iteration 25 and 155.53 s.

At the independent-group level this is a descriptive +3.125 percentage-point
mean difference over 24 groups. The preregistered gate required only that ALNS
not be below random; it did not preregister a development superiority test.
Three discordant groups are insufficient for a new superiority claim.

Among the 70 identities feasible under both ALNS and random LNS at exact 30,
ALNS had a lower objective in 17, tied in 34 and had a higher objective in 19.
The result therefore does not show a stable conditional-quality advantage.

## Failure structure

### Initializer boundary

The shared `hybrid_load_balanced` initializer was already verifier-feasible on
70/96 identities. The other 26 identities were assignment incumbents whose
scheduler/verifier failure was `time_window_failure`:

| cell | initializer failures |
|---|---:|
| `dense_precedence` | 4 |
| `resource_bottleneck` | 2 |
| `tight_windows` | 8 |
| `scale` | 12 |

There were no initializer failures in the two IID cells. No fallback label was
used or polluted; rows distinguish `hybrid_load_balanced` from
`hybrid_load_balanced_assignment_incumbent`.

### Search failure modes

The retained failure library contains 608 failed metric rows across methods
and snapshots, not 608 independent instances:

- `precedence_failure`: 276 rows;
- `time_window_failure`: 332 rows.

Candidate-level failed neighborhood evaluations contain 9,173
`precedence_failure` events and 512 `time_window_failure` events. This shows a
failure transition: all unrecovered searches begin from a time-window-invalid
initializer, while many destroy/repair moves convert the active terminal
diagnostic to precedence failure instead of reaching feasibility. The counts
are event counts and must not be reported as independent samples.

### Runtime opportunity bottleneck

Fixed-time traces completed a median of zero neighborhoods for random and the
single operator and one neighborhood for round-robin and ALNS. Their median
return time was about 4.14--4.24 s because a repair begun before the three-second
cutoff could return later; the evaluator correctly credits only incumbents
verified before the monotonic cutoff.

Exact-30 traces took a median of approximately 193--200 s per method. Median
repair time per neighborhood was 5.76--6.34 s, and the median candidate count
reached the shared cap of 256. Thus the fixed-time comparison usually exposes
zero or one destroy decision. The absence of fixed-time method separation is
therefore consistent with repair-dominated search opportunity, not evidence
that all neighborhood selectors are intrinsically equivalent.

## Closure decision

The following statements are supported:

1. evaluator cutoff/provenance, exact-iteration, replay and failure-denominator
   semantics pass;
2. ordinary LNS/ALNS and the label interface are technically reusable;
3. ALNS satisfies the registered non-systematic-weakness gate by tying random
   at one second and exceeding it descriptively by three identities at exact
   30;
4. the present 0.5/1.0/3.0-second regime does not provide enough completed
   neighborhoods to evaluate neighborhood-selection efficiency.

The v2 closure therefore does **not** authorize learned destroy training. The
formal action remains `HOLD_A4B_LEARNED_DESTROY_TRAINING`.

## Whether to preregister a stronger ordinary-search recovery

If A4b is continued, the evidence supports a separate stronger ordinary-search
recovery protocol before any learned method. This is a recommendation, not a
frozen protocol or authorization to run it.

A defensible follow-up must:

1. use a new master seed, ID prefix and train/development namespace with zero
   overlap; v2 development may diagnose the problem but may not select v3
   parameters;
2. retain the same initializer, atomic-unit definition, acceptance,
   scheduler/verifier, failure denominator and 0.5/1.0/3.0-second primary
   cutoffs;
3. first optimize repair implementation only under exact transition-parity
   tests: identical candidate set/order, selected state, verifier result and
   trace signature on fixtures and fresh train cases;
4. if semantic-preserving acceleration is insufficient, preregister any
   feasibility-first repair or failure-aware neighborhood change as a method
   change shared by every future ordinary and learned selector;
5. freeze on fixtures and fresh train a minimum search-opportunity gate,
   positive initializer-failure recovery gate, per-cell non-regression gate,
   candidate-cap saturation diagnostic and runtime decomposition before the
   new development split is accessed;
6. keep exact-iteration evidence as a secondary mechanism diagnostic and do
   not hide a repair bottleneck by merely increasing the primary time budget;
7. require ordinary ALNS to produce non-degenerate fixed-time recovery and
   operator-dependent outcomes before generating a Neural LNS checkpoint.

If the project does not wish to invest in this ordinary-search recovery, the
scientifically clean alternative is to stop A4b at this completed engineering
foundation and keep the first-paper learned claim limited to the immutable
A3.5 matched-decoder result.

## Claim boundaries

This closure supports no claim of learned-neighborhood superiority, real-robot
performance, collision safety, dynamic insertion, path planning, physical
quality or sim-to-real transfer. It creates no A4b frozen benchmark and no
model checkpoint. A new protocol requires explicit user authorization.
