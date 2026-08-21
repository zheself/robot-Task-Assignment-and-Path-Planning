# A3 one-time frozen evaluation preregistration

Status: **FROZEN BEFORE A3 FROZEN-TEST/STRESS ACCESS**  
Frozen date: 2026-08-10  
Evidence label: **SIM_GEOMETRIC**

This document and `configs/allocation/a3_final_evaluation_v1.json` govern the
single final A3 evaluation. Writing this protocol does not authorise model
tuning, A4 repair, path planning or RL. At the time of freezing, no A3 model
prediction or metric has been computed on v4 `frozen_test` or `stress`.

## 1. Question and known limitation

The final question is whether the validation-selected learned edge-scoring
model produces complete allocations and schedules that pass the unchanged A1
proxy verifier on unseen v4 groups, and how it compares with fixed heuristic,
MILP and LNS baselines.

The A2 v4 frozen outcomes of the non-learning baselines were already observed
while qualifying the benchmark. This is therefore not a fully blinded new
benchmark for those baselines. The protection in A3 is narrower and auditable:
the learned architecture, three checkpoints, decoder, metrics and decision
rules were selected using train/validation only and are frozen here before the
learned model accesses frozen-test/stress. All baselines will be rerun in the
same final invocation; old A2 results are provenance cross-checks only.

## 2. Locked provenance and model

The following objects must match before data access:

| Object | Frozen value |
|---|---|
| Final-evaluation config | `ce574b6b62c2218f8a2f7b3130646444cc00b60ccc69c6822bdde5d3f48ab756` |
| A2 v4 manifest | `0c98f30e92697ce8b5eca724df0f7d1b7053293df1e792707487ecb6c71b5398` |
| A2 benchmark config | `66ca0402ff933f37edff5228c234a7b3ed868f97c62b53edad9ba8c283d7e34b` |
| A3 development config | `33dd1c5e22960823f483c0fec4e40b8cb98b0b30fbf9bb5456b4132da6122034` |
| W10 summary | `06b44d99611974e025063e249373377d4d62f3d6aa191f4957dee6d8d8d4dbda` |
| Train-only vocabulary | `a031d82d29b1d466a8bd4ecd3701a24c2fa89c960859c6798e318ef41ff58aa1` |
| Train-only normaliser | `c1e93d70d6f9a4ffd93f42c638455f2bb12a40e9615fb1125c3d4f83714abb66` |

The selected family is exactly `edge_mlp`. It is an ablation and must not be
renamed a GNN or graph Transformer. All validation-selected checkpoints are
evaluated; no best-seed selection, retraining, averaging, ensemble, test-time
adaptation or fallback is allowed.

| Seed | Best epoch | State SHA-256 | Checkpoint-file SHA-256 |
|---:|---:|---|---|
| 17 | 12 | `ad18413e210e1e671c9fa8f0d3eb3834814735336ffe2a9ca15aa4220f2e6407` | `01589765bad6168270f0ec4cb2ce501706460e3147bd9099309c477142049cd5` |
| 29 | 23 | `9079b4d34818ad08e07b8ff173e9def5556dade8ef88f8cd23d357204beece96` | `bec663a0fee319c2c37d97a2fc6bfa0345fbfd7754806354486346744d0dfe88` |
| 43 | 21 | `a92c4460c1d26a97d8c6307610599463890442cf074664cea77457dd6772248d` | `0694c1b29285e337bae79903f772fa4eb0c44a4bffaa23db6fe7572ff0ed44cb` |

Each model emits masked segment–robot logits and segment order priorities. The
unchanged atomic-unit decoder, precedence-aware robot-local ordering, A1 list
scheduler and verifier produce the evaluated plan. No repair is applied; a
failed schedule remains a failure.

The machine-readable protocol also freezes SHA-256 values for the current
model, decoder, scheduler, verifier, oracle configuration and all eight
baseline implementations. The final evaluator does not yet exist; it must be
implemented and tested using fixtures plus train/validation only, then its hash
must be written to the sealed run manifest before the first frozen access. It
may orchestrate these locked components but may not alter their semantics.

## 3. One-time data access

The primary split contains 144 ordinary frozen-test instances: six difficulty
cells, 12 independent `task_group_id` values per cell and two variants per
group. Evaluation order is lexicographic and every method/checkpoint must see
the same ordered instances.

The 24 stress instances are processed in the same sealed invocation only after
the primary predictions have been saved. Stress is descriptive. The exception
is `designed_edge_infeasible`: all six negative controls must be rejected;
returning a verified feasible plan is a constraint-handling integrity failure.

Witness plans are never supplied to the model, decoder or baseline. They may
be read only after prediction for manifest/hash verification and optional
teacher-agreement diagnostics. They are constructive feasible A1-proxy plans,
not optimum or real expert labels.

## 4. Fixed comparison methods and budgets

All methods use the same analytical oracle, candidate scheduler and verifier.

| Role | Methods | Fixed budget/policy |
|---|---|---|
| Context/weak | greedy, load-balanced greedy, Hungarian+sequencing, assignment MILP+sequencing, deterministic LNS | MILP 3 s and zero relative gap; LNS 100 iterations, seed 0 |
| Strong | hybrid load-balanced, hybrid assignment MILP, order-aware LNS | unchanged v4 policy; MILP 3 s/zero gap; LNS 100 iterations, seed 0 |
| Learned | three locked `edge_mlp` checkpoints | one forward/decode/schedule/verify pass; no repair or fallback |

The budgets are algorithm-specific registered budgets, not an assertion of
equal wall-clock time. Full pipeline runtime is measured for every instance;
training cost is reported separately. Runtime alone cannot establish method
superiority.

## 5. Metrics and statistical units

The primary metric is **verified-candidate coverage**. A candidate counts only
when it covers every segment exactly once, respects atomic-unit/hard-mask
semantics, schedules successfully and passes the unchanged verifier. Failed
candidates stay in the denominator.

The independent unit is `task_group_id`. The two variants are averaged within
each group. For the learned method, the headline estimate averages the group
rates of all three fixed seeds; seeds and variants are not treated as additional
independent samples. Each seed is also reported separately, including its worst
cell, so instability cannot be hidden.

Secondary metrics are the weighted proxy objective and its makespan, load
variance, travel/setup and priority-tardiness components; full-pipeline runtime;
failure reason/status; and witness assignment agreement as a diagnostic only.
The objective is lower-is-better. Quality is compared only on pairwise jointly
verified group means; failures receive no imputed objective.

Paired task-group bootstrap uses seed 260904, 5,000 resamples and 95% confidence
intervals. Overall learned-versus-strong coverage tests form one family of three
one-sided paired Wilcoxon tests with Holm correction at α=0.05. Conditional
quality uses paired Wilcoxon tests on jointly verified groups and is interpreted
only after coverage. Cell-level inference is descriptive; the fixed gates below
are not replaced by post-hoc p-values.

## 6. Frozen per-difficulty gates

The learned value in this table is the mean across the three fixed seed group
rates. All six absolute thresholds must pass.

| Frozen cell | Absolute coverage floor | Intended diagnostic |
|---|---:|---|
| `iid_small` | 0.95 | small in-distribution generalisation |
| `iid_medium` | 0.85 | medium in-distribution generalisation |
| `ood_dense_precedence` | 0.50 | dense precedence robustness |
| `ood_resource_bottleneck` | 0.50 | shared-resource robustness |
| `ood_tight_windows` | 0.50 | tight-window robustness |
| `ood_scale` | 0.50 | unseen scale robustness |

Two relative floors also apply:

1. Overall learned coverage must be at least the best overall context/weak
   baseline coverage.
2. In every cell, learned coverage must be at least the best context/weak
   baseline coverage minus `1/12 = 0.08333`, one whole task-group equivalent.

The same one-group-equivalent margin is used only as a cell robustness limit in
strong-baseline classifications; it is not changed after seeing outcomes.

## 7. Frozen decision and wording rules

Classification is hierarchical:

1. **`A3_FINAL_INVALID`** — any source/checkpoint hash mismatch, incomplete
   method-instance matrix, schema/manifest failure, NaN/unexpected exception,
   hard-mask or atomic-unit violation, or negative-control detection below
   100%. Required wording: *the final run is invalid; no A3 performance
   conclusion is available*.
2. **`A3_FINAL_FAILED_BASELINE_FLOOR`** — integrity passes but any absolute or
   weak-relative floor fails. Required wording: *the fixed learned model failed
   the preregistered feasibility floor; the learned A3 branch is not supported
   and is not retuned on v4*.
3. **`A3_FINAL_BASELINE_FLOOR_ONLY`** — the A3 minimum floor passes, but the
   learned model is not competitive with a strong baseline. Required wording:
   *the learned model clears weak baselines but shows no strong-baseline
   advantage*.
4. **`A3_FINAL_COMPETITIVE_NOT_SUPERIOR`** — versus at least one strong method,
   the overall paired coverage-difference CI lower bound is at least −0.05 and
   no cell deficit exceeds 1/12, but strict superiority fails. Required wording:
   *the learned model is competitive within the registered margin; superiority
   was not established*.
5. **`A3_FINAL_STRONG_ADVANTAGE`** — against all three strong methods, the
   overall paired coverage-difference 95% CI lower bound is above zero, the
   Holm-adjusted one-sided p-value is below 0.05, and no cell deficit exceeds
   1/12. Required wording: *the validation-selected learned edge-scoring model
   improves A1-proxy verified coverage under the registered SIM_GEOMETRIC
   protocol*.

Passing the A3 final gate means integrity plus all absolute and weak-baseline
floors; it does not automatically mean strong superiority. Quality superiority
may be discussed only if the applicable coverage classification passes and the
jointly verified sample count is reported.

No outcome permits the phrases “GNN superiority,” “collision-safe,” “formal
safety,” “real production deployment,” “real hemming validation,” “sim-to-real
success” or “physical-quality improvement.” Even the strongest outcome refers
only to `edge_mlp` and the A1 geometric/timing proxy.

## 8. Failure, rerun and reporting policy

Performance-based reruns are forbidden. An exact rerun is allowed only for a
documented infrastructure/evaluator failure, with all failed logs retained and
without changing model, decoder, baseline, budget, threshold or metric. Any
such change requires a new protocol version and cannot restore pristine unseen
status after results have been inspected.

The sealed run must save configuration checksum, environment versions, Slurm
job IDs, all method/seed predictions, per-instance verifier status, per-seed and
per-cell metrics, paired statistics, failure library and a machine-generated
decision record. Raw outputs/checkpoints stay in ignored `outputs/`; compact
JSON, CSV and Markdown go to `reports/phase1_allocation/`.

A4 remains blocked until the final result class and its allowed wording are
recorded. A failure may motivate a future solver-only A4 plan or a genuinely new
benchmark, but cannot be repaired by tuning on this v4 frozen result.

## 9. Scientific boundary

All instances are programmatic continuous-workcell geometry. The oracle and
verifier do not prove continuous robot motion or collision safety and contain
no stress, contact, plasticity or process-quality model. This evaluation
contains no path planning, robot execution, physical experiment or RL.
