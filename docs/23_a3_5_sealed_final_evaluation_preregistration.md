# A3.5 one-time sealed final evaluation preregistration

Status: **FROZEN BEFORE FINAL GENERATOR IMPLEMENTATION OR DATA GENERATION**  
Date: 2026-08-11  
Evidence: **SIM_GEOMETRIC**  
Configuration: `configs/allocation/a3_5_sealed_final_v1.json`  
Configuration SHA-256:
`8c4d3cb7cc6e61ee589e98786ab78e77958d09694afb102d7f806eda9c208368`

## 1. Confirmatory question

The sole primary hypothesis is:

> Does a dynamic feasible Pair-Pointer significantly improve the final
> feasibility rate of continuous-process allocation candidates relative to a
> static parallel decoder under the same fixed heterogeneous graph encoder?

This is not a test of whether Pair-Pointer exceeds CP-SAT, MILP or LNS. Those
methods remain strong secondary quality–time references. A valid outcome may
therefore be “the Pair-Pointer improves the matched static learned decoder but
does not exceed strong optimisation baselines.”

The failed A3 v4 result remains immutable. This protocol neither reruns nor
reinterprets v4 and cannot be used to rescue `A3_FINAL_FAILED_BASELINE_FLOOR`.

## 2. Fixed learned methods

The two confirmatory methods share the same heterogeneous-GNN encoder family,
hidden dimension 64, two message-passing layers, training corpus, train-only
normaliser and seeds 101/211/307:

1. `hetero_gnn_pair_pointer`: fixed autoregressive atomic-unit–robot pair
   decoder, hard feasibility mask and deterministic greedy rollout;
2. `hetero_gnn_static`: fixed parallel assignment logits, static order score
   and deterministic precedence-aware decoding.

All six checkpoint file hashes and state-dictionary hashes are recorded in the
machine-readable configuration. They are the checkpoints selected before the
A3.5 pilot validation results were closed. Retraining, ensembling, best-seed
selection, beam search, repair, test-time adaptation and RL are forbidden.

The locked training evidence is:

- pilot config SHA-256:
  `876933101dc5ed2e56984d8666dae36509481e93d6d6422ec4eed5380a7bea17`;
- pilot manifest file SHA-256:
  `3a5060e4ac089835c216d44d95fc61768a53b9b174148c3d3aa0275ec1d4b3e1`;
- train/validation access SHA-256:
  `dc9ca758bf09480805a078fd46fac2081fdf8aef95cfe2e28a4540b6711c87bb`;
- normaliser SHA-256:
  `97bba5933739656645fb8f0ca0d555317517fbdcbdbcc586261ab06040ffed8c`.

The train-only categorical vocabulary SHA-256 is
`40ee1fc3baf732487ededa7cc250b276dabe28461d310e7c332d7d847186aeaf`.

The architecture, graph construction, pointer, scheduler, verifier and oracle
source hashes are also locked in the configuration. Any change requires a new
protocol version and is not allowed after final data generation.

## 3. Untouched benchmark

The new protocol ID is `a3_5_sealed_final_v1`, master seed `913751`, ID prefix
`a35f1`. It will contain only one `frozen_test` split:

- six cells: IID-small, IID-medium, dense precedence, resource bottleneck,
  tight windows and scale;
- 12 independent task groups per cell, 72 groups total;
- two variants per group, 144 instances total;
- no stress split and no added negative-control cell.

Cell generator parameters remain identical to the A3.5 pilot except for the
new seed, namespace, split and group count. Workpiece, layout, task-group,
parent-curve and instance IDs must be disjoint from v2, v3, v4 and
`a3_5_pointer_pilot_v1`. Hash/leakage failure invalidates the run.

The final generator and evaluator do not yet exist. They may be implemented and
tested only with A0/A1 fixtures and the A3.5 train/validation corpus. Before
generation, their exact source hashes, dependency versions and Slurm command
must be written into a separate immutable seal. Only after that seal may the
new benchmark be generated. Candidate predictions must be persisted before any
audit witness is opened. The evaluation command may then be invoked exactly
once.

## 4. Primary estimand and test

The independent statistical unit is `task_group_id`, not an instance variant,
action, segment or model seed. For each group, compute the binary verified
outcome difference Pair-Pointer minus matched static decoder for the same
variant and same seed, then average over its two variants and three seed pairs.
The primary estimand is the unweighted mean of these 72 group differences.

The directional confirmatory test is a one-sided paired sign-flip randomisation
test over group differences:

- alternative: mean Pair-Pointer coverage difference is greater than zero;
- alpha: 0.05;
- 100,000 Monte Carlo sign flips; seed `451903`;
- two-sided 95% group-cluster bootstrap percentile interval;
- 20,000 bootstrap draws; seed `451907`.

The primary decoder hypothesis is supported only if:

1. all source, checkpoint, manifest, access, leakage, mask and atomicity checks
   pass;
2. mean paired coverage difference is positive;
3. one-sided randomisation-test p-value is below 0.05;
4. the 95% cluster-bootstrap lower bound is above zero;
5. at least two of three seed-wise paired coverage differences are positive.

Every candidate failure remains in the coverage denominator. No cell-wise test
can replace or override the primary test.

## 5. Secondary evidence

The following are secondary and cannot convert a failed primary result into a
success:

- overall and six-cell coverage by seed, with group-clustered intervals;
- a robustness flag requiring no cell regression greater than one cell-group
  equivalent, `1/12 = 0.083333`;
- decoder dead-end, incomplete assignment, precedence, time-window,
  shared-resource, schedule and integrity failures;
- makespan, load imbalance, travel/setup time and conditional weighted proxy
  score, always accompanied by coverage;
- per-instance median/IQR and total runtime.

Three fixed strong baselines are retained: `hybrid_assignment_milp` with a 3 s
limit, `order_aware_lns` with 100 iterations and seed 0, and
`hybrid_load_balanced`. Pair-Pointer minus baseline coverage differences and
95% group-cluster intervals will be reported. A secondary non-inferiority
margin of one overall frozen group, `1/72 = 0.013889`, is fixed. Strong
advantage may be stated only if the lower 95% bound is above zero; neither
condition is required for the primary decoder hypothesis.

Runtime is measured on one recorded CPU node class and one CPU thread. Model
loading is excluded; graph encoding, decoding, scheduling and verification are
included. There is no frozen-data warm-up or adaptation. The quality–time
trade-off must be reported even if coverage improves.

## 6. Teacher-label interpretation

The frozen benchmark has no teacher-forced accuracy metric because no label is
needed to evaluate the fixed candidate policies. The 21.8% pilot pair accuracy
remains a training diagnostic only. It does not contradict higher verifier
coverage: the combinatorial problem can admit multiple valid assignments and
serialisations, while cross-entropy scores one canonical sequence.

Training labels must be described as **heterogeneous solver-generated verified
incumbents**. They comprise MILP, order-aware LNS and declared constructive
fallback plans. They must not be called “LNS expert solutions,” global optima,
real expert actions or demonstrated factory policies.

## 7. Result classes and immutable wording

- `A3_5_FINAL_INVALID`: any integrity or premature sealed-access failure.
- `A3_5_DECODER_HYPOTHESIS_NOT_SUPPORTED`: valid run, primary test fails.
- `A3_5_DECODER_HYPOTHESIS_SUPPORTED_HETEROGENEOUS`: primary test passes but
  the no-large-cell-regression robustness flag fails.
- `A3_5_DECODER_HYPOTHESIS_SUPPORTED`: primary test and robustness flag pass.
- `A3_5_STRONG_BASELINE_ADVANTAGE`: primary result passes and Pair-Pointer's
  lower confidence bound exceeds every strong baseline.

Failure against MILP/LNS does not erase a supported matched-decoder result. It
must instead be stated directly: “the autoregressive decoder improved the
matched static learned decoder but did not exceed the strong optimisation
baseline.” Conversely, a quality/runtime advantage on successful candidates
cannot rescue failed coverage.

## 8. Boundaries and next decision

All evidence remains `SIM_GEOMETRIC`. The protocol provides no evidence of real
robot execution, real production deployment, collision safety, sim-to-real
success or physical process quality. It includes no repair, beam search, RL,
physical model or complex multi-arm collision planner.

A4 remains closed through the one-time evaluation. After closure, the result
will determine whether Pair-Pointer is retained as a learned-decoder
contribution, used only as a solver warm-start candidate, or recorded as a
negative ablation while solver/LNS remains the main line.
