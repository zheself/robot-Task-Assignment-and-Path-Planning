# A3.5 sealed final evaluation closure

Date: 2026-08-11  
Evidence: **SIM_GEOMETRIC**  
Immutable result: **`A3_5_DECODER_HYPOTHESIS_SUPPORTED`**

## Execution chronology

The registered order was followed without changing model, data, seed, threshold
or decoder:

1. implemented the isolated final generator/evaluator;
2. passed 13 targeted tests and the 146-test non-v4 regression;
3. passed validation-only Slurm preflight `941015` with 12 instances and 108
   rows;
4. wrote source/dependency/command seal
   `64b670831fb5f863462253cf3b79ec9daf96b45f319734ca06006e8db29bcc62`;
5. verified benchmark and evaluation directories were absent;
6. generated the new `a35f1` benchmark once in Slurm `941022`;
7. recorded 72 groups/144 instances and manifest hashes;
8. invoked the sealed evaluation once in Slurm `941024`;
9. classified the unchanged output using the preregistered statistics.

No v4 instance/witness, A4 repair, beam search, RL, retraining, checkpoint
selection or test-time adaptation was used.

## Confirmatory result

Pair-Pointer coverage was 65.05% and matched static-decoder coverage was 40.28%.
The independent group-paired difference was +24.77 percentage points, with 95%
cluster-bootstrap CI [+18.52, +31.48] and one-sided sign-flip p = 0.0000099999.
All three fixed seeds improved and all six difficulty cells were non-regressing.
The primary test and robustness flag therefore passed.

This reproduces the direction of the development result (+27.78 points) on an
untouched final corpus, with a somewhat smaller +24.77-point effect. It supports
the architectural claim that autoregressive dynamic pair decisions improve the
matched static learned decoder's un-repaired verified feasibility.

## Secondary strong-baseline result

Pair-Pointer did not meet the preregistered secondary non-inferiority margin:

- hybrid assignment MILP: 72.22% coverage;
- order-aware LNS: 79.86%;
- hybrid load-balanced: 68.75%.

Pair-Pointer minus baseline intervals were [-13.66, -0.92], [-21.30, -8.56]
and [-10.19, +2.55] percentage points respectively. Therefore the final paper
claim must be “improves matched static decoding but remains below strong
optimisation,” not “outperforms CP-SAT/MILP/LNS” or “is competitively
equivalent.”

The 0.446 s median Pair-Pointer runtime was lower than the registered MILP
(0.665 s) and LNS (0.499 s), but far above static decoding (0.00613 s) and the
load-balanced heuristic (0.0201 s). Report the full coverage–runtime trade-off.

## Integrity and immutable boundaries

All five final integrity checks passed. The evaluator retained 523 failed rows;
all were `schedule_infeasible`, with no dead-end, hard-mask or atomic-unit
violation. Candidate predictions were saved before witness access, and every
witness/hash audit passed.

The final result does not reopen A3 v4. It is not evidence for motion/collision
safety, real industrial allocation, real hemming, sim-to-real transfer or
physical quality. A4 remains not implemented; whether to pursue solver
warm-start/repair is a later scope decision, not part of this result.

Compact evidence is in
`reports/phase1_allocation/a3_5_sealed_final_v1_results.md`, with JSON and CSV
companions. Raw predictions, rows, failure library, benchmark, manifest and
witnesses remain under ignored `outputs/phase1_allocation/`.
