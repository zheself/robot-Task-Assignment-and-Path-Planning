# A3.5 one-time sealed final results

Evidence: **SIM_GEOMETRIC**. Result class:
**`A3_5_DECODER_HYPOTHESIS_SUPPORTED`**.

The unique sealed evaluation supports the preregistered decoder hypothesis:
the fixed hetero-GNN Feasible-Pair Pointer improves final A1-verifier coverage
over the matched fixed hetero-GNN static decoder. It does not establish
non-inferiority or superiority to the strong optimisation baselines.

## Primary result

| Method | Frozen coverage | Median runtime |
|---|---:|---:|
| Pair-Pointer, three fixed seeds | **65.05%** | 0.4460 s |
| Matched static decoder, three fixed seeds | 40.28% | 0.00613 s |

- Group-paired improvement: **+24.77 percentage points**.
- 95% task-group cluster-bootstrap CI: **[+18.52, +31.48] points**.
- One-sided sign-flip randomisation test: **p = 0.0000099999**.
- Seed differences: +22.22, +29.17 and +22.92 points for seeds 101, 211
  and 307.
- All six cells were non-regressing; the preregistered robustness flag passed.
- Zero source/checkpoint/manifest/witness, hard-mask or atomicity failure.

## Difficulty cells

| Cell | Pair-Pointer | Static | Difference |
|---|---:|---:|---:|
| dense precedence | 68.06% | 19.44% | +48.61 points |
| IID medium | 87.50% | 75.00% | +12.50 points |
| IID small | 100.00% | 100.00% | 0.00 points |
| resource bottleneck | 61.11% | 26.39% | +34.72 points |
| scale | 31.94% | 6.94% | +25.00 points |
| tight windows | 41.67% | 13.89% | +27.78 points |

## Strong baselines and quality–time evidence

| Method | Coverage | Median runtime | Pair-Pointer minus baseline 95% CI |
|---|---:|---:|---:|
| hybrid assignment MILP | 72.22% | 0.6652 s | [-13.66, -0.92] points |
| order-aware LNS | **79.86%** | 0.4993 s | [-21.30, -8.56] points |
| hybrid load-balanced | 68.75% | 0.0201 s | [-10.19, +2.55] points |

Pair-Pointer failed the secondary one-group non-inferiority margin against all
three strong baselines. Its median runtime is slightly below the registered
MILP and LNS runs, but about 73 times the matched static decoder and 22 times
the load-balanced heuristic. Conditional proxy scores use different successful
subsets and cannot reverse coverage or establish quality superiority.

## Integrity and failures

The matrix contains 1,296 rows: 144 instances × nine fixed methods. All
registered checks passed. All 523 failed method-instance rows were retained and
classified `schedule_infeasible`: 151 Pair-Pointer, 258 static, and 114 strong-
baseline failures. There were no decoder dead-ends or mask/atomic-unit failures.

## Sealed provenance

- Validation-only preflight: Slurm `941015`, 12 instances/108 rows, passed.
- Source/evaluator seal SHA-256:
  `64b670831fb5f863462253cf3b79ec9daf96b45f319734ca06006e8db29bcc62`.
- Frozen generation: Slurm `941022`, invoked once.
- Manifest internal SHA-256:
  `fe8fa5caf997bca742a794abee6168867de012f624df5558aaaf93a1b5935a6f`.
- Manifest file SHA-256:
  `13b5d093d6edd863128f5dd683f1ff98ce21185c9deac1165e591bc0ae4dd400`.
- Sealed evaluation: Slurm `941024`, invoked once, exit `0:0`.

## Permitted conclusion

> On the untouched SIM_GEOMETRIC benchmark, the dynamic Feasible-Pair Pointer
> significantly improved final verifier coverage over the matched static
> decoder under the same heterogeneous graph encoder, but remained below the
> registered strong optimisation baselines.

This result does not alter the failed A3 v4 conclusion and provides no real
robot, production, collision-safety, sim-to-real or physical-quality evidence.
A4 repair was not used and remains a separate future decision.
