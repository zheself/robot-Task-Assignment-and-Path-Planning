# A3 one-time frozen evaluation closure

Status: **A3_FINAL_FAILED_BASELINE_FLOOR — V4 LEARNED BRANCH CLOSED**  
Date: 2026-08-10  
Evidence: **SIM_GEOMETRIC**

## Protocol integrity

The evaluator was implemented and tested using fixtures plus the isolated 192
train/48 validation development export. Targeted tests passed 17/17, the full
project regression passed 136 tests with one existing Gymnasium warning, and a
CPU-node validation smoke completed 44/44 registered method rows. Only then was
the evaluator sealed and the single frozen invocation submitted.

- Protocol SHA-256: `ce574b6b62c2218f8a2f7b3130646444cc00b60ccc69c6822bdde5d3f48ab756`
- Evaluator-seal SHA-256: `9a357b85f5c5548af9ea277b403e11ec392d586e70fbf757ce31c3a11ff2f1bf`
- A2 manifest: `0c98f30e92697ce8b5eca724df0f7d1b7053293df1e792707487ecb6c71b5398`
- Slurm job: `940906`, `COMPLETED`, exit 0, 10 min 38 s
- Matrix: 144 frozen-test + 24 stress instances × 11 methods = 1,848 rows

All registered source/checkpoint hashes, method-instance counts, schema,
manifest, finite metric, hard-mask, atomic-unit, witness and designed-infeasible
negative-control checks passed. Predictions were persisted before witnesses
were opened for audit.

## Frozen result

| Method | Overall frozen verified coverage |
|---|---:|
| `edge_mlp`, seed 17 | 55.6% |
| `edge_mlp`, seed 29 | 58.3% |
| `edge_mlp`, seed 43 | 56.3% |
| `edge_mlp`, registered seed mean | **56.7%** |
| best context/weak baseline | 55.6% |
| hybrid load-balanced | 68.8% |
| hybrid assignment MILP | 73.6% |
| order-aware LNS | 79.2% |

| Frozen cell | Learned seed-mean coverage | Registered floor | Result |
|---|---:|---:|---|
| IID small | 97.2% | 95% | pass |
| IID medium | 90.3% | 85% | pass |
| OOD dense precedence | 40.3% | 50% | **fail** |
| OOD resource bottleneck | 59.7% | 50% | pass |
| OOD tight windows | 37.5% | 50% | **fail** |
| OOD scale | 15.3% | 50% | **fail** |

The learned method also failed the cell-relative weak-baseline rule on dense
precedence. Its overall paired coverage differences versus hybrid
load-balanced, hybrid assignment MILP and order-aware LNS were respectively
−12.0, −16.9 and −22.5 percentage points; all three 95% confidence intervals
were strictly below zero.

On jointly verified groups, the learned method sometimes had a lower proxy
objective than hybrid load-balanced/MILP, but this conditional subset excludes
many learned failures. Under the preregistered coverage-first protocol it cannot
reverse the failed conclusion.

## Required conclusion

The exact preregistered wording applies:

> The fixed learned model failed the absolute or weak-baseline feasibility
> floor; stop the learned A3 branch without retuning on v4.

This does not mean graph learning is impossible in the broader research
problem. It means the current constructive-witness imitation, edge-MLP
selection and no-repair decoder do not generalise adequately to the registered
constraint and scale shifts. The heterogeneous GNN and graph Transformer were
already rejected on validation and cannot be substituted after seeing frozen
results.

## Next decision boundary

No further v4 model selection, hyperparameter change, seed selection, repair or
performance rerun is permitted. A4 remains blocked until one route is chosen
and separately planned:

1. continue the first-paper engineering line with the strong solver/LNS methods
   and treat learning as a documented negative result; or
2. formulate a materially different learned role, such as solver warm-start or
   repair proposal, and evaluate it only on a new untouched benchmark version.

Neither route changes the current A3 result. No conclusion here concerns motion
planning, collision safety, real robot execution, physical process quality or
sim-to-real transfer.
