# A3 W10 train/validation development result

Status: **DEVELOPMENT GATE PASSED; FINAL FROZEN EVALUATION NOT STARTED**  
Date: 2026-08-10  
Evidence label: **SIM_GEOMETRIC**

## 1. Protocol completion

W10 executed the frozen `configs/allocation/a3_development_v1.json` matrix:

- 192 v4 train instances for vocabulary, normalisation and parameter fitting;
- 48 v4 validation instances for checkpoint and family selection;
- `edge_mlp`, `hetero_gnn` and `graph_transformer` × seeds 17, 29 and 43;
- at most 60 epochs with patience 10 and the registered lexicographic
  checkpoint rule;
- unchanged hard-mask decoder, A1 scheduler and verifier, without repair.

All nine shards completed with matching config, data-access, vocabulary,
normaliser and A2-manifest hashes. The loader and run outputs confirm that
neither `frozen_test` nor `stress` was accessed. The Slurm jobs ran on CPU; a
compute-node preflight verified the isolated project environment and the exact
192/48 record counts before training.

## 2. Validation result

| Family | coverage mean / minimum | atomic-unit accuracy | conditional proxy score | median full-pipeline time | failures |
|---|---:|---:|---:|---:|---:|
| `edge_mlp` | **0.958 / 0.958** | 0.602 | **86.568** | 0.0128 s | 6 |
| `hetero_gnn` | 0.896 / 0.875 | **0.611** | 87.806 | 0.0145 s | 15 |
| `graph_transformer` | 0.889 / 0.875 | 0.605 | 87.171 | 0.0147 s | 16 |

The registered primary metric is verified-candidate coverage, so W10 selects
`edge_mlp`. This is an ablation family, not a GNN. Its three seeds each verify
46 of 48 validation candidates. Most remaining difficulty is concentrated in
the precedence and resource cells.

The comparison with the unchanged validation baselines is mixed:

- W10 `edge_mlp`: 95.8% coverage;
- weak assignment-first baselines: 83.3–85.4%;
- `hybrid_assignment_milp`: 97.9%;
- `hybrid_load_balanced` and `order_aware_lns`: 100%.

The proxy score is a minimisation objective. It is computed only over verified
candidates and must not be treated as an unconditional quality ranking when
coverage differs.

## 3. Decision

W10 passes its engineering and development-selection checks. It authorises a
new final-evaluation preregistration; it does not itself pass A3. The final
protocol should freeze `edge_mlp`, retain all three registered selected
checkpoints, compare them against all registered strong baselines under equal
instances and unchanged verifier semantics, and declare aggregate and
per-difficulty success/failure rules before any frozen split is opened.

The current result is negative for the proposed GNN-superiority narrative:
neither graph model was selected and the selected learned model did not match
the strongest non-learning baselines on validation. This result must be
preserved. A4 repair or extra tuning must not be used retroactively to alter
W10 selection.

## 4. Claim boundary

W10 provides only train/validation evidence in a data-generated geometric
allocation proxy. It is not a frozen-test result, motion-level collision proof,
real multi-robot execution, industrial deployment, physical process model or
quality-improvement result. The constructive witness is a feasible A1-proxy
teacher, not an optimum or real expert demonstration.

Machine-readable evidence is in
`reports/phase1_allocation/a3_w10_development_v1_summary.json`; compact tables
and the generated interpretation are in
`reports/phase1_allocation/a3_w10_development_v1_results.md`.
