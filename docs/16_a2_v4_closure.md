# A2 v4 closure record

Decision date: 2026-08-10. Gate: **PASSED_AND_FROZEN_V4**.

## Decision chain

V2 and v3 remain failed, immutable diagnostics because their ordinary-instance policy proved only edge-mask coverage, not joint schedule feasibility. Two stronger search developments were then attempted using allowed development data only. The earlier beam-ALNS tied validation coverage; the later assignment-beam regressed the precedence validation cell. Both failed their registered gates and are retained as negative method-development results.

A separate benchmark-integrity protocol constructed an A1-proxy schedule, reconstructed only time windows around it, and independently verified/hash-bound the resulting witness. It passed all 240 v3 train/validation instances with deterministic output and preservation of geometry, capability, tool, precedence, handoff and resource semantics. That evidence authorized v4; it did not authorize either rejected beam method.

## Frozen artefacts and result

- Preregistration: `docs/15_a2_paper_v4_preregistration.md`.
- Config: `configs/allocation/benchmark_v4.json`.
- Manifest: `data/manifests/allocation/a2_paper_manifest_v4.json`.
- Manifest SHA-256: `0c98f30e92697ce8b5eca724df0f7d1b7053293df1e792707487ecb6c71b5398`.
- Compact result: `reports/phase1_allocation/a2_paper_v4_results.md` and associated JSON/CSV.
- Corpus: 408 instances, 216 independent groups, 18 cells; 402 constructive-witness ordinary cases and six designed-infeasible controls.
- Evaluation: 3,264 registered baseline runs; all nine acceptance checks passed.

Candidate coverage was 99.5% on train and 100% on validation. Frozen cells were IID-small 100%, IID-medium 100%, dense precedence 83.3%, resource bottleneck 91.7%, OOD scale 50% and tight windows 50%. All negative controls were detected. The exact 50% values are not evidence of robustness; they merely meet the preregistered engineering gate and expose substantial room for A3/A4 improvement.

## Failure conclusions retained

There were 692 `schedule_infeasible` ordinary method runs and 48 expected `infeasible` negative-control runs. Because every ordinary instance has an audit-only verified witness, each ordinary failure is a failure of that candidate assignment/sequencing pipeline under its budget. It is not evidence that the instance lacks a solution. Assignment-beam is not a v4 method and cannot be selected retrospectively.

## What A2 closure permits

A3 may now fit graph models on v4 train and select architecture/checkpoints on v4 validation. Before any access to v4 frozen-test/stress, A3 must preregister model families, features, losses, training seeds/budgets, baselines, repair policy and acceptance/statistical protocol. Frozen witness plans remain audit evidence, not model labels unless a future protocol explicitly versions a train-only teacher-label export.

A2 closure does not establish GNN benefit, solver optimality, continuous-motion feasibility, collision safety, real robot execution, factory deployment or physical process quality. All evidence remains `SIM_GEOMETRIC`.
