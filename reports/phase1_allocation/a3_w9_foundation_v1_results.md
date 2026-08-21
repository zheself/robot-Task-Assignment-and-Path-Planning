# A3 W9 graph/model foundation results

Evidence: **SIM_GEOMETRIC**. This is a train/validation-only engineering smoke run, not a frozen-test result.

- Gate: **PASSED**.
- Accessed records: train 192, validation 48; forbidden split access: zero.
- Smoke subset: train 16, validation 8 instances, selected by lexicographic cell-balanced groups.
- Vocabulary SHA-256: `a031d82d29b1d466a8bd4ecd3701a24c2fa89c960859c6798e318ef41ff58aa1`; normalizer SHA-256: `c1e93d70d6f9a4ffd93f42c638455f2bb12a40e9615fb1125c3d4f83714abb66`.
- Same-seed checkpoint SHA-256: `e041485c658bda487bd41ade35a3c469692905f44ef946e3f944a2fa831708ed`.

## Engineering checks

- PASS — `record_counts_192_train_48_validation`
- PASS — `zero_train_validation_group_leakage`
- PASS — `zero_forbidden_split_access`
- PASS — `zero_teacher_hash_or_verifier_failures`
- PASS — `normalizer_fit_count_equals_train_graph_count`
- PASS — `deterministic_same_seed`
- PASS — `smoke_subset_is_cell_balanced`

## Smoke metrics

- Train atomic-unit assignment accuracy: 0.619; verified candidate coverage: 0.875.
- Validation atomic-unit assignment accuracy: 0.602; verified candidate coverage: 0.875.
- Validation mean imitation loss: 0.850986.

The four-epoch smoke metrics are not used as a paper claim or as a frozen-test selection result. Failures are retained in the ignored raw output.

## Boundaries

The constructive witness is a feasible A1-proxy teacher, not an optimal or real expert. Hard edge masking and A1 verification do not establish joint motion feasibility, collision safety, robot execution or physical process quality. No A4 repair, path planning or RL was used.
