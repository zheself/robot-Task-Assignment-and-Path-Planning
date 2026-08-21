# A4b v2 calibration metadata-recovery amendment

Date: 2026-08-19  
Status: **FROZEN AFTER JOB 984111 FAILURE AND BEFORE RECOVERY IMPLEMENTATION**  
Evidence: **SIM_GEOMETRIC development-only**  
Action: **HOLD_A4B_LEARNED_DESTROY_TRAINING**

## Trigger

Packed calibration job `984111` ran all six pinned cell workers and produced
six complete JSONL trace shards. Each worker then failed at the same final
metadata call because the NumPy version attribute in `_record()` was misspelled.
No cell JSON metadata file was written, `984112` became
`DependencyNeverSatisfied`, and labels, smoke, development and aggregation did
not run.

The scientific trace matrix is complete but is not accepted by the existing
merge gate: each cell has exactly 40 exact-30 and 12 fixed-time traces, all 312
identities are unique, all internal trace hashes pass, and all 72 one-second
calibration deadlines retain the exact monotonic cutoff. This amendment does
not declare calibration passed; it authorizes only an auditable reconstruction
of the missing provenance envelope.

## Frozen recovery boundary

Recovery may read only the six trace JSONL files and six stderr logs whose
paths and SHA-256 values are frozen in
`configs/allocation/a4b_ordinary_lns_dev_v2_metadata_recovery_amendment.json`.
It must revalidate the corpus manifest, every trace content hash, cell matrix,
exact-30 completion, fixed-time cutoff, identity uniqueness, CPU mapping and
the frozen global transition signature before writing metadata.

Recovery must not call initializer, destroy, repair, scheduler, verifier or any
search runner. It must not rewrite a JSONL byte. Calibration rows are derived
deterministically from the preserved trace snapshots and the unchanged train
manifest. Original execution source hashes and recovery-tool source hashes are
stored separately. Fields that were not persisted by the failed worker are
marked unavailable; they may not be guessed or presented as native records.

Any missing/changed artifact, unexpected existing metadata, count mismatch,
hash mismatch, cutoff drift, duplicate identity, CPU/log mismatch or source
provenance mismatch fails closed. Only a complete six-cell recovery may enter
the unchanged merge, train gate and label gate.

## Continuation

After implementation and non-frozen tests pass, the never-satisfied jobs
`984112`--`984115` may be cancelled at elapsed zero. A unique replacement chain
starts with metadata recovery plus merge/train-gate/labels, followed by smoke,
six-cell packed development and replay/aggregation through `afterok`. Search
budgets, the 1,800-second watchdog, repair cap 256, methods, seeds, acceptance,
metrics, denominators and development data are unchanged.

This operational recovery does not authorize Neural LNS training and does not
change any A3, A3.5, A4a or A4b v1 conclusion.
