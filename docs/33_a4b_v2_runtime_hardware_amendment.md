# A4b v2 runtime hardware amendment

Date: 2026-08-18  
Status: **FROZEN BEFORE FIRST SEARCH EXECUTION**  
Protocol: `a4b_ordinary_lns_dev_v2`

## Reason

The original CPU-only job chain remained pending for more than six hours
because all 17 usable nodes in the `cpu` partition were allocated and the
remaining node was drained. Jobs `978039`, `978084`, `978085_[0-5]` and
`978086` never received a node, produced no logs or search outputs, and were
cancelled with elapsed time zero.

An initial amended submission to `hexm_l40` exposed a scheduler constraint not
reported by `--test-only`: QOS `partition_hexm` requires a minimum GPU GRES.
Jobs `980995` and `980996` were cancelled with elapsed time zero and no output.
Reserving an unused L40 neither accelerated the predicted start nor respected
the CPU-only intent. The project account can instead use the `normal` GPU-node
partition without a GPU GRES when submitted under `account=v-chengwy`.
An intermediate `normal/account=hexm` chain (`981050`--`981053`) inherited the
same `QOSMinGRES`, never ran, and was cancelled at elapsed zero. This amendment
changes runtime placement only;
it does not alter
data, IDs, split policy, search methods, seeds, operators, repair, acceptance,
budgets, metrics, gates or verifier semantics.

## Frozen amended hardware

- partition: `normal`;
- account: `v-chengwy`;
- fixed node: `sist_gpu59`, preventing heterogeneous CPU hardware across arms;
- one CPU and 8 GiB per task;
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`;
- no `--gres`, no GPU request, and `CUDA_VISIBLE_DEVICES=""`;
- train, smoke, every development cell, replay and aggregation use this same
  environment.

The machine contains GPUs, but this protocol allocates and uses only CPU and
memory. Results must be described as **CPU execution on the sist_gpu59 GPU
node**, not GPU acceleration.

## Integrity boundary

The base config SHA-256 remains
`90b20a1335d48bac28252615284509d5993e9b21811f80175a5c53fe703c7cbc`,
matching the already generated corpus. The separate machine-readable amendment
is `configs/allocation/a4b_ordinary_lns_dev_v2_runtime_amendment.json` and its
hash is recorded by every new runner record. No old job result is pooled with
the amended execution because no old job ran.
