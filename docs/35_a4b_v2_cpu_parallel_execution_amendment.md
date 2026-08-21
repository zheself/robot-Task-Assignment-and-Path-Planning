# A4b v2 CPU-parallel execution amendment

Date: 2026-08-18  
Status: **FROZEN DURING SERIAL CALIBRATION BEFORE PARALLEL SUBMISSION**  
Action: **HOLD_A4B_LEARNED_DESTROY_TRAINING**

## Trigger and scope

Replacement serial job `983899` was inspected while still inside calibration:
it had no complete calibration JSON/JSONL and had not entered train gate or
label generation. Historical trace runtimes project about 13.5 CPU-hours for
the serial fixed-iteration matrix. The slowest `scale` cell projects about 7.3
hours, so six cell workers provide an estimated 1.86x wall-clock speedup while
leaving every trace unchanged.

This amendment changes execution placement only. It does not change data,
methods, ordering within a cell, search or repair budgets, random seeds,
acceptance, scheduler, verifier, metrics, denominators, selection logic or
gates.

## Frozen parallel unit

Train calibration and development use six shards in the preregistered cell
order: `iid_small`, `iid_medium`, `dense_precedence`, `resource_bottleneck`,
`tight_windows`, `scale`. Each Slurm worker owns one cell, one allocated CPU
task and 4 GiB; each Python process is pinned to one CPU from its Slurm cpuset.
OMP and MKL are one thread and CUDA visibility is empty. A trace remains
serial. Within a cell, instances and methods retain deterministic order.

Train and development both use concurrency six on `sist_gpu59`. Fixed-time
methods for a given cell therefore execute sequentially on the same pinned
worker while only independent cells overlap. The worker records its node,
Slurm allocation, affinity, process start/end, concurrency and system load.
An affinity or environment mismatch fails before search.

## Calibration merge gate

Each cell must produce exactly four instance identities, 40 fixed-iteration
traces and 12 fixed-time traces. Merge requires all six shards, 240/240
complete exact-30 traces, 72 fixed-time traces, no duplicate trace identity,
and identical manifest, base config, runtime/watchdog/parallel amendment and
source hashes. Trace content hashes are rechecked. Missing, partial, duplicate
or foreign files fail closed.

Only after validation are traces and rows sorted by frozen cell, instance,
method, budget-mode and budget order. Operator/update-scheme selection and
metric references are recomputed from the complete merged train data. Failed
and infeasible traces remain in the denominator; incomplete exact-iteration
traces cause failure and emit no exact-iteration rows.

The dependency chain is calibration array -> merge/train gate/labels -> smoke
-> development array -> replay/aggregate, entirely through `afterok`.

## Historical preservation

No output from `983899` may be reused. If it remains in incomplete calibration
after implementation and tests, its state, log and any partial files are
hashed and preserved before cancellation together with downstream jobs. There
must never be two live calibration matrices.
