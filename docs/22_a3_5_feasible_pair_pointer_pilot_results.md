# A3.5 Feasible-Pair Pointer pilot closure

Date: 2026-08-11  
Evidence label: **SIM_GEOMETRIC — development only**  
Decision: **CONTINUE_TO_NEW_PREREGISTRATION**  
Frozen/stress generated or accessed: **no**  
V4 instance/witness accessed: **no**

## 1. Question and implementation difference

The closed A3 model encoded a graph once, emitted all segment–robot assignment
logits in parallel, emitted one static order score per segment, aggregated
atomic units only at decode time, and then applied deterministic precedence-
aware sorting. It had no autoregressive representation of current robot load,
completion time, last position, satisfied predecessors, shared resources or the
previous action.

A3.5 retains the graph encoders and the unchanged A1 scheduler/verifier, but
replaces the learned candidate decoder with an **Atomic-Unit–Robot Feasible-
Pair Pointer**. At every step it scores `(atomic_allocation_unit, robot)` pairs
using graph, unit, robot, edge and dynamic-state embeddings. Infeasible pairs,
already selected units and predecessor-blocked units receive `-inf` before the
softmax. Greedy rollout is deterministic. No A4 repair is applied, and failed
raw candidates remain failures.

This is not a classic fixed-class robot classifier and is not described as a
classic Pointer Network. The pilot uses supervised teacher forcing only; it
does not use RL.

## 2. Frozen development protocol

- Protocol: `a3_5_pointer_pilot_v1`
- Configuration SHA-256:
  `876933101dc5ed2e56984d8666dae36509481e93d6d6422ec4eed5380a7bea17`
- Generator seed: `351907`; identifier prefix: `a35p1`
- Model seeds: `101`, `211`, `307`
- Data: 96 train instances in 72 groups; 48 validation instances in 24 groups
- Cells: `iid_small`, `iid_medium`, `dense_precedence`,
  `resource_bottleneck`, `tight_windows`, `scale`
- Registered overall one-group margin: `1/24 = 0.041667`
- Registered maximum per-cell regression: `1/4 = 0.25`
- Teacher budget: assignment MILP 1 s followed by order-aware LNS 50 iterations
- Baseline budget: assignment MILP 3 s and order-aware LNS 100 iterations
- Neural budget: hidden size 64, two layers, up to 30 epochs, patience 6

The manifest internal hash is
`362d3146fa4befa39430806811f035825e660602dd8e8bad5bb3e0abe06ed94e`;
the manifest file SHA-256 is
`3a5060e4ac089835c216d44d95fc61768a53b9b174148c3d3aa0275ec1d4b3e1`.
The loader access hash is
`dc9ca758bf09480805a078fd46fac2081fdf8aef95cfe2e28a4540b6711c87bb`.

All IDs are new and have zero task-group or instance overlap with v2/v3/v4.
The loader rejects frozen-test, stress, and v4 instance/witness paths. No such
split was generated.

## 3. Teacher integrity

The teacher source was order-aware LNS for 79 instances, hybrid assignment MILP
for 11, and a pointer-compatible constructive fallback for 54. “Teacher” means
a verified solver/LNS incumbent, not a globally optimal plan or real expert
action. The fallback was needed because an arbitrary interleaved segment plan
cannot always be represented by one atomic-unit action; it constructs a
canonical unit-block schedule and then reconstructs time windows around that
verified schedule without changing geometry, capabilities, precedence or
resource semantics.

Teacher plans were canonically serialized by planned start time, robot ID and
unit ID. Across 4,474 replayed actions there were zero invalid hard-mask
prefixes. Tests verify deterministic serialization and exact recovery of the
assignment and robot-local unit order.

## 4. Validation result

| Variant | Coverage mean ± std | Three seed coverages | Pair accuracy | Median inference |
|---|---:|---|---:|---:|
| `edge_mlp_static` | 0.493 ± 0.010 | 0.479 / 0.500 / 0.500 | n/a | 0.0154 s |
| `hetero_gnn_static` | 0.507 ± 0.039 | 0.479 / 0.563 / 0.479 | n/a | 0.0175 s |
| `graph_transformer_static` | 0.479 ± 0.017 | 0.479 / 0.458 / 0.500 | n/a | 0.0186 s |
| `hetero_gnn_pair_pointer` | **0.785 ± 0.010** | **0.792 / 0.792 / 0.771** | 0.218 | 0.4002 s |
| `graph_transformer_pair_pointer` | 0.715 ± 0.039 | 0.688 / 0.688 / 0.771 | 0.210 | 0.4056 s |

The preregistered selected pointer is `hetero_gnn_pair_pointer`, compared with
`hetero_gnn_static`. Its mean coverage difference is +0.2778. The seed-wise
differences are +0.3125, +0.2292 and +0.2917. Cell-wise differences are:

| Cell | Pointer minus matched static coverage |
|---|---:|
| dense precedence | +0.5417 |
| iid medium | +0.3333 |
| iid small | +0.0000 |
| resource bottleneck | +0.2500 |
| scale | +0.4167 |
| tight windows | +0.1250 |

All pointer rollouts completed. There were zero hard-mask, atomic-unit and
decoder-dead-end failures. The selected pointer had 31 failed validation
candidates across three seeds; all were retained and classified by the final
scheduler/verifier as `schedule_infeasible`.

## 5. Strong baselines and runtime

| Method | Coverage | Conditional proxy score | Median runtime |
|---|---:|---:|---:|
| `hybrid_assignment_milp` | 0.792 | 107.8312 | 0.8339 s |
| `order_aware_lns` | **0.813** | 91.2818 | 0.4860 s |
| `hybrid_load_balanced` | 0.771 | 112.8866 | 0.0178 s |
| selected Pair-Pointer | 0.785 | 90.5247 | 0.4002 s |

The pointer is about 23 times slower than its matched static decoder. It does
not exceed order-aware LNS and is slightly below assignment MILP in coverage.
Conditional proxy score excludes failures and cannot replace coverage.

## 6. Registered decision

All integrity and continuation checks passed: all 15 shards were complete and
provenance-consistent; all metrics were finite; train/validation were the only
accessed splits; all three pointer seeds beat their matched static seeds; the
mean gain exceeded one independent group; at least two constrained cells
improved; no cell regressed beyond one group; and no repair was used.

The exact permitted conclusion is:

> development-only evidence supports preregistering a new untouched final
> protocol; no frozen benchmark is generated

This pilot does not reopen A3 v4, does not establish GNN/Pointer superiority,
and does not unblock A4 automatically. A new final benchmark may be generated
only after explicit authorisation and a new preregistration.

## 7. Verification and artefacts

- Targeted A3.5 suite: 10 passed.
- Non-v4 regression suite: 133 passed, one warning.
- Thirteen legacy A3 tests requiring the observed v4 development export were
  intentionally excluded to preserve the prohibition on v4 access. The prior
  sealed A3 regression evidence remains immutable; this is reported as a
  boundary, not rewritten as a fresh full-v4 regression.
- Successful Slurm jobs: data generation `940930`, 15-shard array `940937`,
  aggregation `940967`.
- Two pre-manifest failed attempts (`940928`, `940929`) are retained in the job
  record. Neither produced a manifest or evaluation result; partial data from
  the second was discarded before the protocol was regenerated.

Compact results are in
`reports/phase1_allocation/a3_5_pointer_pilot_v1_results.md` and the associated
JSON/CSV tables. Checkpoints, manifests, raw shards, failure libraries and the
data-access record remain under ignored
`outputs/phase1_allocation/a3_5_pointer_pilot_v1/`.

## 8. Evidence boundaries

All new instances and results are `SIM_GEOMETRIC`. They are not real robot
trajectories, factory process data, collision-safety evidence, sim-to-real
evidence, or physical hemming-quality evidence. The pilot implements no
physical model, complex multi-arm collision planning, RL or A4 repair.
