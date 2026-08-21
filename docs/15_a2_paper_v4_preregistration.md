# A2 paper-scale v4 preregistration

Status: **FROZEN BEFORE GENERATION OR EVALUATION** (2026-08-10).

## Why a new benchmark is necessary

The v2 and v3 ordinary instances guaranteed only edge-mask coverage for each atomic assignment unit. That condition was incorrectly too weak for evaluating joint assignment and sequencing: it did not prove that precedence, resource and time-window constraints admitted any schedule. Their frozen/stress results remain immutable diagnostic evidence, but neither benchmark may be silently repaired or reused.

The v3 train/validation-only readiness study (`a2_witness_readiness_v1`) constructed and independently verified witnesses for all 240 accessed instances with deterministic output and preservation of non-window semantics. This is sufficient evidence for a benchmark-integrity correction. It is not evidence that the new assignment-beam method is superior: that method failed its separate validation no-regression gate and is excluded from v4.

## Frozen design

- Configuration: `configs/allocation/benchmark_v4.json`.
- New master seed: `260901`; bootstrap seed: `260902`.
- Independent groups: 216; instances: 408; ordinary constructive-witness instances: 402; designed-infeasible negative controls: 6.
- Splits: train 192, validation 48, frozen test 144, stress 24.
- Difficulty cells, robot/segment ranges, group counts, variants, objectives, eight baselines and time budgets are unchanged from v3.
- All task-group and instance identifiers use a v4 prefix and must be disjoint from v2/v3.

For every ordinary instance, the generator first preserves sampled geometry, capabilities, tools, precedence, handoff and resource requirements, constructs a deterministic A1-proxy schedule under the registered 1200 s horizon, and reconstructs time windows around that schedule. Tight-window classification is inherited from the sampled instance. Registered margins are 0.25 process durations before and 0.75 after for tight windows; loose windows receive a 30 s pre-margin and end at the horizon.

The instance and witness plan receive independent hashes in the manifest. Generation must fail if construction or verification fails. Candidate solvers receive only the calibrated instance; the witness file is audit evidence and is not a solver input.

## Frozen methods and statistics

The registered methods are greedy, load-balanced greedy, Hungarian plus sequencing, assignment MILP plus sequencing, deterministic LNS, and the three v3-selected hybrid/order-aware methods. Assignment-beam is not included. The independent statistical unit is `task_group_id`; variants are averaged inside groups. Frozen-test comparisons use cluster bootstrap confidence intervals, paired jointly verified samples, Wilcoxon tests where applicable, and Holm correction. Stress remains descriptive.

## Acceptance gate

The gate requires zero schema, leakage, instance-hash, witness-hash/verification and unexpected-status failures; train/validation candidate coverage at least 0.90/0.80; every frozen cell at least 0.50; and 100% detection of designed-infeasible controls. These thresholds and the complete manifest are frozen before looking at any v4 result.

## Evidence boundaries

All v4 evidence is **SIM_GEOMETRIC**. A constructive A1 proxy witness establishes feasibility only under the implemented schema, geometric proxy oracle and deterministic schedule verifier. It is not proof of continuous robot motion, collision safety, controller execution, physical process quality or real-production validity.
