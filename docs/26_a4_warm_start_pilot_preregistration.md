# A4a Pair-Pointer warm-start development-pilot preregistration

Status: **FROZEN BEFORE A4a DATA GENERATION**  
Date: 2026-08-11  
Evidence: **SIM_GEOMETRIC — DEVELOPMENT ONLY**

## Confirmatory development question

Under the same repair/LNS implementation, operators, random seeds, iteration
or end-to-end budget, acceptance rule, scheduler and verifier, does an immutable
Pair-Pointer initializer reach verified feasibility or a train-frozen target
quality faster than matched static, hybrid load-balanced, assignment-MILP and
cold initializers?

This does not reopen A3/A3.5, repair their sealed outputs, or test the already
supported raw-decoder hypothesis again. No frozen/stress data are generated.

## Locked data and inference state

Protocol `a4_warm_start_pilot_v1` uses master seed `440317`, prefix `a4wsp1`,
six registered cells, eight train and four validation groups per cell, and two
variants per group: 48/24 independent groups and 96/48 instances. Thus one
overall validation group is `1/24 = 0.041667`; one cell group is `1/4 = 0.25`.
All IDs must be disjoint from v2/v3/v4, A3.5 pilot and A3.5 final.

The six Pair-Pointer/static checkpoints and seed pairing 101/211/307 are
immutable. Their train-fitted vocabulary and normalizer are part of the frozen
inference pipeline. Because earlier checkpoints did not embed those objects,
they are exported once into a compact, hash-locked inference-metadata artifact
from A3.5 **train only**. This export is not training, selection or benchmark
evidence and may not read A3/A3.5 frozen data. All later A4 runs load only this
artifact, checkpoints and new A4 data. This technical dependency is disclosed
rather than silently refitting preprocessing on A4 data.

## Identical repair and budgets

Every initializer is converted to the same atomic-unit state and enters
`identical_atomic_unit_alns_v1`. It uses the same hard-feasible atomic
reassignment, random/worst-load destroy, load/regret reinsert, robot-local
relocate/swap, precedence-safe reorder, acceptance rule, random seeds 401/907,
and unchanged A1 scheduler/verifier. No initializer-specific parameter exists.

Fixed-work views use 10, 50 and 100 iterations; the 50-iteration view is the
co-primary diagnostic. Fixed end-to-end budgets are 0.5, 1.0 and 3.0 seconds;
the 1.0-second view is primary. Initializer runtime is charged before repair.
Time-budget runs also have a common 100-iteration ceiling, so they terminate at
the first of the wall-clock or work ceilings.
An initializer exceeding the total budget is an unsuccessful row, even if it
would later return a plan. Raw initializers and original 100-iteration
order-aware LNS are retained as references.

The time-to-target threshold is the per-cell median verified weighted score of
100-iteration order-aware LNS on A4 train. The rule and resulting values are
sealed before validation. Validation cannot change an operator, target, budget,
checkpoint, seed or gate.

## Metrics, gate and failure wording

All failures remain in coverage denominators. Report fixed-budget/fixed-work
coverage, time-to-first-feasible, time-to-target, final score, makespan, load
variance, runtime decomposition, edit counts, initializer retention, failure
taxonomy, difficulty, scale and all seed results. Statistical unit is
`task_group_id`; variants are clustered.
Time-to-target is aggregated as a restricted mean, censoring failures at the
registered end-to-end budget; successful-only timing cannot establish an
advantage.

Continuation requires zero integrity failures; repair improving raw Pointer;
at least two matched seed wins and mean Pointer-minus-static improvement of at
least 1/24 at the primary budget; and one preregistered practical advantage
over load-balanced+repair (coverage +1/24, coverage non-inferior within 1/24
plus 10% faster target, non-inferior coverage plus 5% objective gain, or a
one-cell-group scale gain without another cell regressing by more than one
group). Mean Pair-Pointer initializer assignment retention must be at least
50%. End-to-end timing must include inference.

Pass wording is only `CONTINUE_TO_NEW_A4_PREREGISTRATION`. Otherwise use
`STOP_A4_LEARNING_WARM_START_BRANCH`, retain Pair-Pointer as the validated raw
decoder ablation, and return the first-paper main line to solver/heuristic
methods. Neither outcome permits a real-robot, collision, physical-quality,
RL, path-planning or sim-to-real claim.

Machine-readable configuration: `configs/allocation/a4_warm_start_pilot_v1.json`.
Its SHA-256 is recorded before generation in the A4 output seal.
