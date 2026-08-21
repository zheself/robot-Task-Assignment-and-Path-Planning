# A2 paper-scale v3 benchmark card

Evidence label: `SIM_GEOMETRIC`  
Outcome: `FROZEN_FAILED`  
Manifest SHA-256: `a039238d50ac10a6ecc44d6001f07aa75cbc3169c9ac5c8c1800a1794742ae12`

## Composition and independence

V3 contains 408 programmatic instances from 216 independent `task_group_id`s
and 18 registered difficulty cells: train 192/96 groups, validation 48/24,
frozen test 144/72 and stress 24/24. It uses master seed 260821 and v3-prefixed
group identifiers. No v2 task-group or instance identifier is reused.

Variants within a group share workpiece geometry, layout and parent curves and
remain in one split. Statistical inference clusters at task-group level.

## Policies and use

402 ordinary instances satisfy only atomic assignment-unit edge-mask coverage;
this does not establish joint schedule feasibility. Six stress instances are
designed edge-infeasible negative controls. Train is fit/development-only,
validation is selection-only, and frozen/stress became evaluation-only at
manifest creation. Frozen/stress are now observed and diagnostic-only.

## Recorded quality

All schema, hash and leakage audits passed. Eight methods produced 3,264 runs.
The gate passed train, validation, IID-small, IID-medium, dense precedence,
resource bottleneck, tight-window and negative-control checks. It failed only
OOD-scale at 45.8% versus the registered 50% minimum.

## Boundaries

These are programmatic continuous curves and proxy timings, not real robot
trajectories, collision certificates, factory schedules or physical-quality
evidence. `schedule_infeasible` is a method result and never a proof of global
infeasibility. No GNN, RL, detailed physical model or motion planner was used.
