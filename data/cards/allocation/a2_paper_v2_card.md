# A2 paper-scale geometric benchmark v2 card

Version: `a2-paper-geometric-benchmark-v2`  
Manifest: `a2-paper-split-manifest-v2`  
Evidence: `SIM_GEOMETRIC`  
Gate outcome: `FROZEN_FAILED`  
Manifest SHA-256: `c9f532e72db33324e8f84677001cab028eb9a8e5634857998375ae7ed99f0842`

## Frozen design

The design was preregistered before running v2 solvers. It contains 408
instances from 216 independent task groups: train 192/96, validation 48/24,
frozen test 144/72 and stress 24/24 instances/groups. Sibling variants share
geometry/layout and never cross splits.

All 402 ordinary instances passed the preregistered proxy-admissibility check;
all six negative controls failed it by construction. Proxy admissibility is
edge-mask coverage for atomic assignment units, not proof of a feasible joint
schedule.

## Frozen protocol

Every instance ran greedy, load-balanced greedy, Hungarian plus ordering,
assignment MILP and deterministic assignment-LNS. MILP received three seconds
with requested gap zero; LNS used 100 iterations and seed zero. Every candidate
was passed through one deterministic verifier.

The independent statistical unit is task group. Variants are averaged within
group. Frozen comparisons use 5,000 group-cluster bootstrap resamples, paired
jointly verified quality samples, Wilcoxon tests where supported and Holm
correction. Failed plans receive no imputed quality.

## Outcome

- Solver runs: 2,040; verified plans: 1,440.
- Failures: 570 `schedule_infeasible` and 30 expected negative-control
  `infeasible` results.
- Candidate coverage: train 94.8%, validation 93.8%.
- Frozen coverage: IID-small 100%, IID-medium 87.5%, dense-precedence 37.5%,
  resource-bottleneck 54.2%, scale 12.5%, tight-window 12.5%.
- Designed-infeasible detection: 100%.
- Assignment MILP reached SciPy status 0 on 367 instances, status 1 on 35,
  and stopped before SciPy on six edge-infeasible controls.

The preregistered minimum frozen-cell coverage was 50%, so v2 failed A2. No
threshold, instance, seed or budget was changed after inspection.

## Interpretation and future use

The shared assignment-first methods all use essentially the same deterministic
topological ordering and list scheduler. Tight deadlines, dense dependencies
and larger resource-constrained instances frequently fail downstream. This is
evidence of a baseline/scheduler limitation, not global infeasibility and not
evidence for or against a GNN.

V2 frozen data are now observed and diagnostic-only. Future method development
may use v2 train/validation, but final evaluation requires new v3 seeds/groups.
A3 remains blocked while A2 is failed.

## Boundaries

There is no verified IK, continuous collision guarantee, CAD, factory timing,
physical quality, real deployment or sim-to-real evidence. Assignment MIP gap
is not joint scheduling/path optimality.
