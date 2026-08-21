# A2 paper-scale v3 failure analysis

## Decision

The preregistered v3 gate **failed**. The result must not be rounded up, rescued
by changing 50% to 45%, or rerun with a larger budget on the observed frozen
groups. A3 remains blocked.

## What improved

Relative to v2 candidate coverage, the revised scheduler family improved every
difficult frozen cell on entirely new groups: dense precedence 37.5% to 75.0%,
resource bottleneck 54.2% to 83.3%, tight windows 12.5% to 54.2%, scale 12.5%
to 45.8%, and IID-medium 87.5% to 95.8%. These cross-version percentages are
descriptive because the instances differ.

On v3, order-aware LNS was the highest-coverage registered method for resource
bottleneck (20/24), tight windows (13/24) and scale (11/24). The hybrid methods
prevented pure minimum-slack dispatch from accepting an avoidable proxy-quality
regression when the fixed-topological schedule was already feasible.

## Remaining failure

Only `minimum_frozen_cell_candidate_coverage` failed. OOD-scale required 12/24
variants but the union of methods verified 11/24. Across all methods, 740 runs
ended `schedule_infeasible`; the original five methods fail together much more
often than the three revised methods.

The current search still separates assignment from timing. Order-aware LNS
changes assignment and reconstructs two deterministic order policies, but does
not branch over multiple partial resource/robot sequences and does not jointly
optimize start times. Increasing iterations only on the observed scale cases
would be post-hoc tuning and is forbidden.

No failure establishes global instance infeasibility. The assignment MILP's
gap is still not a joint scheduling optimality gap.

## Required next A2 work

1. Add a bounded small-instance joint assignment/sequencing reference, clearly
   distinguishing complete, time-limited and heuristic statuses.
2. Add beam/backtracking or ALNS order moves that branch over robot and shared-
   resource sequences instead of choosing only fixed-topological/minimum-slack.
3. Develop and choose budgets on hand counterexamples plus v3 train/validation
   only; report feasibility and proxy-quality trade-offs.
4. Predeclare a scale-development criterion before choosing any larger search
   budget; do not use v3 frozen/stress for that choice.
5. If development evidence supports another gate, generate v4 with a new seed
   and entirely new task groups. Preserve v2/v3 as observed diagnostics.
