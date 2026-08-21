# A1 foundation results

Evidence: **SIM_GEOMETRIC**. This is a deterministic engineering smoke matrix, not the A2 paper benchmark.

- Runs: 12; feasible/optimal plans: 8; unexpected failures: 0.
- The deliberately capability-infeasible scenario must return `infeasible` for every method.
- MILP numbers concern assignment proxy load; they do not establish joint schedule/path optimality.

| scenario | method | status | verified | makespan (s) | load variance (s²) | runtime (s) | MIP gap |
|---|---|---:|---:|---:|---:|---:|---:|
| balanced_4_segments_2_robots | greedy-cost-v1 | feasible | True | 11.0 | 4.0 | 0.000853 | None |
| balanced_4_segments_2_robots | load-balanced-greedy-v1 | feasible | True | 11.0 | 4.0 | 0.000495 | None |
| balanced_4_segments_2_robots | hungarian-order-v1 | feasible | True | 11.0 | 4.0 | 0.000544 | None |
| balanced_4_segments_2_robots | assignment-milp-v1 | optimal | True | 11.0 | 4.0 | 0.011558 | 0.0 |
| capacity1_resource_2_segments | greedy-cost-v1 | feasible | True | 8.0 | 0.0 | 0.000343 | None |
| capacity1_resource_2_segments | load-balanced-greedy-v1 | feasible | True | 8.0 | 0.0 | 0.000322 | None |
| capacity1_resource_2_segments | hungarian-order-v1 | feasible | True | 8.0 | 0.0 | 0.000354 | None |
| capacity1_resource_2_segments | assignment-milp-v1 | optimal | True | 8.0 | 0.0 | 0.009540 | 0.0 |
| infeasible_capability_mask | greedy-cost-v1 | infeasible | False | None | None | 0.000252 | None |
| infeasible_capability_mask | load-balanced-greedy-v1 | infeasible | False | None | None | 0.000249 | None |
| infeasible_capability_mask | hungarian-order-v1 | infeasible | False | None | None | 0.000240 | None |
| infeasible_capability_mask | assignment-milp-v1 | infeasible | False | None | None | 0.000249 | None |

## Evidence boundaries

- Synthetic fixture-derived curves; no real production trajectory evidence.
- Analytical reach/no-go proxies; no IK or continuous collision certificate.
- Assignment MILP followed by deterministic proxy scheduling; no joint motion-planning optimality claim.

## Reproduction

`PYTHONPATH=src .venv/bin/python scripts/run_a1_foundation.py`
