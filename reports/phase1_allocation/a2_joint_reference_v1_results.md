# A2 bounded joint reference results

Evidence: `SIM_GEOMETRIC`; benchmark access: v3 `train` only.
Acceptance: **PASSED**.

| source | case | segments | robots | status | verified | runtime s |
|---|---|---:|---:|---|---|---:|
| fixture | 01_valid_minimal.json | 1 | 1 | optimal | True | 0.000439 |
| fixture | 02_valid_same_robot_segments.json | 2 | 1 | optimal | True | 0.000494 |
| fixture | 03_valid_explicit_boundary.json | 2 | 1 | optimal | True | 0.000445 |
| fixture | 04_valid_shared_zone.json | 1 | 1 | optimal | True | 0.000308 |
| fixture | 05_valid_priority_window.json | 1 | 1 | optimal | True | 0.000273 |
| v3_train | v3-train-small_sparse-group-004-v00 | 8 | 2 | optimal | True | 0.581297 |
| v3_train | v3-train-small_sparse-group-004-v01 | 8 | 2 | optimal | True | 2.650534 |
| v3_train | v3-train-small_sparse-group-010-v00 | 8 | 2 | feasible_limit | True | 10.002319 |
| v3_train | v3-train-small_sparse-group-010-v01 | 8 | 2 | optimal | True | 3.491370 |
| v3_train | v3-train-small_sparse-group-021-v00 | 8 | 2 | optimal | True | 4.309715 |
| v3_train | v3-train-small_sparse-group-021-v01 | 8 | 2 | feasible_limit | True | 10.003134 |
| v3_train | v3-train-small_sparse-group-008-v00 | 9 | 3 | feasible_limit | True | 10.007081 |
| v3_train | v3-train-small_sparse-group-008-v01 | 9 | 3 | feasible_limit | True | 10.016193 |

`optimal` means complete enumeration only inside the A1 assignment/timing proxy. It is not motion-planning, collision, process-physics or factory optimality.
