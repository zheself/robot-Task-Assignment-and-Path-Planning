# UR5 adapter audit summary

Evidence: `REAL_STATIC_UNVERIFIED_METADATA`  
Runtime assumptions: joint angles in degrees; positions in millimetres. These assumptions still require confirmation.

- Accepted: 1340 rows from 14 source files.
- Grouping: one session/path group per source file; row order is not treated as RL transitions.
- Date IDs detected: `20250714`, `20250806`, `20250807`, and `unverified_date`.
- Position-error norm: mean 2.788 mm, median 2.720 mm, P95 4.684 mm, maximum 5.447 mm.
- Excluded `data_all.csv` because of merged/component duplication risk.
- Excluded `data08.csv` because its declared measured-coordinate order is suspect.
- Dataset status: `not_an_offline_rl_dataset`.

No real train/validation/test manifest has been frozen. Coordinate frames, measurement device, TCP/load/calibration consistency and file semantics must be confirmed first. The complete per-file audit is generated under ignored `outputs/ur5_legacy_audit/audit.json`.
