# UR5 legacy measurements — draft data card

Status: `REAL_STATIC_UNVERIFIED_METADATA`; not approved for final model training.

## Known from files

- Six joint columns are named either `θ1...θ6` or `a1...a6`.
- Nominal position columns are `x,y,z`; measured position columns are normally `x-real,y-real,z-real`.
- Some dated folders include `Time`, `Sequence` and measured orientation, but these fields do not by themselves prove controller transitions or actions.
- `data_all.csv` is excluded because it is a merged-file duplicate risk.
- `data08.csv` is excluded because its final header is `z-real,y-real,x-real` and the positional interpretation conflicts with the declared labels.

## Runtime assumptions used only for preliminary audit

- joint unit: degree;
- position unit: millimetre;
- each source file is a separate group/session;
- an eight-digit parent directory is used as a date ID; root-level dates remain `unverified_date`.

## Must be confirmed before freezing a real manifest

- measurement device and accuracy;
- base/world/tool transforms and controller pose convention;
- whether root `data01...data08` are sessions, workspace partitions or components of one campaign;
- relationship between `data_all.csv` and component files;
- semantics of 建模/验证/10Pos files and whether measurements are independent points or executed paths;
- whether 2025-07/08 sessions share robot calibration, TCP, load and measurement frame;
- DH/controller convention agreement and real joint limits.

## Permitted current use

Column/schema auditing and static error-distribution diagnostics only. Do not use as offline RL transitions, and do not label row adjacency as a real trajectory until metadata confirm it.

## 2026-08-04 provisional findings

- `20250806/10.csv` and `20250807/10Pos.csv` contain nine threshold-matched static repeat points. Their paired error-vector change norm is mean 0.084 mm and P95 0.161 mm. This is a reserved `REAL_STATIC` case study.
- Most included files have median UR5-DH FK to nominal-TCP differences of about 0.8–1.4 mm.
- `20250714/建模数据.csv` differs by about 49.7 mm and is reserved as a frame/TCP diagnostic; it is not used to calibrate the current continuous simulator.
- `data/manifests/ur5_candidate_split_v1.json` is a candidate only and must not be called a final split.
