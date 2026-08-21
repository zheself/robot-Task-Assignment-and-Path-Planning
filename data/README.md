# Data policy

Raw measurements remain outside this repository and are mounted/read by path.
Never edit them in place or commit them to Git.

Before any model training, create:

1. a data card in `cards/`;
2. a file inventory with checksums;
3. a split manifest in `manifests/` grouped by robot/date/session/path;
4. a processed dataset whose unit/frame conversions are recorded.

Known unresolved issues in the legacy source:

- `data_all.csv` is commonly a merge of `data01...`; do not include both.
- `UR5_DATA/data08.csv` declares real columns as `z-real,y-real,x-real`; verify
  whether only the header is permuted or the values are permuted.
- root `train00.csv/test00.csv` have unusual error magnitudes and unknown robot,
  frame, and unit semantics.
- MPI/TXT/CAM files may contain paired theoretical, before-compensation, and
  after-compensation information; their correspondence needs confirmation.

