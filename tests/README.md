# Test plan

Required before RL training:

- UR5 FK against trusted known poses and legacy outputs.
- analytic/numerical Jacobian agreement and finite-difference scale tests.
- degree/mm to radian/metre conversions exactly once.
- split manifests contain no file/session/path overlap.
- merged files and component files cannot both enter a dataset.
- environment reset/step determinism by seed.
- action bounds and local-frame-to-Cartesian conversion.
- safety projection near joint limits and singularities.
- no hidden test-domain statistics in priors or randomization.

The dependency-light initial suite is runnable now:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The expanded suite also covers REAL_STATIC candidate-group leakage, static
matching thresholds, all prior interfaces, training-only support/OOD fitting,
calibration terminology, smooth trajectory generation, history validity,
delayed applied-action history, explicit delay state, and train/test prior-set
separation.
