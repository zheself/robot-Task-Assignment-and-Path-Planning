# A4b v2 fixed-iteration watchdog recovery amendment

Date: 2026-08-18  
Status: **FROZEN AFTER FAILED TRAIN GATE AND BEFORE RESTART**  
Evidence: **SIM_GEOMETRIC development-only**

## Observed failure

Train-only job `981071` ran on the registered CPU-only environment and failed
the preregistered train gate before labels or development. Of 240
fixed-iteration calibration traces, 180 terminated as `safety_watchdog` under
the 60-second operational watchdog. Every incomplete trace returned at or
after 60 seconds. The failure affected all methods symmetrically and was
difficulty-dependent: `iid_small` completed, while all dense-precedence,
resource-bottleneck, tight-window and scale traces were truncated.

The six exact-30 rows per method in the failed output are a selected complete
subset and are not performance evidence. The failed output is preserved and
excluded from restarted calibration and selection.

## Root cause and amendment

The 60-second watchdog was incorrectly sized as if it were an experimental
time budget, although the frozen protocol defines it only as a protection
against a stalled exact-iteration run. Runtime extrapolation from the failed
train traces estimates about 13.5 CPU-hours for the complete serial fixed-
iteration calibration, with the slowest observed scale trace projecting to
about 1,115 seconds for 30 iterations.

Before any development access, the watchdog is therefore changed from 60 to
1,800 seconds per fixed-iteration trace. The train and development Slurm wall
times are changed to 24 hours. This does not add neighborhoods or repair
evaluations: every accepted exact-iteration trace still contains exactly 30
completed neighborhoods under the same seed and algorithm. The 0.5, 1.0 and
3.0-second end-to-end deadlines are unchanged and continue to include the
initializer.

## Restart integrity

- reuse only the already sealed `a4blnsd2` train/development corpus;
- rerun the complete train calibration rather than merge successful rows from
  job `981071`;
- preserve the failed output, logs, job state and hashes under a failed-attempt
  subdirectory;
- retain all original train and label gates without relaxation;
- continue to labels, smoke and development only through `afterok`;
- keep `HOLD_A4B_LEARNED_DESTROY_TRAINING` regardless of process progress.

This amendment does not authorize neural training, frozen data generation,
reinsertion learning, RL, or access to old frozen material.
