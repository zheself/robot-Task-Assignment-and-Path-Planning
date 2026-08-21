# A4b ordinary LNS/ALNS development summary

Evidence: **SIM_GEOMETRIC development-only**.  No frozen/stress split or neural
checkpoint was created.

- A4b-0 evaluator semantics: passed; old A4a remains invalid and untouched.
- Data: 48 train + 24 development instances in 24 + 12 independent groups.
- Train-selected single operator: `relatedness_destroy`.
- Corrected Slurm jobs: smoke `942372`; six-cell array `942373_[0-5]`, all
  `COMPLETED` with exit `0:0` on `sist-cpu-16`.
- 1 s group-level coverage: random 62.5%, round-robin 62.5%, train-selected
  62.5%, ALNS 62.5%.
- 3 s coverage: round-robin 66.7%; the other three 62.5%.
- Search failure taxonomy: 399 precedence and 33 time-window candidate
  failures; no post-fix incomplete initializer.
- Replay: 24 controlled traces and 96 search-generated candidate outcomes
  reproduced under the same repair cap.
- Tests: 34 A4b targeted, 52 including affected A1 solvers, and 169
  non-frozen regression.
- Decision: **HOLD_A4B_LEARNED_DESTROY_TRAINING**.  Infrastructure is reusable,
  but ALNS has not shown a non-degenerate advantage/recovery signal and the
  normalized-anytime target/penalty and fine timing split must be frozen first.

Full audit, boundaries, hashes and next gate:
`docs/30_a4b_evaluator_and_alns_results.md`.
