# Execution status

Last updated: 2026-08-20  
Current priority: first-paper evidence/writing plus a decision on stronger A4b ordinary-search recovery for a later Pair-Pointer-derived guided-LNS question; A4a stopped and Neural LNS training held.

| Work package / gate | Status | Evidence / next action |
|---|---|---|
| WP-A A0 schema and constraints | passed and frozen v1 | versioned instance/plan contracts, 11 hard constraints, 4 objectives, 12 auditable fixtures and 19 A0 tests; frozen manifest recorded |
| WP-A A1 solver/verifier | passed, foundation v1 | analytical edge oracle/mask; deterministic verifier and list scheduler; greedy, load-balanced greedy, Hungarian+ordering and assignment MILP; 28 A1 tests and 12-run smoke matrix |
| WP-A A2 geometric benchmark | **passed and frozen v4** | 408 instances/216 independent groups; 402 ordinary instances carry verified constructive A1-proxy witnesses; all preregistered integrity, coverage and negative-control checks passed |
| WP-A A3 learned allocation | **closed: `A3_FINAL_FAILED_BASELINE_FLOOR`** | the sealed evaluator ran exactly once; all integrity gates passed, but the fixed three-seed `edge_mlp` failed absolute/weak-baseline coverage floors; v4 is observed and prohibited for retuning or repeat evaluation |
| WP-A A3.5 Pair-Pointer | **final closed: `A3_5_DECODER_HYPOTHESIS_SUPPORTED`** | unique untouched run: 65.05% versus matched static 40.28%, +24.77 points, 95% CI [+18.52,+31.48], p≈1e-5; below all three strong baselines and not non-inferior |
| WP-A A4a warm-start repair pilot | **closed: `A4A_PRIMARY_EVALUATION_INVALID_STOP`** | 5,616-row matrix completed, but MILP adapter and anytime-cutoff semantic defects invalidate the primary; no rerun/frozen benchmark |
| WP-A A4b evaluator + ordinary LNS/ALNS | **v2 complete: `A4B_V2_DEVELOPMENT_COMPLETE`; `HOLD_A4B_LEARNED_DESTROY_TRAINING`** | recovery/train/labels `984885`, smoke `984886`, six-worker development `984887` and replay/aggregation `984888` passed; 768 traces, 2,304 rows, 384/384 exact-30 and 12,484 replayed steps; all methods tied at 72.92% fixed-time coverage, while exact-30 ALNS recovered only three more identities than random and required 21.99--155.53 s; repair-dominated fixed-time search is not a learned-selector foundation |
| WP-A A4 dynamic replanning | not started; out of A4b scope | no dynamic insertion, RL, path or physical model was introduced |
| WP-B multi-arm path planning | not started | only future interfaces are planned |
| WP-C UR5 kinematics/safety | reusable, metadata partly unverified | FK/Jacobian and projection tests exist; real frame/TCP semantics remain unverified |
| WP-C static prior/simulator | reusable preliminary basis | 14 files/1340 rows audited; candidate split is not frozen |
| WP-C sequence SAC/TD3 | protocol completed, performance failed | three seeds and separated validation/frozen test; neither beats projected base on primary unseen-prior test |

## Reusable assets

UR5 FK/Jacobian, safety projection, smooth joint-trajectory generator, OOD
metrics, data-card/manifest workflow, static-error priors, calibrated simulator,
and separated training/validation/frozen-test reporting may later support WP-B/C.
They do not establish multi-robot allocation, collision safety, real path
execution, physical process quality or RL superiority.

## Immediate next action

Use `docs/27_a4_warm_start_pilot_results.md` as the immutable A4a closure and
`docs/30_a4b_evaluator_and_alns_results.md` as the preserved v1 A4b result and
erratum, `docs/32_a4b_ordinary_lns_recovery_results.md` as the v2 execution
record, and `docs/37_a4b_v2_development_closure.md` as the v2 result closure.
Do not patch or rerun A4a validation. Keep the A3.5 matched-decoder result as
the learned contribution and disclose strong-baseline dominance. Do not train
an A4b learned destroy model from v2. The corrected future question is not a
generic GNN destroy network: it is Pair-Pointer-derived guidance that reuses or
explicitly inherits A3.5 atomic-unit/robot-compatibility representations while
changing only destroy-set selection. If A4b continues, explicitly authorise
and freeze a new-ID ordinary-search recovery protocol that first makes repair
and fixed-time search opportunity non-degenerate; otherwise stop A4b at the v2
engineering foundation. See `docs/28_a4b_neural_lns_research_plan.md` and the
unapproved `docs/38_a4b_pair_pointer_guided_search_protocol_draft.md`.

Protocol R is not yet ready to execute: its final namespace, seeds, challenge
stratum, semantic-parity implementation hashes, paired multi-metric gates and
expected acceleration still require a detailed freeze and user review. It has
no A3.5 checkpoint dependency or access permission. Any later Protocol P must
separately resolve fixed-hash read-only checkpoint permission; otherwise it is
architecture-derived only.

A complete review-only Protocol-R candidate now exists in `docs/39` with the
full draft JSON config and hash-bound v2 profile evidence. It proposes
`a4b_ordinary_search_recovery_v3`/`a4blnsr3`, 56/24 groups, a deterministic
eight-group train challenge stratum, exact reference/accelerated parity and a
7x/0.80-second global repair target. It is not frozen, its seal hashes remain
null, and no execution is authorized. Implementation/testing was subsequently
authorized: prepared repair/search, guarded data/challenge interfaces,
conjunctive gates and a nine-stage CPU-only Slurm chain now exist, with 35
targeted tests passing. Fixture/in-memory speed measurements are diagnostic
only. No `a4blnsr3` corpus, output or job was created, and the formal HOLD is
unchanged. See `docs/40_a4b_protocol_r_implementation_review.md`.

Close and preserve A3.5. Do not rerun `a35f1`, change a fixed checkpoint,
retune thresholds, add repair or select on its results. The admissible paper
claim is significant improvement over the matched static learned decoder with
explicitly lower coverage than the strong optimisation baselines. The next
project decision is whether this is sufficient for the first-paper learned
contribution or whether future A4 work should study solver warm-start/repair
under a separate protocol. V4 and `a35f1` are both evaluation-only.

The descriptive paper package is now generated: six figure pairs (PNG/PDF),
six compact tables, a claim–evidence–boundary table and an evidence manifest.
Use `docs/25_a3_5_first_paper_evidence_skeleton.md` as the writing entry point.
The Pareto figure explicitly records hybrid load-balanced's dominance; the
failure figure does not invent subcauses beyond the retained
`schedule_infeasible` label.

## Latest A3.5 development-only pilot evidence

- The protocol and configuration were frozen before generation. Config SHA-256:
  `876933101dc5ed2e56984d8666dae36509481e93d6d6422ec4eed5380a7bea17`.
- The independent `SIM_GEOMETRIC` corpus contains 96 train and 48 validation
  instances in 72 and 24 task groups respectively, balanced across six
  difficulty cells. It has zero task-group/instance ID overlap with v2/v3/v4;
  no frozen-test or stress split exists.
- The 5-model-family by 3-seed matrix completed. The selected
  `hetero_gnn_pair_pointer` obtained seed coverages 79.2%, 79.2% and 77.1%
  (mean 78.5%, std 1.0 percentage point), compared with 47.9%, 56.3% and
  47.9% for `hetero_gnn_static` (mean 50.7%).
- All six cells were non-regressing under the preregistered one-group rule;
  dense precedence, resource bottleneck and tight windows improved by 54.2,
  25.0 and 12.5 percentage points. There were zero hard-mask, atomicity and
  decoder-dead-end failures, and all greedy rollouts completed.
- Pair-Pointer median inference was 0.400 s versus 0.0175 s for the matched
  static decoder. Its 78.5% coverage was below `hybrid_assignment_milp` at
  79.2% and `order_aware_lns` at 81.3%; conditional scores do not reverse this.
- The registered outcome is: “development-only evidence supports
  preregistering a new untouched final protocol; no frozen benchmark is
  generated.” This does not alter the failed, immutable A3 v4 result.
- Details: `docs/21_a3_5_feasible_pair_pointer_pilot_preregistration.md`,
  `docs/22_a3_5_feasible_pair_pointer_pilot_results.md`, and
  `reports/phase1_allocation/a3_5_pointer_pilot_v1_results.md`.

## A3.5 sealed-final preregistration

- Protocol: `a3_5_sealed_final_v1`; configuration SHA-256
  `8c4d3cb7cc6e61ee589e98786ab78e77958d09694afb102d7f806eda9c208368`.
- Fixed methods: existing hetero-GNN Pair-Pointer and matched static decoder,
  seeds 101/211/307 with exact checkpoint file/state hashes; no retraining,
  seed selection, beam, repair, adaptation or RL.
- Planned untouched corpus: six cells × 12 groups × two variants = 72 groups
  and 144 instances, new seed/ID namespace and zero overlap with all earlier
  corpora. It has not been generated.
- Primary test: group-paired final verifier coverage difference, one-sided
  sign-flip randomisation test plus 95% task-group cluster-bootstrap interval.
- MILP, order-aware LNS and hybrid load-balanced remain secondary quality–time
  comparisons. Failure to exceed them does not invalidate a supported matched-
  decoder hypothesis and must be reported explicitly.
- Full protocol: `docs/23_a3_5_sealed_final_evaluation_preregistration.md`.

## A3.5 sealed-final closure

- Sequence preserved: 13 targeted tests; 146-test non-v4 regression;
  validation-only preflight `941015`; seal
  `64b670831fb5f863462253cf3b79ec9daf96b45f319734ca06006e8db29bcc62`;
  one generation `941022`; one evaluation `941024`.
- Frozen corpus: 72 independent groups/144 instances, six balanced cells;
  manifest internal SHA-256
  `fe8fa5caf997bca742a794abee6168867de012f624df5558aaaf93a1b5935a6f`.
- Pair-Pointer versus static: 65.05% versus 40.28%; paired difference +24.77
  points, 95% CI [+18.52,+31.48], one-sided p=0.0000099999. All three seed
  differences were positive and no cell regressed.
- Strong baselines: MILP 72.22%, LNS 79.86%, hybrid load-balanced 68.75%.
  Pair-Pointer failed the secondary non-inferiority criterion against all.
- All 1,296 rows and 523 failures were retained; integrity, mask, atomicity and
  witness checks passed. Full closure:
  `docs/24_a3_5_sealed_final_evaluation_closure.md`.

## Latest A3 sealed final evidence

- The evaluator was implemented and tested using fixtures and isolated
  train/validation only. Seventeen targeted tests, a 44-row local development
  smoke, a 44-row Slurm preflight and the 136-test full regression passed before
  sealing.
- The evaluator seal SHA-256 is
  `9a357b85f5c5548af9ea277b403e11ec392d586e70fbf757ce31c3a11ff2f1bf`.
  It locks evaluator, runner, tests and sealed-job source hashes; all locks were
  rechecked before first final access.
- The only sealed invocation was Slurm job `940906`: 144 frozen-test plus 24
  stress instances, 11 methods and 1,848 rows. Predictions were persisted
  before witness access. All source, checkpoint, corpus, schema, manifest,
  finiteness, mask, atomicity, witness and negative-control checks passed.
- Fixed `edge_mlp` coverage was 80/144, 84/144 and 81/144 for seeds 17, 29 and
  43; registered seed-mean coverage was 56.7%. It failed the absolute cell
  floors on dense precedence (40.3%), tight windows (37.5%) and scale (15.3%),
  and failed the weak-baseline relative gate on dense precedence.
- Strong-baseline coverage was 68.8% for `hybrid_load_balanced`, 73.6% for
  `hybrid_assignment_milp` and 79.2% for `order_aware_lns`. All paired learned
  minus strong-baseline coverage differences were negative, with bootstrap
  confidence intervals below zero.
- The immutable registered conclusion is: “The fixed learned model failed the
  absolute or weak-baseline feasibility floor; stop the learned A3 branch
  without retuning on v4.” Full closure evidence is in
  `docs/20_a3_final_evaluation_closure.md` and
  `reports/phase1_allocation/a3_final_evaluation_v1_results.md`.

## Latest A3 final-evaluation preregistration

- `edge_mlp` is fixed with seeds 17/29/43 and exact state/checkpoint hashes; no
  retraining, ensemble, best-seed selection, test-time adaptation or repair is
  allowed.
- The five context/weak and three strong v4 baselines must be rerun under their
  unchanged 3 s MILP and 100-iteration LNS budgets.
- Primary evidence is group-aggregated verified-candidate coverage on all 144
  frozen-test instances. Six absolute cell floors and a one-group-equivalent
  relative weak-baseline margin are frozen.
- Five result classes distinguish invalid execution, baseline-floor failure,
  floor-only success, strong competitiveness and strict strong advantage. No
  class permits a GNN, collision-safety, real-production or physical-quality
  claim.
- Stress is descriptive except that all six designed-edge-infeasible controls
  must be rejected. The config SHA-256 is
  `ce574b6b62c2218f8a2f7b3130646444cc00b60ccc69c6822bdde5d3f48ab756`.
- This section records the protocol state at preregistration time. The one-time
  sealed result is now recorded in the preceding closure section; the
  preregistration itself remains unchanged as an immutable historical record.

## Latest A3 W10 development evidence

- The complete registered 3-family × 3-seed matrix trained on 192 train
  instances and selected checkpoints on 48 validation instances. All nine
  shards share the expected config, v4 manifest, data-access, vocabulary and
  normaliser hashes.
- The isolated loader accessed train/validation only; `frozen_test` and
  `stress` remained unopened. The result is `SIM_GEOMETRIC`, not a frozen or
  real-robot result.
- `edge_mlp` was selected by the registered primary metric: all three seeds
  verified 46/48 candidates (95.8%). Mean coverage was 89.6% for
  `hetero_gnn` and 88.9% for `graph_transformer`.
- The selected learned family exceeded weak baseline coverage (83.3–85.4%) but
  remained below `hybrid_assignment_milp` (97.9%) and
  `hybrid_load_balanced`/`order_aware_lns` (100%). Conditional quality excludes
  failed candidates and cannot reverse this feasibility result.
- All registered completeness, provenance, finite-metric, baseline-completion
  and validation-only selection checks passed. The selected model being an
  edge-MLP ablation is negative evidence against a GNN-superiority claim.
- Compact evidence is recorded in
  `docs/18_a3_w10_development_results.md` and
  `reports/phase1_allocation/a3_w10_development_v1_results.md`.

## Latest A3 W9 foundation evidence

- The development protocol, feature families, three model families, losses,
  seeds, budgets and validation selection order were frozen before training.
- The isolated development workspace contained exactly 192 train and 48
  validation instances plus their witnesses; no frozen-test/stress directory
  was copied or opened. The loader independently rejects both forbidden split
  names.
- Implemented canonical segment/robot/resource graph tensors, six directed
  relation types, train-only categorical vocabulary and train-only numerical
  normalisation. Current dimensions are segment 39, robot 23, resource 7 and
  robot–segment pair 12.
- Implemented edge-MLP, relation-aware heterogeneous GNN and relational graph
  Transformer in pure PyTorch. The hard mask is applied inside model forward
  and loss; the decoder assigns atomic units and produces a precedence-aware
  local order without A4 repair.
- Eleven A3-specific access/leakage/hash/permutation/mask/model/decoder tests
  pass. The registered four-epoch, cell-balanced smoke repeated the same seed
  twice and produced the same checkpoint hash.
- Smoke-only result: 16 train and 8 validation instances; validation atomic-unit
  assignment accuracy 60.2% and verified candidate coverage 7/8 (87.5%). This
  is an engineering sanity check, not a paper result or frozen-test claim.

## Latest A2 paper-v4 evidence and closure

- The benchmark-integrity change was preregistered before generation in
  `docs/15_a2_paper_v4_preregistration.md`: new seed and v4-prefixed groups,
  unchanged 408-instance/216-group/18-cell design and unchanged eight baseline
  methods/budgets.
- A separate readiness run accessed only v3 train/validation: all 240 instances
  produced deterministic, verified witnesses while preserving geometry,
  capabilities, tools, precedence, handoff and shared resources. Only time
  windows were reconstructed around the witness schedule.
- The assignment-beam method was evaluated separately on v3 train/validation,
  failed the registered no-cell-regression gate (precedence validation 10/12
  versus 11/12), and was excluded from v4.
- V4 contains 402 ordinary constructive-witness instances and six designed
  edge-infeasible controls. There were zero schema, leakage, instance-hash,
  witness-hash/verification or unexpected-status failures.
- Eight methods × 408 instances produced 3,264 runs. The gate passed: train
  candidate coverage 99.5%, validation 100%; frozen IID-small/IID-medium 100%,
  dense precedence 83.3%, resource bottleneck 91.7%, scale 50.0% and tight
  windows 50.0%; negative-control detection 100%.
- There are 692 retained `schedule_infeasible` candidate-method runs and 48
  correct negative-control `infeasible` runs. Because ordinary instances have
  hidden verified witnesses, the former are conclusively pipeline failures,
  not global infeasibility.
- A2 is closed at the engineering/benchmark gate. This does not establish
  learning superiority, motion-level collision safety, path feasibility, real
  execution or physical process quality.

## Latest A2 joint/beam development evidence

- Added a bounded joint assignment/sequencing reference with explicit
  `optimal`, `feasible_limit`, `limit`, `infeasible` and `unsupported_scale`
  semantics. Optimality is restricted to complete enumeration of the A1 proxy.
- All five valid fixtures completed and verified. Of eight registered small
  v3-train cases, four completed and four retained verified incumbents at the
  10-second limit; the reference protocol passed.
- Added beam sequence search that branches per-robot and shared-resource order,
  wrapped in 10/25/40% assignment-destroy ALNS moves.
- Development accessed 192 v3-train and 48 v3-validation instances only. No
  v2/v3 frozen-test or stress instance was read.
- Beam-ALNS validation coverage was 47/48 (97.9%), exactly tied with order-aware
  LNS in every cell; median runtime was 1.845 s versus 0.516 s. It improved
  conditional proxy score in three cells but not coverage.
- The registered gate required a strict validation coverage gain in at least
  one cell. That check failed, so v4 was not created and A3 remains blocked.

## Latest A2 paper-v3 evidence

- Preregistered before generation: unchanged 408-instance/216-group/18-cell
  design, new master seed and v3-prefixed groups; no v2 group or instance ID was
  reused.
- Development access before v3 was restricted to v2 train/validation. Pure
  deadline dispatch improved feasibility but degraded conditional quality;
  hybrid fixed/deadline selection and order-aware LNS were therefore frozen.
- Integrity: zero schema, leakage, hash, verifier or unexpected-status failure;
  all 48 method runs on six negative controls correctly returned `infeasible`.
- Full evaluation: eight methods × 408 instances = 3,264 runs; 740 retained
  `schedule_infeasible` and 48 correct negative-control rejections.
- Candidate coverage: train 99.0%, validation 97.9%, IID-small 100%, IID-medium
  95.8%, dense precedence 75.0%, resource bottleneck 83.3%, tight windows 54.2%
  and scale 45.8%.
- The frozen 50% per-cell gate therefore failed only OOD-scale. The strongest
  order-aware LNS verified 11/24 scale variants; this is a method failure, not
  proof that the remaining 13 instances are globally infeasible.
- V3 frozen/stress is observed and cannot select a future method, budget or
  threshold.

## Latest A2 paper-v2 evidence

- Preregistered corpus: 408 instances, 216 independent groups, 18 difficulty
  cells; 402 ordinary instances proxy-admissible and six negative controls not.
- Integrity: zero schema, leakage, hash, verifier or unexpected-status failures.
- Evaluation: 2,040 runs; 1,440 verified plans, 570 `schedule_infeasible`, and
  30 correct negative-control rejections.
- Coverage passed train (94.8%) and validation (93.8%), but failed frozen
  dense-precedence (37.5%), scale (12.5%) and tight-window (12.5%); resource
  bottleneck narrowly passed at 54.2%.
- Full regression: 99 passed with one existing Gymnasium warning.
- V2 frozen is now observed and diagnostic-only; it cannot be reused as a new
  method's unseen final test.

## Latest A2 pilot evidence

- Generator/config tests: fixed seeds reproduce canonical instance bytes;
  line, arc, B-spline and closed-loop curves preserve continuous endpoints.
- Split audit: workpiece, layout, task group and parent-curve IDs have zero
  cross-split leakage; each materialized instance has a verified SHA-256.
- Frozen corpus: 19 instances across 10 task groups, covering 2–8 robots and
  9–74 realized segments in this seed.
- Baselines: 5 methods × 19 instances = 95 runs. There are 23 retained
  `schedule_infeasible` method-instance results; these are failures of the
  current assignment-first candidate plus deterministic scheduler, not proof
  that the full combinatorial problem is infeasible.
- Candidate labels: train 8/8, validation 3/4, frozen test 5/6 and stress 0/1
  have at least one verified candidate. Frozen/stress labels are evaluation-only.
- A2 adds 13 tests; full-project regression is 88 passed with one existing
  Gymnasium deprecation warning.
- Gate decision: pipeline-level A2 acceptance passes, but paper-level A2 remains
  open because 8 train, 4 validation and 6 frozen-test instances are not a
  defensible sample for GNN training or per-difficulty statistical claims.

## Latest A1 evidence

- W3: 10 tests for curve features, reason-coded feasibility masks and proxy boundaries.
- W4–W5: 18 tests for plan verification, handoff, transition time, time windows, shared-resource
  capacity, deterministic heuristics, MILP optimal/infeasible/limit states and
  recorded bound/gap fields.
- Foundation smoke matrix: 3 fixture-derived scenarios × 4 methods = 12 runs;
  8 expected feasible/optimal plans were independently verified, and 4 expected
  capability-mask cases returned `infeasible`; no unexpected failure.
- Full project regression: 75 tests pass (one upstream Gymnasium warning).
- This closes only the A1 engineering gate. It is not the A2 frozen benchmark
  and supplies no paper-level superiority result.

## Visible evidence gaps

No current data contain verified multi-robot layout, continuous process curves,
tool capability, time-window, assignment, collision or execution labels. UR5
static CSVs have no transitions. No physical model, validated full collision
geometry, real factory log or real hemming-quality data is available.
