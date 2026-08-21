# A3 heterogeneous graph data card v1

Evidence label: **SIM_GEOMETRIC**. Development source: A2 paper benchmark v4.

## Access and provenance

The development loader accepts only `train` and `validation`; it raises `PermissionError` for `frozen_test` and `stress`. Vocabulary, numerical mean and standard deviation are fitted on all 192 train instances only. Validation contributes no fitted preprocessing or gradient. The access-manifest, vocabulary and normalizer hashes are saved with every run.

The assignment/order teacher is the matching `constructive-witness-v1` plan. It is independently hashed and verified under the A1 analytical oracle. It is a deterministic feasible proxy schedule—not a global optimum, real engineering action or factory demonstration.

## Node features

- Segment, 39 dimensions for the current train vocabulary: polyline/chord/tortuosity, centroid, AABB, direction, turning, declared length, process duration, priority, time window, predecessor/resource counts, segment index, process-direction and handoff one-hot values, required-capability and tool multi-hot values.
- Robot, 23 dimensions: base pose, availability, nominal Cartesian speed, joint-state summaries, capability/tool multi-hot values and kinematic-model category.
- Resource, 7 dimensions: resource type, capacity and availability window.

Category dimensions are versioned by the train-only vocabulary. Unknown validation categories map to `<UNK>`; they do not expand the feature space.

## Relations and edge features

Relations are bidirectional robot–segment, directed segment precedence/follow, and bidirectional segment–resource usage. Parent-curve segment order is added to declared precedence. Robot–segment pair features contain the A1 oracle feasible flag, travel/process time, path length, kinematic-risk proxy, conflict proxy, confidence and five reason-code indicators.

The raw feasibility boolean is retained separately as `allowed_mask`. Normalizing an edge feature cannot change this mask. Model logits on invalid edges are set to negative infinity before both cross-entropy and decoding.

## Decoder semantics

Assignments operate on A1 atomic allocation units: same-robot and non-splittable members cannot be separated. A unit may select a robot only when every member edge is allowed. Learned order scores are converted to a deterministic robot-local topological order. The unchanged A1 list scheduler creates timing; no A4 repair is applied. Any scheduling or verifier failure is retained.

## Excluded evidence

The graph does not contain full IK, continuous collision geometry, controller dynamics, contact, stress, plasticity or process quality. A verified graph candidate is not a motion-level safety certificate or real execution result.
