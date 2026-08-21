# A2 joint reference and beam-search development

## Scope

This batch addresses the v3 diagnosis without reading v2/v3 frozen-test or
stress instances. It adds a small bounded reference and a scalable heuristic;
neither is motion planning, collision certification or physical modelling.

## Small joint proxy reference

The reference enumerates every edge-admissible atomic-unit assignment and every
deduplicated precedence-ready robot/resource sequence within declared segment,
assignment, node and wall-time bounds. It distinguishes:

- `optimal`: enumeration completed and the best verified A1-proxy objective is
  known for the bounded case;
- `feasible_limit`: a verified incumbent exists but enumeration did not finish;
- `limit`: budget ended without an incumbent;
- `infeasible`: complete enumeration found none;
- `unsupported_scale`: the registered segment or assignment-combination bound
  was exceeded.

The word optimal never extends to IK, continuous collision, real timing,
physical process quality or a factory schedule.

Protocol results: five valid fixtures were complete/verified. Of eight
deterministically selected v3-train instances (8–9 segments), four completed
within 10 seconds and four returned verified `feasible_limit` incumbents. The
reference protocol passed and now supplies auditable small-case labels/bounds.

## Beam sequencing plus ALNS

For each assignment candidate, beam search branches over precedence-ready next
segments. Those branches induce alternative per-robot sequences and alternative
capacity-one shared-resource occupation orders. An ALNS-style outer loop uses
10%, 25% and 40% assignment destroy sizes. The registered development budget
was 12 iterations, beam width 8 and 30,000 nodes per candidate.

On v3 validation, beam-ALNS verified 47/48 instances, equal to order-aware LNS;
coverage was 100%, 91.7%, 100% and 100% across small, precedence, resource and
balanced cells. Median runtime was 1.845 seconds versus 0.516 seconds for
order-aware LNS. Conditional score improved in three cells and regressed in one.

## Gate decision

The predeclared development gate required at least one validation cell with a
strict coverage gain over order-aware LNS. There were zero, so the gate failed.
Train-only coverage gains do not override that condition. V4 was intentionally
not generated; A3 remains blocked.
