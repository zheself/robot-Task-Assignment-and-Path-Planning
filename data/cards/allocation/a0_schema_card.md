# A0 continuous-allocation schema card

Version: `allocation-instance-v1`  
Constraint dictionary: `allocation-constraints-v1`  
Evidence: schema/fixtures only; no allocation experiment has run.

## Semantics

The atomic task is an ordered continuous process segment, not an independent
point. SI units are mandatory. A parent curve uses unique zero-based contiguous
segment indices; adjacent segments meet geometrically within `1e-6 m`.
`not_splittable` forbids multiple segments for one parent. Predecessors must
exist and form a DAG. Shared resources are conservative scheduling abstractions,
not verified geometric collision models.

Handoff policies:

- `free`: no same-robot requirement at this boundary;
- `same_robot`: A1 allocation must keep the parent segments on one robot;
- `explicit_boundary`: a handoff is permitted only at the declared boundary;
- `not_splittable`: the entire parent curve is one allocation task.

Process direction is `forward`, `reverse`, or `either`. Direction and handoff
are semantic constraints for later assignment/scheduling; A0 validates their
representation but does not solve them.

## Fixtures

There are twelve auditable JSON fixtures: five valid cases and seven invalid
cases covering duplicate IDs, non-contiguous indices, curve discontinuity,
precedence cycles, unknown resources, non-splittable parents and endpoint
mismatch. Derived fixtures use explicit JSON-pointer patch operations against
the minimal base to keep differences reviewable.

## Exclusions

No fixture is real workcell evidence. They are `SYNTHETIC`; the schema does not
claim IK reachability, robot collision safety, process physics, factory cycle
time, real hemming quality or RL transitions.
