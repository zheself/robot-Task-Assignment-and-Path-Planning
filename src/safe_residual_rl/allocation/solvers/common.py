"""Shared contracts and deterministic utilities for A1 baselines."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..masks import EdgeMask, build_edge_mask
from ..oracle import OracleContext
from ..schema import AllocationInstance, AllocationPlan, HandoffPolicy, ProcessSegment
from ..scheduling import ScheduleBuildResult, build_deadline_aware_schedule, build_schedule
from ..verifier import verify_plan


@dataclass(frozen=True)
class SolverProtocol:
    version: str
    time_limit_s: float
    relative_gap: float
    deterministic_seed: int


@dataclass(frozen=True)
class SolverResult:
    method_id: str
    status: str
    plan: AllocationPlan | None
    runtime_s: float
    objective_value: float | None
    best_bound: float | None
    mip_gap: float | None
    diagnostics: tuple[str, ...]
    # A scheduling failure must not erase a genuine assignment-solver
    # incumbent.  A4b consumes this field for provenance only; the assignment
    # is still unverified until it passes the unchanged scheduler/verifier.
    assignment_incumbent: tuple[tuple[int, str], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "status": self.status,
            "runtime_s": self.runtime_s,
            "objective_value": self.objective_value,
            "best_bound": self.best_bound,
            "mip_gap": self.mip_gap,
            "diagnostics": list(self.diagnostics),
            "assignment_incumbent": (
                None
                if self.assignment_incumbent is None
                else [[index, robot] for index, robot in self.assignment_incumbent]
            ),
            "plan": None if self.plan is None else self.plan.to_dict(),
        }


def load_solver_protocol(path: str | Path) -> SolverProtocol:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    protocol = SolverProtocol(
        version=str(data["version"]),
        time_limit_s=float(data["time_limit_s"]),
        relative_gap=float(data["relative_gap"]),
        deterministic_seed=int(data["deterministic_seed"]),
    )
    if protocol.version != "a1-solver-protocol-v1":
        raise ValueError("unknown solver protocol")
    if protocol.time_limit_s <= 0 or not 0 <= protocol.relative_gap <= 1:
        raise ValueError("invalid solver limits")
    return protocol


def allocation_units(instance: AllocationInstance) -> tuple[tuple[str, ...], ...]:
    """Return atomic assignment units implied by handoff semantics."""
    groups: dict[str, list[ProcessSegment]] = {}
    for segment in instance.segments:
        groups.setdefault(segment.parent_curve_id, []).append(segment)
    units: list[tuple[str, ...]] = []
    for parent_id in sorted(groups):
        group = sorted(groups[parent_id], key=lambda item: item.segment_index)
        coupled = any(
            item.handoff_policy in {HandoffPolicy.SAME_ROBOT, HandoffPolicy.NOT_SPLITTABLE}
            for item in group
        )
        if coupled:
            units.append(tuple(item.id for item in group))
        else:
            units.extend((item.id,) for item in group)
    rank = {segment_id: index for index, segment_id in enumerate(topological_segment_order(instance))}
    return tuple(sorted(units, key=lambda unit: min(rank[item] for item in unit)))


def topological_segment_order(instance: AllocationInstance) -> tuple[str, ...]:
    """Stable order for declared precedence plus within-curve sequence."""
    successors: dict[str, set[str]] = {item.id: set() for item in instance.segments}
    indegree = {item.id: 0 for item in instance.segments}
    for segment in instance.segments:
        for predecessor in segment.predecessor_ids:
            if segment.id not in successors[predecessor]:
                successors[predecessor].add(segment.id)
                indegree[segment.id] += 1
    groups: dict[str, list[ProcessSegment]] = {}
    for segment in instance.segments:
        groups.setdefault(segment.parent_curve_id, []).append(segment)
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item.segment_index)
        for left, right in zip(ordered, ordered[1:]):
            if right.id not in successors[left.id]:
                successors[left.id].add(right.id)
                indegree[right.id] += 1
    ready = sorted(key for key, value in indegree.items() if value == 0)
    result: list[str] = []
    while ready:
        current = ready.pop(0)
        result.append(current)
        for successor in sorted(successors[current]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(result) != len(instance.segments):
        raise ValueError("combined precedence graph is cyclic")
    return tuple(result)


def unit_robot_costs(
    instance: AllocationInstance, mask: EdgeMask
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[float, ...], ...]]:
    units = allocation_units(instance)
    robots = tuple(item.id for item in instance.robots)
    estimates = {
        (segment_id, robot_id): mask.estimates[mask.segment_ids.index(segment_id)][mask.robot_ids.index(robot_id)]
        for segment_id in mask.segment_ids
        for robot_id in mask.robot_ids
    }
    costs: list[tuple[float, ...]] = []
    for unit in units:
        row = []
        for robot_id in robots:
            selected = [estimates[(segment_id, robot_id)] for segment_id in unit]
            row.append(
                sum(item.process_time_s + item.travel_time_s for item in selected)
                if all(item.feasible for item in selected)
                else float("inf")
            )
        costs.append(tuple(row))
    return units, robots, tuple(costs)


def orders_from_assignment(
    instance: AllocationInstance,
    unit_assignment: Mapping[int, str],
    units: Sequence[Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    assigned = {
        segment_id: unit_assignment[index]
        for index, unit in enumerate(units)
        for segment_id in unit
    }
    order = topological_segment_order(instance)
    return {
        robot.id: tuple(item for item in order if assigned.get(item) == robot.id)
        for robot in instance.robots
    }


def finalize_assignment(
    instance: AllocationInstance,
    context: OracleContext,
    method_id: str,
    unit_assignment: Mapping[int, str],
    units: Sequence[Sequence[str]],
    started_at: float,
    diagnostics: Sequence[str],
    objective_value: float | None = None,
    best_bound: float | None = None,
    mip_gap: float | None = None,
    success_status: str = "feasible",
    scheduling_policy: str = "fixed_topological_v1",
    schedule_weights: Mapping[str, float] | None = None,
    assignment_incumbent: Mapping[int, str] | None = None,
) -> SolverResult:
    assignment = {
        segment_id: unit_assignment[index]
        for index, unit in enumerate(units)
        for segment_id in unit
    }
    if scheduling_policy == "fixed_topological_v1":
        orders = orders_from_assignment(instance, unit_assignment, units)
        built = build_schedule(instance, orders, context, method_id)
    elif scheduling_policy == "deadline_aware_v2":
        built = build_deadline_aware_schedule(instance, assignment, context, method_id)
    elif scheduling_policy == "best_of_fixed_and_deadline_v2":
        orders = orders_from_assignment(instance, unit_assignment, units)
        fixed = build_schedule(instance, orders, context, method_id)
        deadline = build_deadline_aware_schedule(instance, assignment, context, method_id)
        feasible = [item for item in (fixed, deadline) if item.plan is not None]
        if not feasible:
            built = ScheduleBuildResult(
                "infeasible",
                None,
                fixed.diagnostics + deadline.diagnostics + ("HYBRID_NO_FEASIBLE_ORDER",),
            )
        else:
            weights = dict(
                schedule_weights
                or {
                    "makespan": 1.0,
                    "load_variance": 0.05,
                    "travel_setup_time": 0.1,
                    "priority_tardiness": 1.0,
                }
            )
            selected = min(
                feasible,
                key=lambda item: sum(
                    weights.get(key, 0.0) * value
                    for key, value in item.plan.objective_terms
                ),
            )
            built = ScheduleBuildResult(
                selected.status,
                selected.plan,
                selected.diagnostics + ("BEST_OF_FIXED_AND_DEADLINE_V2",),
            )
    else:
        raise ValueError(f"unknown scheduling policy: {scheduling_policy}")
    combined = tuple(diagnostics) + built.diagnostics
    if built.plan is None:
        status = "schedule_infeasible" if built.status == "infeasible" else built.status
        return SolverResult(
            method_id,
            status,
            None,
            time.perf_counter() - started_at,
            objective_value,
            best_bound,
            mip_gap,
            combined,
            None if assignment_incumbent is None else tuple(sorted(assignment_incumbent.items())),
        )
    checked = verify_plan(instance, built.plan, context)
    if not checked.feasible:
        codes = sorted({item.code for item in checked.violations})
        return SolverResult(
            method_id,
            "verification_failed",
            built.plan,
            time.perf_counter() - started_at,
            objective_value,
            best_bound,
            mip_gap,
            combined + tuple(f"VERIFY={item}" for item in codes),
            None if assignment_incumbent is None else tuple(sorted(assignment_incumbent.items())),
        )
    return SolverResult(
        method_id,
        success_status,
        built.plan,
        time.perf_counter() - started_at,
        objective_value,
        best_bound,
        mip_gap,
        combined + checked.diagnostics,
        None if assignment_incumbent is None else tuple(sorted(assignment_incumbent.items())),
    )


def edge_mask_and_costs(instance: AllocationInstance, context: OracleContext):
    mask = build_edge_mask(instance, context)
    return (mask,) + unit_robot_costs(instance, mask)
