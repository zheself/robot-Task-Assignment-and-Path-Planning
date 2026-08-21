"""Hard-mask, atomic-unit A3 decoding without A4 repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from .graphs import A3Graph
from .oracle import OracleContext
from .schema import AllocationInstance, AllocationPlan
from .scheduling import build_schedule
from .solvers.common import allocation_units


@dataclass(frozen=True)
class DecodedCandidate:
    status: str
    assignment: tuple[tuple[str, str], ...]
    robot_orders: tuple[tuple[str, tuple[str, ...]], ...]
    plan: AllocationPlan | None
    diagnostics: tuple[str, ...]


def decode_masked_candidate(
    graph: A3Graph,
    instance: AllocationInstance,
    context: OracleContext,
    assignment_logits: torch.Tensor | np.ndarray,
    order_scores: torch.Tensor | np.ndarray,
    *,
    method_id: str = "a3-masked-candidate-v1",
) -> DecodedCandidate:
    logits = _numpy(assignment_logits)
    priorities = _numpy(order_scores).reshape(-1)
    if logits.shape != graph.allowed_mask.shape or priorities.shape != (len(graph.segment_ids),):
        raise ValueError("decoder tensor shape mismatch")
    if np.any(np.isfinite(logits[~graph.allowed_mask])):
        raise ValueError("invalid-edge logits must remain negative infinity")
    segment_lookup = {item: index for index, item in enumerate(graph.segment_ids)}
    assignment: dict[str, str] = {}
    for unit in allocation_units(instance):
        indices = [segment_lookup[item] for item in unit]
        allowed = np.all(graph.allowed_mask[indices], axis=0)
        unit_logits = np.mean(logits[indices], axis=0)
        candidates = [
            (float(unit_logits[index]), graph.robot_ids[index])
            for index in range(len(graph.robot_ids))
            if allowed[index] and np.isfinite(unit_logits[index])
        ]
        if not candidates:
            return DecodedCandidate(
                "infeasible",
                (),
                (),
                None,
                ("ATOMIC_UNIT_WITHOUT_ALLOWED_ROBOT", "+".join(unit)),
            )
        _, robot_id = max(candidates, key=lambda item: (item[0], _reverse_id(item[1])))
        assignment.update({segment_id: robot_id for segment_id in unit})
    robot_orders = _precedence_aware_orders(instance, graph, assignment, priorities)
    built = build_schedule(instance, robot_orders, context, method_id)
    ordered = tuple((robot_id, tuple(robot_orders[robot_id])) for robot_id in sorted(robot_orders))
    flat_assignment = tuple(sorted(assignment.items()))
    if built.plan is None:
        return DecodedCandidate(
            "schedule_infeasible",
            flat_assignment,
            ordered,
            None,
            ("NO_REPAIR_APPLIED",) + built.diagnostics,
        )
    return DecodedCandidate(
        "feasible",
        flat_assignment,
        ordered,
        built.plan,
        ("HARD_MASK_APPLIED", "ATOMIC_UNITS_PRESERVED", "NO_REPAIR_APPLIED")
        + built.diagnostics,
    )


def _precedence_aware_orders(
    instance: AllocationInstance,
    graph: A3Graph,
    assignment: Mapping[str, str],
    priorities: np.ndarray,
) -> dict[str, tuple[str, ...]]:
    priority = {segment_id: float(priorities[index]) for index, segment_id in enumerate(graph.segment_ids)}
    dependencies: dict[str, set[str]] = {item.id: set(item.predecessor_ids) for item in instance.segments}
    groups: dict[str, list[object]] = {}
    for segment in instance.segments:
        groups.setdefault(segment.parent_curve_id, []).append(segment)
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item.segment_index)
        for left, right in zip(ordered, ordered[1:]):
            dependencies[right.id].add(left.id)
    result: dict[str, tuple[str, ...]] = {}
    for robot in sorted(instance.robots, key=lambda item: item.id):
        remaining = {segment_id for segment_id, robot_id in assignment.items() if robot_id == robot.id}
        order: list[str] = []
        while remaining:
            ready = [
                segment_id
                for segment_id in remaining
                if not (dependencies[segment_id] & remaining)
            ]
            if not ready:
                raise ValueError("decoder encountered cyclic robot-local dependencies")
            selected = min(ready, key=lambda item: (-priority[item], item))
            order.append(selected)
            remaining.remove(selected)
        result[robot.id] = tuple(order)
    return result


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _reverse_id(value: str) -> tuple[int, ...]:
    return tuple(-ord(character) for character in value)
