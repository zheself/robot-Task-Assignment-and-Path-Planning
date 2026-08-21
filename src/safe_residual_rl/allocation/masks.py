"""Deterministic reason-coded robot-to-segment feasibility masks."""

from __future__ import annotations

from dataclasses import dataclass

from .oracle import EdgeEstimate, OracleContext, estimate_edge
from .schema import AllocationInstance


@dataclass(frozen=True)
class EdgeMask:
    segment_ids: tuple[str, ...]
    robot_ids: tuple[str, ...]
    allowed: tuple[tuple[bool, ...], ...]
    reason_codes: tuple[tuple[tuple[str, ...], ...], ...]
    estimates: tuple[tuple[EdgeEstimate, ...], ...]

    def is_allowed(self, segment_id: str, robot_id: str) -> bool:
        segment_index = self.segment_ids.index(segment_id)
        robot_index = self.robot_ids.index(robot_id)
        return self.allowed[segment_index][robot_index]


def build_edge_mask(instance: AllocationInstance, context: OracleContext) -> EdgeMask:
    rows = tuple(
        tuple(estimate_edge(robot, segment, context) for robot in instance.robots)
        for segment in instance.segments
    )
    return EdgeMask(
        segment_ids=tuple(segment.id for segment in instance.segments),
        robot_ids=tuple(robot.id for robot in instance.robots),
        allowed=tuple(tuple(item.feasible for item in row) for row in rows),
        reason_codes=tuple(tuple(item.reason_codes for item in row) for row in rows),
        estimates=rows,
    )
