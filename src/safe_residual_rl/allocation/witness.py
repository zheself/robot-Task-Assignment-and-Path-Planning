"""Constructive proxy-feasibility witnesses for future A2 benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Mapping

from .generation import canonical_instance_bytes
from .oracle import OracleContext
from .schema import AllocationInstance, AllocationPlan, TimeWindow
from .scheduling import build_deadline_aware_schedule
from .solvers.common import edge_mask_and_costs
from .verifier import verify_plan


@dataclass(frozen=True)
class ConstructiveWitness:
    instance: AllocationInstance
    plan: AllocationPlan
    witness_sha256: str
    diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance.instance_id,
            "instance_sha256": hashlib.sha256(
                canonical_instance_bytes(self.instance)
            ).hexdigest(),
            "plan": self.plan.to_dict(),
            "witness_sha256": self.witness_sha256,
            "diagnostics": list(self.diagnostics),
        }


def construct_feasible_witness(
    instance: AllocationInstance,
    context: OracleContext,
    *,
    tight_pre_margin_duration: float = 0.25,
    tight_post_margin_duration: float = 0.75,
    loose_pre_margin_s: float = 30.0,
) -> ConstructiveWitness:
    """Return a window-calibrated instance and independently verified witness.

    Geometry, capabilities, precedence and resources are preserved. Time windows
    are reconstructed around a deterministic proxy schedule, so feasibility is
    known by construction instead of inferred from solver success.
    """
    if min(tight_pre_margin_duration, tight_post_margin_duration, loose_pre_margin_s) < 0:
        raise ValueError("witness margins must be non-negative")
    horizon = min(
        [item.availability.end_s for item in instance.robots]
        + [item.availability.end_s for item in instance.resources]
    )
    relaxed = replace(
        instance,
        segments=tuple(
            replace(item, time_window=TimeWindow(0.0, horizon))
            for item in instance.segments
        ),
    )
    _, units, robots, costs = edge_mask_and_costs(relaxed, context)
    if any(not any(math.isfinite(value) for value in row) for row in costs):
        raise ValueError("cannot construct witness: assignment unit lacks feasible robot")
    loads = {robot_id: 0.0 for robot_id in robots}
    unit_assignment: dict[int, str] = {}
    for index, row in enumerate(costs):
        _, robot_id, raw_cost = min(
            (loads[robot_id] + value, robot_id, value)
            for robot_id, value in zip(robots, row)
            if math.isfinite(value)
        )
        unit_assignment[index] = robot_id
        loads[robot_id] += raw_cost
    assignment = {
        segment_id: unit_assignment[index]
        for index, unit in enumerate(units)
        for segment_id in unit
    }
    built = build_deadline_aware_schedule(
        relaxed, assignment, context, "constructive-witness-v1"
    )
    if built.plan is None:
        raise ValueError("cannot construct witness within registered horizon")

    schedule_by_id = {item.segment_id: item for item in built.plan.schedule}
    calibrated_segments = []
    tight_count = 0
    for segment in instance.segments:
        scheduled = schedule_by_id[segment.id]
        original_width = segment.time_window.end_s - segment.time_window.start_s
        tight = original_width < 0.5 * horizon
        if tight:
            tight_count += 1
            start = max(
                0.0,
                scheduled.planned_start_s
                - tight_pre_margin_duration * segment.process_duration_s,
            )
            end = min(
                horizon,
                scheduled.planned_end_s
                + tight_post_margin_duration * segment.process_duration_s,
            )
        else:
            start = max(0.0, scheduled.planned_start_s - loose_pre_margin_s)
            end = horizon
        calibrated_segments.append(
            replace(segment, time_window=TimeWindow(start, end))
        )
    calibrated = replace(instance, segments=tuple(calibrated_segments))
    verification = verify_plan(calibrated, built.plan, context)
    if not verification.feasible:
        codes = ",".join(sorted({item.code for item in verification.violations}))
        raise ValueError(f"constructed witness failed verification: {codes}")
    digest = _witness_digest(calibrated, built.plan)
    return ConstructiveWitness(
        calibrated,
        built.plan,
        digest,
        (
            "CONSTRUCTIVE_A1_PROXY_FEASIBILITY_WITNESS",
            f"TIGHT_WINDOWS_RECALIBRATED={tight_count}",
            f"HORIZON_S={horizon:.9g}",
            "GEOMETRY_CAPABILITY_PRECEDENCE_AND_RESOURCES_PRESERVED",
            "TIME_WINDOWS_RECONSTRUCTED_AROUND_WITNESS",
            "NOT_REAL_EXECUTION_COLLISION_OR_PHYSICAL_EVIDENCE",
        ),
    )


def verify_constructive_witness(
    witness: ConstructiveWitness, context: OracleContext
) -> tuple[str, ...]:
    issues = []
    if _witness_digest(witness.instance, witness.plan) != witness.witness_sha256:
        issues.append("WITNESS_HASH_MISMATCH")
    checked = verify_plan(witness.instance, witness.plan, context)
    if not checked.feasible:
        issues.extend(f"VERIFY={item.code}" for item in checked.violations)
    return tuple(issues)


def _witness_digest(instance: AllocationInstance, plan: AllocationPlan) -> str:
    payload: Mapping[str, object] = {
        "instance": instance.to_dict(),
        "plan": plan.to_dict(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
