"""Auditable A4b search traces, cutoff snapshots and deterministic replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from ..repair import InitializerState


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def state_to_dict(state: InitializerState) -> dict[str, object]:
    return {
        "assignments": list(state.assignments),
        "robot_orders": [[robot, list(order)] for robot, order in state.robot_orders],
    }


def state_from_dict(value: Mapping[str, Any]) -> InitializerState:
    return InitializerState(
        tuple(None if item is None else str(item) for item in value["assignments"]),
        tuple((str(robot), tuple(int(item) for item in order)) for robot, order in value["robot_orders"]),
    )


def state_hash(state: InitializerState) -> str:
    return canonical_hash(state_to_dict(state))


@dataclass(frozen=True)
class IncumbentEvent:
    iteration: int
    monotonic_ns: int
    elapsed_s: float
    objective: float
    state: InitializerState
    state_sha256: str
    plan_sha256: str
    verifier_sha256: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["state"] = state_to_dict(self.state)
        return result


@dataclass(frozen=True)
class SearchStep:
    iteration: int
    iteration_seed: int
    operator: str
    destroy_ratio: float
    destroy_set: tuple[int, ...]
    before_state: InitializerState
    candidate_state: InitializerState
    before_verified: bool
    after_verified: bool
    before_failure_reason: str | None
    after_failure_reason: str | None
    current_objective: float | None
    best_so_far_objective: float | None
    objective_delta: float | None
    violation_reduction: float
    repair_runtime_s: float
    candidate_evaluations: int
    accepted: bool
    incumbent_updated: bool
    reward: float
    operator_weight_before: float
    operator_weight_after: float
    monotonic_ns: int
    elapsed_s: float
    before_state_sha256: str
    candidate_state_sha256: str
    plan_sha256: str | None
    verifier_sha256: str
    dynamic_features: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["destroy_set"] = list(self.destroy_set)
        result["before_state"] = state_to_dict(self.before_state)
        result["candidate_state"] = state_to_dict(self.candidate_state)
        result["dynamic_features"] = dict(self.dynamic_features)
        return result


@dataclass(frozen=True)
class SearchTrace:
    version: str
    protocol_id: str
    instance_id: str
    task_group_id: str
    difficulty: str
    split: str
    method: str
    random_seed: int
    clock: str
    start_monotonic_ns: int
    return_monotonic_ns: int
    initializer: Mapping[str, object]
    initial_state: InitializerState
    static_unit_features: tuple[Mapping[str, object], ...]
    config_sha256: str
    steps: tuple[SearchStep, ...]
    incumbents: tuple[IncumbentEvent, ...]
    termination_reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "protocol_id": self.protocol_id,
            "instance_id": self.instance_id,
            "task_group_id": self.task_group_id,
            "difficulty": self.difficulty,
            "split": self.split,
            "method": self.method,
            "random_seed": self.random_seed,
            "clock": self.clock,
            "start_monotonic_ns": self.start_monotonic_ns,
            "return_monotonic_ns": self.return_monotonic_ns,
            "initializer": dict(self.initializer),
            "initial_state": state_to_dict(self.initial_state),
            "static_unit_features": [dict(item) for item in self.static_unit_features],
            "config_sha256": self.config_sha256,
            "steps": [item.to_dict() for item in self.steps],
            "incumbents": [item.to_dict() for item in self.incumbents],
            "termination_reason": self.termination_reason,
        }

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class AnytimeSnapshot:
    budget_s: float
    verified: bool
    objective: float | None
    incumbent_iteration: int | None
    incumbent_elapsed_s: float | None
    incumbent_state_sha256: str | None
    failure_reason: str | None


def best_at_budget(trace: SearchTrace, budget_s: float) -> AnytimeSnapshot:
    """Return the best incumbent *observed no later than the cutoff*.

    The trace may return after ``budget_s``.  Return time is deliberately not
    used to erase an incumbent whose own monotonic timestamp is in budget.
    """
    if budget_s < 0:
        raise ValueError("budget must be non-negative")
    eligible = [item for item in trace.incumbents if item.elapsed_s <= budget_s + 1e-12]
    if not eligible:
        init_elapsed = float(trace.initializer.get("completion_elapsed_s", 0.0))
        reason = "initializer_timeout" if init_elapsed > budget_s + 1e-12 else "no_feasible_incumbent"
        return AnytimeSnapshot(budget_s, False, None, None, None, None, reason)
    selected = min(eligible, key=lambda item: (item.objective, item.elapsed_s, item.iteration))
    return AnytimeSnapshot(
        budget_s,
        True,
        selected.objective,
        selected.iteration,
        selected.elapsed_s,
        selected.state_sha256,
        None,
    )


def best_at_iteration(trace: SearchTrace, iteration: int) -> AnytimeSnapshot:
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    eligible = [item for item in trace.incumbents if item.iteration <= iteration]
    if not eligible:
        return AnytimeSnapshot(float(iteration), False, None, None, None, None, "no_feasible_incumbent")
    selected = min(eligible, key=lambda item: (item.objective, item.iteration, item.elapsed_s))
    return AnytimeSnapshot(
        float(iteration), True, selected.objective, selected.iteration,
        selected.elapsed_s, selected.state_sha256, None,
    )


def replay_trace(
    trace: SearchTrace,
    replay_step: Callable[[SearchStep], InitializerState],
) -> str:
    """Replay every candidate transition and verify the recorded state hash."""
    previous = trace.initial_state
    for step in trace.steps:
        if state_hash(previous) != step.before_state_sha256:
            raise RuntimeError(f"trace before-state mismatch at iteration {step.iteration}")
        candidate = replay_step(step)
        if state_hash(candidate) != step.candidate_state_sha256:
            raise RuntimeError(f"trace replay mismatch at iteration {step.iteration}")
        previous = candidate if step.accepted else previous
    return trace.sha256


def violation_score(verified: bool, failure_reason: str | None, surrogate: float) -> float:
    if verified:
        return 0.0
    severity = {
        "initializer_incomplete": 5.0,
        "mask_integrity_failure": 5.0,
        "precedence_failure": 4.0,
        "time_window_failure": 3.0,
        "shared_resource_failure": 3.0,
        "schedule_infeasible": 2.0,
    }.get(failure_reason, 1.0)
    return severity + min(max(surrogate, 0.0), 10_000_000.0) / 10_000_000.0
