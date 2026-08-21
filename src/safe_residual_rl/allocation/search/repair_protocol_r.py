"""Audited reference and prepared repair backends for Protocol R."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from ..oracle import OracleContext
from ..repair import InitializerState
from ..repair.identical import canonicalize_state
from ..schema import AllocationInstance
from .alns_v2 import RepairV2Outcome, repair_destroyed_state_v2
from .diagnostics import StateDiagnostic, TimedEvaluation, analyze_state
from .prepared_repair import (
    PreparedRepairProblem,
    analyze_state_prepared,
    evaluate_state_timed_prepared,
    prepare_repair_problem,
)
from .trace import canonical_hash, state_hash


@dataclass
class RepairTraceCache:
    """Trace-local cache; sharing across methods, seeds or traces is forbidden."""

    diagnostics: dict[str, StateDiagnostic] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0


@dataclass(frozen=True)
class RepairProtocolROutcome:
    state: InitializerState
    evaluation: TimedEvaluation
    runtime_s: float
    selection_runtime_s: float
    candidate_evaluations: int
    budget_exhausted: bool
    deadline_exhausted: bool
    assignment_edits: int
    order_edits: int
    total_modified_units: int
    first_feasible_elapsed_s: float | None
    candidate_sequence_sha256: str
    candidate_sequence: tuple[tuple[int, str, int, str], ...]
    diagnostic_vector_sha256: str
    selected_state_sha256: str
    cache_hits: int
    cache_misses: int
    prepared_sha256: str


def _positions(state: InitializerState) -> dict[int, tuple[str, int]]:
    return {
        unit: (robot, position)
        for robot, order in state.robot_orders
        for position, unit in enumerate(order)
    }


def _edit_counts(initial: InitializerState, final: InitializerState) -> tuple[int, int, int]:
    assignment_changed = {
        index
        for index, (left, right) in enumerate(zip(initial.assignments, final.assignments))
        if left != right
    }
    before, after = _positions(initial), _positions(final)
    order_changed = {
        unit for unit in set(before) | set(after) if before.get(unit) != after.get(unit)
    }
    return len(assignment_changed), len(order_changed), len(assignment_changed | order_changed)


def _make_state(
    instance: AllocationInstance,
    assignments: Sequence[str | None],
    orders: Mapping[str, Sequence[int]],
    robots: Sequence[str],
) -> InitializerState:
    return canonicalize_state(
        instance,
        InitializerState(
            tuple(assignments), tuple((robot, tuple(orders[robot])) for robot in robots)
        ),
    )


def _vector_payload(diagnostic: StateDiagnostic) -> dict[str, object]:
    return {
        "vector": diagnostic.vector.to_dict(),
        "failure_segment_ids": list(diagnostic.failure_segment_ids),
        "failure_unit_indices": list(diagnostic.failure_unit_indices),
    }


def _repair_impl(
    instance: AllocationInstance,
    context: OracleContext,
    prepared: PreparedRepairProblem,
    state: InitializerState,
    destroy_set: Sequence[int],
    *,
    weights: Mapping[str, float],
    candidate_evaluation_budget: int,
    accelerated: bool,
    cache: RepairTraceCache | None,
    deadline_ns: int | None,
    clock: Callable[[], int],
) -> RepairProtocolROutcome:
    started = clock()
    units, robots, costs = prepared.units, prepared.robots, prepared.costs
    destroyed = tuple(int(item) for item in destroy_set)
    if len(set(destroyed)) != len(destroyed) or any(
        item < 0 or item >= len(units) for item in destroyed
    ):
        raise ValueError("repair received duplicate or non-atomic units")
    assignments = list(state.assignments)
    orders = state.order_map()
    for index in destroyed:
        assignments[index] = None
        for order in orders.values():
            if index in order:
                order.remove(index)

    local_cache = cache if cache is not None else RepairTraceCache()
    initial_hits, initial_misses = local_cache.hits, local_cache.misses
    candidate_sequence: list[tuple[int, str, int, str]] = []
    diagnostic_payloads: list[dict[str, object]] = []

    def diagnose(trial: InitializerState) -> tuple[str, StateDiagnostic]:
        digest = state_hash(trial)
        if accelerated and digest in local_cache.diagnostics:
            local_cache.hits += 1
            return digest, local_cache.diagnostics[digest]
        diagnostic = (
            analyze_state_prepared(instance, prepared, trial)
            if accelerated
            else analyze_state(instance, context, trial)
        )
        local_cache.misses += 1
        if accelerated:
            local_cache.diagnostics[digest] = diagnostic
        return digest, diagnostic

    def fallback_insert(
        unit_index: int,
    ) -> tuple[list[str | None], dict[str, list[int]]]:
        best = None
        for robot_index, robot in enumerate(robots):
            edge_cost = costs[unit_index][robot_index]
            if not math.isfinite(edge_cost):
                continue
            for position in range(len(orders[robot]) + 1):
                trial_assignments = list(assignments)
                trial_assignments[unit_index] = robot
                trial_orders = {key: list(value) for key, value in orders.items()}
                trial_orders[robot].insert(position, unit_index)
                trial = _make_state(instance, trial_assignments, trial_orders, robots)
                digest, diagnostic = diagnose(trial)
                key = (
                    diagnostic.vector.rank(), edge_cost, robot, position, digest
                )
                if best is None or key < best[0]:
                    best = (key, trial)
        if best is None:
            raise RuntimeError("no edge-feasible fallback insertion")
        trial = best[1]
        return list(trial.assignments), trial.order_map()

    remaining = list(destroyed)
    evaluations = 0
    budget_exhausted = False
    deadline_exhausted = False
    while remaining:
        if evaluations >= candidate_evaluation_budget:
            budget_exhausted = True
        if deadline_ns is not None and clock() >= deadline_ns:
            deadline_exhausted = True
        if budget_exhausted or deadline_exhausted:
            selected = min(remaining)
            assignments, orders = fallback_insert(selected)
            remaining.remove(selected)
            continue

        per_unit = []
        stop = False
        for unit_index in remaining:
            choices = []
            for robot_index, robot in enumerate(robots):
                if not math.isfinite(costs[unit_index][robot_index]):
                    continue
                for position in range(len(orders[robot]) + 1):
                    if evaluations >= candidate_evaluation_budget:
                        budget_exhausted = True
                        stop = True
                        break
                    if deadline_ns is not None and clock() >= deadline_ns:
                        deadline_exhausted = True
                        stop = True
                        break
                    trial_assignments = list(assignments)
                    trial_assignments[unit_index] = robot
                    trial_orders = {key: list(value) for key, value in orders.items()}
                    trial_orders[robot].insert(position, unit_index)
                    trial = _make_state(instance, trial_assignments, trial_orders, robots)
                    digest, diagnostic = diagnose(trial)
                    evaluations += 1
                    candidate_sequence.append((unit_index, robot, position, digest))
                    diagnostic_payloads.append(_vector_payload(diagnostic))
                    choices.append(
                        (
                            diagnostic.vector.rank(),
                            diagnostic.vector.scalar(),
                            digest,
                            trial,
                        )
                    )
                if stop:
                    break
            if choices:
                choices.sort(key=lambda item: item[:3])
                best = choices[0]
                second = choices[1][1] if len(choices) > 1 else best[1]
                regret = second - best[1]
                per_unit.append((-regret, best[0], unit_index, best[2], best[3]))
            if stop:
                break
        if not per_unit:
            continue
        _, _, selected, _, trial = min(per_unit)
        assignments = list(trial.assignments)
        orders = trial.order_map()
        remaining.remove(selected)

    final_state = _make_state(instance, assignments, orders, robots)
    selection_done = clock()
    final = evaluate_state_timed_prepared(
        instance, context, prepared, final_state, weights, clock=clock
    )
    returned = clock()
    assignment_edits, order_edits, total = _edit_counts(state, final_state)
    first_feasible = (returned - started) / 1e9 if final.evaluation.verified else None
    return RepairProtocolROutcome(
        final_state,
        final,
        (returned - started) / 1e9,
        (selection_done - started) / 1e9 + final.timing.structural_s,
        evaluations,
        budget_exhausted,
        deadline_exhausted,
        assignment_edits,
        order_edits,
        total,
        first_feasible,
        canonical_hash(candidate_sequence),
        tuple(candidate_sequence),
        canonical_hash(diagnostic_payloads),
        state_hash(final_state),
        local_cache.hits - initial_hits,
        local_cache.misses - initial_misses,
        prepared.prepared_sha256,
    )


def repair_destroyed_state_protocol_r(
    instance: AllocationInstance,
    context: OracleContext,
    state: InitializerState,
    destroy_set: Sequence[int],
    *,
    weights: Mapping[str, float],
    candidate_evaluation_budget: int,
    prepared: PreparedRepairProblem | None = None,
    cache: RepairTraceCache | None = None,
    deadline_ns: int | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> RepairProtocolROutcome:
    problem = prepared or prepare_repair_problem(instance, context)
    return _repair_impl(
        instance,
        context,
        problem,
        state,
        destroy_set,
        weights=weights,
        candidate_evaluation_budget=candidate_evaluation_budget,
        accelerated=True,
        cache=cache,
        deadline_ns=deadline_ns,
        clock=clock,
    )


def repair_destroyed_state_reference_audited(
    instance: AllocationInstance,
    context: OracleContext,
    state: InitializerState,
    destroy_set: Sequence[int],
    *,
    weights: Mapping[str, float],
    candidate_evaluation_budget: int,
    prepared: PreparedRepairProblem | None = None,
    deadline_ns: int | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> RepairProtocolROutcome:
    """Reference loop with audit signatures; partial diagnostics remain v2 calls."""
    problem = prepared or prepare_repair_problem(instance, context)
    return _repair_impl(
        instance,
        context,
        problem,
        state,
        destroy_set,
        weights=weights,
        candidate_evaluation_budget=candidate_evaluation_budget,
        accelerated=False,
        cache=None,
        deadline_ns=deadline_ns,
        clock=clock,
    )


def parity_report(
    reference: RepairProtocolROutcome,
    accelerated: RepairProtocolROutcome,
    *,
    historical_reference: RepairV2Outcome | None = None,
) -> dict[str, object]:
    """Return an exact, fail-closed semantic comparison."""
    ref_plan = reference.evaluation.evaluation.plan
    accelerated_plan = accelerated.evaluation.evaluation.plan
    checks = {
        "candidate_sequence": reference.candidate_sequence_sha256
        == accelerated.candidate_sequence_sha256,
        "diagnostic_vectors": reference.diagnostic_vector_sha256
        == accelerated.diagnostic_vector_sha256,
        "selected_state": reference.selected_state_sha256
        == accelerated.selected_state_sha256,
        "evaluation": (
            reference.evaluation.evaluation.verified,
            reference.evaluation.evaluation.objective,
            reference.evaluation.evaluation.surrogate,
            reference.evaluation.evaluation.failure_reason,
        )
        == (
            accelerated.evaluation.evaluation.verified,
            accelerated.evaluation.evaluation.objective,
            accelerated.evaluation.evaluation.surrogate,
            accelerated.evaluation.evaluation.failure_reason,
        ),
        "plan": canonical_hash(None if ref_plan is None else ref_plan.to_dict())
        == canonical_hash(None if accelerated_plan is None else accelerated_plan.to_dict()),
        "counts_and_flags": (
            reference.candidate_evaluations,
            reference.budget_exhausted,
            reference.deadline_exhausted,
        )
        == (
            accelerated.candidate_evaluations,
            accelerated.budget_exhausted,
            accelerated.deadline_exhausted,
        ),
    }
    if historical_reference is not None:
        checks["historical_v2_state"] = (
            state_hash(historical_reference.state) == accelerated.selected_state_sha256
        )
        checks["historical_v2_evaluation"] = (
            historical_reference.evaluation.evaluation.verified,
            historical_reference.evaluation.evaluation.objective,
            historical_reference.evaluation.evaluation.surrogate,
            historical_reference.evaluation.evaluation.failure_reason,
        ) == (
            accelerated.evaluation.evaluation.verified,
            accelerated.evaluation.evaluation.objective,
            accelerated.evaluation.evaluation.surrogate,
            accelerated.evaluation.evaluation.failure_reason,
        )
    return {
        "version": "a4b-protocol-r-repair-parity-v1",
        "checks": checks,
        "passed": all(checks.values()),
        "reference_candidate_sequence_sha256": reference.candidate_sequence_sha256,
        "accelerated_candidate_sequence_sha256": accelerated.candidate_sequence_sha256,
    }


def compare_with_historical_v2(
    instance: AllocationInstance,
    context: OracleContext,
    state: InitializerState,
    destroy_set: Sequence[int],
    *,
    weights: Mapping[str, float],
    candidate_evaluation_budget: int,
) -> dict[str, object]:
    """Convenience fixture comparison; never used for timed evidence."""
    prepared = prepare_repair_problem(instance, context)
    historical = repair_destroyed_state_v2(
        instance,
        context,
        state,
        destroy_set,
        weights=weights,
        candidate_evaluation_budget=candidate_evaluation_budget,
    )
    reference = repair_destroyed_state_reference_audited(
        instance,
        context,
        state,
        destroy_set,
        weights=weights,
        candidate_evaluation_budget=candidate_evaluation_budget,
        prepared=prepared,
    )
    accelerated = repair_destroyed_state_protocol_r(
        instance,
        context,
        state,
        destroy_set,
        weights=weights,
        candidate_evaluation_budget=candidate_evaluation_budget,
        prepared=prepared,
    )
    return parity_report(reference, accelerated, historical_reference=historical)
