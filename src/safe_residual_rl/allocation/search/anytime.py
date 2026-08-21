"""Initializer provenance and monotonic anytime semantics for A4b."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Mapping

from ..oracle import OracleContext
from ..repair import InitializerState, evaluate_state, state_from_plan
from ..schema import AllocationInstance
from ..solvers import SolverResult, solve_hybrid_load_balanced
from ..solvers.common import allocation_units, orders_from_assignment
from ..verifier import verify_plan
from .trace import canonical_hash, state_hash


@dataclass(frozen=True)
class InitializerProvenance:
    requested_initializer: str
    actual_initializer: str
    solver_status: str
    has_true_incumbent: bool
    fallback_used: bool
    fallback_reason: str | None
    initializer_plan_hash: str | None
    verifier_feasible: bool
    verifier_failure_reason: str | None
    start_monotonic_ns: int
    completion_monotonic_ns: int
    completion_elapsed_s: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InitializerOutcome:
    state: InitializerState
    provenance: InitializerProvenance


def _state_from_assignment_incumbent(
    instance: AllocationInstance,
    incumbent: tuple[tuple[int, str], ...],
) -> InitializerState:
    units = allocation_units(instance)
    mapping = dict(incumbent)
    if set(mapping) != set(range(len(units))):
        raise ValueError("assignment incumbent does not cover every atomic unit")
    orders = orders_from_assignment(instance, mapping, units)
    segment_to_unit = {segment: index for index, unit in enumerate(units) for segment in unit}
    unit_orders = []
    for robot in sorted(orders):
        order = []
        for segment in orders[robot]:
            index = segment_to_unit[segment]
            if index not in order:
                order.append(index)
        unit_orders.append((robot, tuple(order)))
    return InitializerState(
        tuple(mapping[index] for index in range(len(units))),
        tuple(unit_orders),
    )


def adapt_solver_initializer(
    instance: AllocationInstance,
    context: OracleContext,
    requested_initializer: str,
    solver_result: SolverResult,
    weights: Mapping[str, float],
    *,
    start_monotonic_ns: int,
    completion_monotonic_ns: int | None = None,
    fallback: InitializerOutcome | None = None,
) -> InitializerOutcome:
    """Preserve a solver's actual state and label; never silently relabel fallback."""
    completed = time.monotonic_ns() if completion_monotonic_ns is None else completion_monotonic_ns
    if completed < start_monotonic_ns:
        raise ValueError("monotonic clock moved backwards")
    actual = requested_initializer
    fallback_used = False
    fallback_reason = None
    has_true = False
    verifier_feasible = False
    verifier_failure = None
    plan_hash = None

    if solver_result.plan is not None:
        state = state_from_plan(instance, solver_result.plan)
        checked = verify_plan(instance, solver_result.plan, context)
        verifier_feasible = checked.feasible
        verifier_failure = None if checked.feasible else "verification_failed"
        has_true = True
        plan_hash = canonical_hash(solver_result.plan.to_dict())
    elif solver_result.assignment_incumbent is not None:
        state = _state_from_assignment_incumbent(instance, solver_result.assignment_incumbent)
        evaluated = evaluate_state(instance, context, state, weights, method_id="a4b-initializer-audit")
        verifier_feasible = evaluated.verified
        verifier_failure = evaluated.failure_reason
        has_true = True
        actual = f"{requested_initializer}_assignment_incumbent"
        plan_hash = state_hash(state)
    elif fallback is not None:
        state = fallback.state
        fallback_used = True
        fallback_reason = f"{requested_initializer}:{solver_result.status}:no_incumbent"
        actual = fallback.provenance.actual_initializer
        verifier_feasible = fallback.provenance.verifier_feasible
        verifier_failure = fallback.provenance.verifier_failure_reason
        plan_hash = fallback.provenance.initializer_plan_hash
    else:
        robots = tuple(sorted(item.id for item in instance.robots))
        state = InitializerState(
            tuple(None for _ in allocation_units(instance)),
            tuple((robot, ()) for robot in robots),
        )
        verifier_failure = "initializer_incomplete"

    provenance = InitializerProvenance(
        requested_initializer=requested_initializer,
        actual_initializer=actual,
        solver_status=solver_result.status,
        has_true_incumbent=has_true,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        initializer_plan_hash=plan_hash,
        verifier_feasible=verifier_feasible,
        verifier_failure_reason=verifier_failure,
        start_monotonic_ns=start_monotonic_ns,
        completion_monotonic_ns=completed,
        completion_elapsed_s=(completed - start_monotonic_ns) / 1e9,
    )
    return InitializerOutcome(state, provenance)


def build_hybrid_load_balanced_initializer(
    instance: AllocationInstance,
    context: OracleContext,
    weights: Mapping[str, float],
    *,
    clock: Callable[[], int] = time.monotonic_ns,
) -> InitializerOutcome:
    started = clock()
    result = solve_hybrid_load_balanced(instance, context)
    completed = clock()
    return adapt_solver_initializer(
        instance,
        context,
        "hybrid_load_balanced",
        result,
        weights,
        start_monotonic_ns=started,
        completion_monotonic_ns=completed,
    )


def provenance_hash(outcome: InitializerOutcome) -> str:
    return canonical_hash(outcome.provenance.to_dict())
