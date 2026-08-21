"""Initializer-agnostic atomic-unit ALNS used only by the A4a pilot.

The backend deliberately has no learned-method branch.  Every initializer is
converted to :class:`InitializerState`, then receives the same operators,
budgets, acceptance rule, scheduler and verifier.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..oracle import OracleContext
from ..schema import AllocationInstance, AllocationPlan
from ..scheduling import build_schedule
from ..solvers.common import allocation_units, edge_mask_and_costs
from ..verifier import verify_plan


@dataclass(frozen=True)
class InitializerState:
    assignments: tuple[str | None, ...]
    robot_orders: tuple[tuple[str, tuple[int, ...]], ...]

    def order_map(self) -> dict[str, list[int]]:
        return {robot: list(order) for robot, order in self.robot_orders}


@dataclass(frozen=True)
class StateEvaluation:
    verified: bool
    plan: AllocationPlan | None
    objective: float | None
    surrogate: float
    failure_reason: str | None


@dataclass(frozen=True)
class RepairTraceStep:
    iteration: int
    operator: str
    accepted: bool
    verified: bool
    objective: float | None
    surrogate: float
    elapsed_s: float

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RepairResult:
    status: str
    initial_evaluation: StateEvaluation
    final_evaluation: StateEvaluation
    final_state: InitializerState
    first_feasible_iteration: int | None
    first_feasible_time_s: float | None
    time_to_target_s: float | None
    iterations_completed: int
    destroy_reinsert_count: int
    assignment_modifications: int
    order_modifications: int
    modified_atomic_units: int
    initializer_assignment_retention: float
    repair_runtime_s: float
    timed_out: bool
    trace: tuple[RepairTraceStep, ...]

    def to_dict(self) -> dict[str, object]:
        value = dict(self.__dict__)
        value["initial_evaluation"] = _evaluation_dict(self.initial_evaluation)
        value["final_evaluation"] = _evaluation_dict(self.final_evaluation)
        value["final_state"] = {
            "assignments": list(self.final_state.assignments),
            "robot_orders": [[r, list(o)] for r, o in self.final_state.robot_orders],
        }
        value["trace"] = [item.to_dict() for item in self.trace]
        return value


def state_from_plan(instance: AllocationInstance, plan: AllocationPlan) -> InitializerState:
    units = allocation_units(instance)
    scheduled = {item.segment_id: item for item in plan.schedule}
    assignments: list[str | None] = []
    for unit in units:
        robots = {scheduled[item].robot_id for item in unit if item in scheduled}
        assignments.append(next(iter(robots)) if len(robots) == 1 and len([x for x in unit if x in scheduled]) == len(unit) else None)
    segment_to_unit = {segment: index for index, unit in enumerate(units) for segment in unit}
    orders: list[tuple[str, tuple[int, ...]]] = []
    for robot in sorted(instance.robots, key=lambda x: x.id):
        items = sorted((x for x in plan.schedule if x.robot_id == robot.id), key=lambda x: x.order_index)
        unit_order: list[int] = []
        for item in items:
            index = segment_to_unit[item.segment_id]
            if index not in unit_order:
                unit_order.append(index)
        orders.append((robot.id, tuple(unit_order)))
    return canonicalize_state(instance, InitializerState(tuple(assignments), tuple(orders)))


def canonicalize_state(instance: AllocationInstance, state: InitializerState) -> InitializerState:
    robots = tuple(sorted(item.id for item in instance.robots))
    orders = state.order_map()
    seen: set[int] = set()
    fixed: dict[str, list[int]] = {robot: [] for robot in robots}
    for robot in robots:
        for unit_index in orders.get(robot, []):
            if unit_index in seen or unit_index >= len(state.assignments):
                continue
            if state.assignments[unit_index] == robot:
                fixed[robot].append(unit_index)
                seen.add(unit_index)
    for index, robot in enumerate(state.assignments):
        if robot in fixed and index not in seen:
            fixed[robot].append(index)
    return InitializerState(tuple(state.assignments), tuple((r, tuple(fixed[r])) for r in robots))


def evaluate_state(
    instance: AllocationInstance,
    context: OracleContext,
    state: InitializerState,
    weights: Mapping[str, float],
    *,
    method_id: str = "a4-identical-repair-v1",
) -> StateEvaluation:
    units = allocation_units(instance)
    _, _, robots, costs = edge_mask_and_costs(instance, context)
    missing = sum(item is None for item in state.assignments)
    invalid = 0
    loads = {robot: 0.0 for robot in robots}
    for index, robot in enumerate(state.assignments):
        if robot is None or robot not in robots:
            continue
        value = costs[index][robots.index(robot)]
        if not math.isfinite(value):
            invalid += 1
        else:
            loads[robot] += value
    state = canonicalize_state(instance, state)
    listed = [item for _, order in state.robot_orders for item in order]
    integrity = len(listed) != len(set(listed)) or set(listed) != {i for i, r in enumerate(state.assignments) if r is not None}
    surrogate = 1_000_000.0 * missing + 1_000_000.0 * invalid + 500_000.0 * int(integrity) + max(loads.values(), default=0.0)
    if missing or invalid or integrity:
        reason = "initializer_incomplete" if missing else "mask_integrity_failure"
        return StateEvaluation(False, None, None, surrogate, reason)
    robot_orders = {}
    for robot, order in state.robot_orders:
        robot_orders[robot] = tuple(segment for unit_index in order for segment in units[unit_index])
    built = build_schedule(instance, robot_orders, context, method_id)
    if built.plan is None:
        diagnostic = " ".join(built.diagnostics)
        if "TIME_OR_RESOURCE" in diagnostic:
            reason = "time_window_failure"
        elif "DEPENDENCY_OR_ORDER" in diagnostic:
            reason = "precedence_failure"
        else:
            reason = "schedule_infeasible"
        return StateEvaluation(False, None, None, surrogate + 100_000.0, reason)
    checked = verify_plan(instance, built.plan, context)
    if not checked.feasible:
        codes = {item.code for item in checked.violations}
        if any("RESOURCE" in item for item in codes):
            reason = "shared_resource_failure"
        elif any("PRECEDENCE" in item or "ORDER" in item for item in codes):
            reason = "precedence_failure"
        elif any("TIME" in item for item in codes):
            reason = "time_window_failure"
        else:
            reason = "schedule_infeasible"
        return StateEvaluation(False, built.plan, None, surrogate + 100_000.0, reason)
    terms = dict(checked.objective_terms)
    objective = sum(float(weights.get(key, 0.0)) * float(value) for key, value in terms.items())
    return StateEvaluation(True, built.plan, objective, objective, None)


def identical_repair(
    instance: AllocationInstance,
    context: OracleContext,
    initial_state: InitializerState,
    *,
    iterations: int,
    random_seed: int,
    weights: Mapping[str, float],
    time_limit_s: float | None = None,
    target_score: float | None = None,
    destroy_fractions: Sequence[float] = (0.1, 0.25, 0.4),
) -> RepairResult:
    if iterations < 0 or (time_limit_s is not None and time_limit_s < 0):
        raise ValueError("invalid repair budget")
    started = time.perf_counter()
    rng = np.random.default_rng(random_seed)
    units, robots, costs = _problem(instance, context)
    initial = canonicalize_state(instance, initial_state)
    initial_eval = evaluate_state(instance, context, initial, weights)
    current, current_eval = initial, initial_eval
    best, best_eval = (initial, initial_eval) if initial_eval.verified else (initial, initial_eval)
    first_iteration = 0 if initial_eval.verified else None
    first_time = 0.0 if initial_eval.verified else None
    target_time = 0.0 if initial_eval.verified and target_score is not None and initial_eval.objective is not None and initial_eval.objective <= target_score else None
    trace: list[RepairTraceStep] = []
    destroy_count = 0
    timed_out = False
    completed = 0

    for iteration in range(1, iterations + 1):
        elapsed = time.perf_counter() - started
        if time_limit_s is not None and elapsed >= time_limit_s:
            timed_out = True
            break
        operator = ("random_destroy", "worst_load_destroy", "atomic_reassign", "robot_local_relocate", "robot_local_swap", "precedence_safe_reorder")[(iteration - 1) % 6]
        candidate = _mutate(current, operator, iteration, rng, robots, costs, destroy_fractions)
        if "destroy" in operator:
            destroy_count += 1
        candidate_eval = evaluate_state(instance, context, candidate, weights)
        accepted = _accept(current_eval, candidate_eval, iteration, rng)
        if accepted:
            current, current_eval = candidate, candidate_eval
        if candidate_eval.verified and (not best_eval.verified or float(candidate_eval.objective) < float(best_eval.objective)):
            best, best_eval = candidate, candidate_eval
        elapsed = time.perf_counter() - started
        if first_iteration is None and candidate_eval.verified:
            first_iteration, first_time = iteration, elapsed
        if target_time is None and target_score is not None and candidate_eval.verified and float(candidate_eval.objective) <= target_score:
            target_time = elapsed
        trace.append(RepairTraceStep(iteration, operator, accepted, candidate_eval.verified, candidate_eval.objective, candidate_eval.surrogate, elapsed))
        completed = iteration

    final_state, final_eval = (best, best_eval) if best_eval.verified else (current, current_eval)
    assignment_changes, order_changes, modified, retention = _edit_distance(initial, final_state)
    runtime = time.perf_counter() - started
    status = "feasible" if final_eval.verified else ("repair_timeout" if timed_out else final_eval.failure_reason or "repair_dead_end")
    return RepairResult(status, initial_eval, final_eval, final_state, first_iteration, first_time, target_time, completed, destroy_count, assignment_changes, order_changes, modified, retention, runtime, timed_out, tuple(trace))


def _problem(instance: AllocationInstance, context: OracleContext):
    _, units, robots, costs = edge_mask_and_costs(instance, context)
    return units, robots, costs


def _mutate(state, operator, iteration, rng, robots, costs, fractions):
    assignment = list(state.assignments)
    orders = state.order_map()
    count = len(assignment)
    if any(item is None for item in assignment):
        destroyed = [i for i, item in enumerate(assignment) if item is None]
        _reinsert(assignment, orders, destroyed, robots, costs, rng)
        return _make_state(assignment, orders, robots)
    if operator in {"random_destroy", "worst_load_destroy"}:
        fraction = fractions[(iteration - 1) % len(fractions)]
        amount = max(1, int(math.ceil(fraction * count)))
        if operator == "random_destroy":
            destroyed = [int(x) for x in rng.choice(count, min(amount, count), replace=False)]
        else:
            loads = {r: sum(costs[i][robots.index(r)] for i, x in enumerate(assignment) if x == r) for r in robots}
            worst = max(robots, key=lambda r: (loads[r], r))
            candidates = [i for i, r in enumerate(assignment) if r == worst]
            destroyed = candidates[:amount] or [int(rng.integers(0, count))]
        for index in destroyed:
            assignment[index] = None
            for order in orders.values():
                if index in order:
                    order.remove(index)
        _reinsert(assignment, orders, destroyed, robots, costs, rng)
    elif operator == "atomic_reassign":
        index = int(rng.integers(0, count))
        choices = [r for j, r in enumerate(robots) if math.isfinite(costs[index][j]) and r != assignment[index]]
        if choices:
            old = assignment[index]
            if index in orders[old]:
                orders[old].remove(index)
            new = choices[int(rng.integers(0, len(choices)))]
            assignment[index] = new
            orders[new].append(index)
    elif operator in {"robot_local_relocate", "robot_local_swap"}:
        candidates = [r for r in robots if len(orders[r]) >= (1 if operator.endswith("relocate") else 2)]
        if candidates:
            robot = candidates[int(rng.integers(0, len(candidates)))]
            if operator.endswith("relocate"):
                old = int(rng.integers(0, len(orders[robot])))
                item = orders[robot].pop(old)
                new = int(rng.integers(0, len(orders[robot]) + 1))
                orders[robot].insert(new, item)
            else:
                a, b = rng.choice(len(orders[robot]), 2, replace=False).tolist()
                orders[robot][a], orders[robot][b] = orders[robot][b], orders[robot][a]
    else:
        for robot in robots:
            orders[robot].sort()
    return _make_state(assignment, orders, robots)


def _reinsert(assignment, orders, destroyed, robots, costs, rng):
    loads = {r: sum(costs[i][robots.index(r)] for i, x in enumerate(assignment) if x == r) for r in robots}
    for index in rng.permutation(destroyed).tolist():
        choices = sorted((loads[r] + costs[index][j], r, costs[index][j]) for j, r in enumerate(robots) if math.isfinite(costs[index][j]))
        if not choices:
            continue
        pick = int(rng.integers(0, min(2, len(choices))))
        _, robot, value = choices[pick]
        assignment[index] = robot
        loads[robot] += value
        position = int(rng.integers(0, len(orders[robot]) + 1))
        orders[robot].insert(position, index)


def _make_state(assignment, orders, robots):
    return InitializerState(tuple(assignment), tuple((r, tuple(orders[r])) for r in robots))


def _accept(current, candidate, iteration, rng):
    if candidate.verified and not current.verified:
        return True
    if candidate.verified and current.verified:
        return float(candidate.objective) <= float(current.objective) + 1e-12
    if not candidate.verified and current.verified:
        return False
    delta = candidate.surrogate - current.surrogate
    if delta <= 0:
        return True
    temperature = max(1.0, 1000.0 / math.sqrt(iteration))
    return bool(rng.random() < math.exp(-min(delta / temperature, 50.0)))


def _edit_distance(initial, final):
    changed = sum(a != b for a, b in zip(initial.assignments, final.assignments))
    initial_positions = {u: (r, i) for r, order in initial.robot_orders for i, u in enumerate(order)}
    final_positions = {u: (r, i) for r, order in final.robot_orders for i, u in enumerate(order)}
    order_changed = sum(initial_positions.get(u) != final_positions.get(u) for u in set(initial_positions) | set(final_positions))
    modified = len({i for i, (a, b) in enumerate(zip(initial.assignments, final.assignments)) if a != b} | {u for u in set(initial_positions) | set(final_positions) if initial_positions.get(u) != final_positions.get(u)})
    denominator = sum(x is not None for x in initial.assignments)
    retained = 1.0 if denominator == 0 else 1.0 - changed / denominator
    return changed, order_changed, modified, retained


def _evaluation_dict(value):
    return {"verified": value.verified, "objective": value.objective, "surrogate": value.surrogate, "failure_reason": value.failure_reason, "plan": None if value.plan is None else value.plan.to_dict()}
