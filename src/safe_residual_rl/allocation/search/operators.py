"""Feasibility-aware, atomic-unit destroy operators for ordinary A4b LNS."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..features import point_distance
from ..oracle import OracleContext
from ..repair import InitializerState
from ..repair.identical import StateEvaluation
from ..schema import AllocationInstance
from ..solvers.common import allocation_units, edge_mask_and_costs

DESTROY_OPERATORS = (
    "random_destroy",
    "worst_cost_destroy",
    "load_imbalance_destroy",
    "precedence_chain_destroy",
    "critical_slack_destroy",
    "shared_resource_conflict_destroy",
    "relatedness_destroy",
    "compound_destroy",
)
HANDCRAFTED_OPERATORS = DESTROY_OPERATORS[1:]


@dataclass(frozen=True)
class OperatorProblem:
    units: tuple[tuple[str, ...], ...]
    robots: tuple[str, ...]
    costs: tuple[tuple[float, ...], ...]
    unit_midpoints: tuple[tuple[float, float, float], ...]
    unit_windows: tuple[tuple[float, float], ...]
    unit_resources: tuple[frozenset[str], ...]
    predecessor_edges: frozenset[tuple[int, int]]


def build_operator_problem(
    instance: AllocationInstance, context: OracleContext
) -> OperatorProblem:
    _, units, robots, costs = edge_mask_and_costs(instance, context)
    segment_by_id = {item.id: item for item in instance.segments}
    segment_to_unit = {segment: index for index, unit in enumerate(units) for segment in unit}
    midpoints = []
    windows = []
    resources = []
    edges: set[tuple[int, int]] = set()
    for index, unit in enumerate(units):
        points = [point for segment in unit for point in segment_by_id[segment].sampled_curve_m]
        midpoints.append(tuple(sum(point[axis] for point in points) / len(points) for axis in range(3)))
        windows.append(
            (
                min(segment_by_id[segment].time_window.start_s for segment in unit),
                max(segment_by_id[segment].time_window.end_s for segment in unit),
            )
        )
        resources.append(
            frozenset(resource for segment in unit for resource in segment_by_id[segment].shared_resource_ids)
        )
        for segment in unit:
            for predecessor in segment_by_id[segment].predecessor_ids:
                left = segment_to_unit[predecessor]
                if left != index:
                    edges.add((left, index))
    return OperatorProblem(
        units,
        robots,
        costs,
        tuple(midpoints),
        tuple(windows),
        tuple(resources),
        frozenset(edges),
    )


def destroy_count(unit_count: int, ratio: float) -> int:
    if unit_count < 1 or not 0 < ratio <= 1:
        raise ValueError("destroy ratio and unit count must be positive")
    return min(unit_count, max(1, int(math.ceil(unit_count * ratio))))


def validate_destroy_set(indices: Sequence[int], unit_count: int, expected: int) -> tuple[int, ...]:
    result = tuple(int(item) for item in indices)
    if len(result) != expected or len(set(result)) != len(result):
        raise ValueError("destroy set contains duplicates or has the wrong size")
    if any(item < 0 or item >= unit_count for item in result):
        raise ValueError("destroy set contains a non-atomic-unit index")
    return result


def select_destroy_set(
    operator: str,
    instance: AllocationInstance,
    context: OracleContext,
    state: InitializerState,
    evaluation: StateEvaluation,
    ratio: float,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Select whole atomic units without duplicates under one fixed ratio."""
    if operator not in DESTROY_OPERATORS:
        raise ValueError(f"unknown destroy operator: {operator}")
    problem = build_operator_problem(instance, context)
    amount = destroy_count(len(problem.units), ratio)
    if operator == "random_destroy":
        selected = rng.choice(len(problem.units), amount, replace=False).tolist()
    elif operator == "worst_cost_destroy":
        selected = _ranked(_worst_cost(problem, state, evaluation), amount)
    elif operator == "load_imbalance_destroy":
        selected = _ranked(_load_imbalance(problem, state), amount)
    elif operator == "precedence_chain_destroy":
        selected = _ranked(_precedence(problem, evaluation), amount)
    elif operator == "critical_slack_destroy":
        selected = _ranked(_slack(problem, evaluation), amount)
    elif operator == "shared_resource_conflict_destroy":
        selected = _ranked(_resource(problem, evaluation), amount)
    elif operator == "relatedness_destroy":
        selected = _related(problem, state, amount, rng)
    else:
        selected = _compound(problem, state, evaluation, amount, rng)
    return validate_destroy_set(selected, len(problem.units), amount)


def _ranked(scores: Mapping[int, float], amount: int) -> list[int]:
    return [index for index, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:amount]]


def _worst_cost(problem: OperatorProblem, state: InitializerState, evaluation: StateEvaluation):
    schedule = {} if evaluation.plan is None else {item.segment_id: item for item in evaluation.plan.schedule}
    result = {}
    for index, unit in enumerate(problem.units):
        robot = state.assignments[index]
        base = 1e9 if robot not in problem.robots else problem.costs[index][problem.robots.index(robot)]
        completion = max((schedule[item].planned_end_s for item in unit if item in schedule), default=0.0)
        result[index] = float(base) + completion * 1e-3
    return result


def _load_imbalance(problem: OperatorProblem, state: InitializerState):
    loads = {robot: 0.0 for robot in problem.robots}
    for index, robot in enumerate(state.assignments):
        if robot in loads:
            loads[robot] += problem.costs[index][problem.robots.index(robot)]
    mean = sum(loads.values()) / len(loads)
    return {
        index: max(0.0, loads.get(robot, 0.0) - mean) + (
            problem.costs[index][problem.robots.index(robot)] if robot in problem.robots else 1e9
        )
        for index, robot in enumerate(state.assignments)
    }


def _precedence(problem: OperatorProblem, evaluation: StateEvaluation):
    schedule = {} if evaluation.plan is None else {item.segment_id: item for item in evaluation.plan.schedule}
    scores = {index: 0.0 for index in range(len(problem.units))}
    for left, right in problem.predecessor_edges:
        gap = 0.0
        if schedule:
            left_end = max((schedule[item].planned_end_s for item in problem.units[left] if item in schedule), default=0.0)
            right_start = min((schedule[item].planned_start_s for item in problem.units[right] if item in schedule), default=left_end)
            gap = max(0.0, right_start - left_end)
        critical = 1.0 / (1.0 + gap)
        scores[left] += 1.0 + critical
        scores[right] += 1.0 + critical
    return scores


def _slack(problem: OperatorProblem, evaluation: StateEvaluation):
    schedule = {} if evaluation.plan is None else {item.segment_id: item for item in evaluation.plan.schedule}
    result = {}
    for index, unit in enumerate(problem.units):
        window = problem.unit_windows[index]
        if schedule and all(item in schedule for item in unit):
            end = max(schedule[item].planned_end_s for item in unit)
            slack = window[1] - end
        else:
            slack = window[1] - window[0]
        result[index] = 1.0 / max(slack + 1e-9, 1e-9)
    return result


def _resource(problem: OperatorProblem, evaluation: StateEvaluation):
    frequency: dict[str, int] = {}
    for resources in problem.unit_resources:
        for resource in resources:
            frequency[resource] = frequency.get(resource, 0) + 1
    schedule = {} if evaluation.plan is None else {item.segment_id: item for item in evaluation.plan.schedule}
    result = {}
    for index, resources in enumerate(problem.unit_resources):
        occupancy = sum(frequency[item] for item in resources)
        duration = 0.0
        if schedule:
            duration = sum(
                schedule[item].planned_end_s - schedule[item].planned_start_s
                for item in problem.units[index]
                if item in schedule
            )
        result[index] = float(occupancy) + duration * 1e-3
    return result


def _related(
    problem: OperatorProblem,
    state: InitializerState,
    amount: int,
    rng: np.random.Generator,
) -> list[int]:
    seed = int(rng.integers(0, len(problem.units)))
    seed_midpoint = problem.unit_midpoints[seed]
    seed_resources = problem.unit_resources[seed]
    seed_robot = state.assignments[seed]
    linked = {edge for edge in problem.predecessor_edges if seed in edge}
    scores = {}
    for index in range(len(problem.units)):
        distance = point_distance(seed_midpoint, problem.unit_midpoints[index])
        window_distance = abs(problem.unit_windows[seed][0] - problem.unit_windows[index][0])
        score = 1.0 / (1.0 + distance)
        score += 2.0 * bool(seed_resources & problem.unit_resources[index])
        score += 1.0 * (state.assignments[index] == seed_robot)
        score += 1.5 * any(index in edge for edge in linked)
        score += 1.0 / (1.0 + window_distance)
        scores[index] = score
    return _ranked(scores, amount)


def _compound(
    problem: OperatorProblem,
    state: InitializerState,
    evaluation: StateEvaluation,
    amount: int,
    rng: np.random.Generator,
) -> list[int]:
    rankings = [
        _ranked(_precedence(problem, evaluation), len(problem.units)),
        _ranked(_slack(problem, evaluation), len(problem.units)),
        _ranked(_resource(problem, evaluation), len(problem.units)),
        _related(problem, state, len(problem.units), rng),
    ]
    selected: list[int] = []
    cursor = 0
    while len(selected) < amount:
        ranking = rankings[cursor % len(rankings)]
        index = ranking[(cursor // len(rankings)) % len(ranking)]
        if index not in selected:
            selected.append(index)
        cursor += 1
    return selected
