"""Failure-aware atomic-unit destroy operators for A4b recovery v2."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from ..features import point_distance
from ..oracle import OracleContext
from ..repair import InitializerState
from ..schema import AllocationInstance
from .diagnostics import TimedEvaluation
from .operators import (
    DESTROY_OPERATORS,
    OperatorProblem,
    build_operator_problem,
    destroy_count,
    validate_destroy_set,
)


def _ranked(scores: Mapping[int, float], amount: int) -> list[int]:
    return [
        index
        for index, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
            :amount
        ]
    ]


def _failure_context(
    problem: OperatorProblem,
    state: InitializerState,
    evaluation: TimedEvaluation,
) -> tuple[set[int], set[int], set[int]]:
    failed = set(evaluation.diagnostic.failure_unit_indices)
    predecessors = {
        left for left, right in problem.predecessor_edges if right in failed
    }
    local = set()
    for _, order in state.robot_orders:
        for position, unit in enumerate(order):
            if unit not in failed:
                continue
            if position:
                local.add(order[position - 1])
            if position + 1 < len(order):
                local.add(order[position + 1])
    return failed, predecessors, local


def _worst_cost(problem: OperatorProblem, state: InitializerState) -> dict[int, float]:
    result = {}
    for index, robot in enumerate(state.assignments):
        if robot not in problem.robots:
            result[index] = 1e9
        else:
            result[index] = float(problem.costs[index][problem.robots.index(robot)])
    return result


def _load(problem: OperatorProblem, state: InitializerState) -> dict[int, float]:
    loads = {robot: 0.0 for robot in problem.robots}
    for index, robot in enumerate(state.assignments):
        if robot in loads:
            value = problem.costs[index][problem.robots.index(robot)]
            if math.isfinite(value):
                loads[robot] += value
    mean = sum(loads.values()) / max(len(loads), 1)
    return {
        index: max(0.0, loads.get(robot, 0.0) - mean)
        + (
            problem.costs[index][problem.robots.index(robot)]
            if robot in problem.robots
            else 1e9
        )
        for index, robot in enumerate(state.assignments)
    }


def _precedence(
    problem: OperatorProblem,
    state: InitializerState,
    evaluation: TimedEvaluation,
) -> dict[int, float]:
    failed, predecessors, local = _failure_context(problem, state, evaluation)
    scores = {index: 0.0 for index in range(len(problem.units))}
    for left, right in problem.predecessor_edges:
        scores[left] += 1.0
        scores[right] += 1.0
    for index in failed:
        scores[index] += 100.0
    for index in predecessors:
        scores[index] += 60.0
    for index in local:
        scores[index] += 30.0
    return scores


def _slack(
    problem: OperatorProblem,
    state: InitializerState,
    evaluation: TimedEvaluation,
) -> dict[int, float]:
    failed, predecessors, local = _failure_context(problem, state, evaluation)
    result = {}
    for index, window in enumerate(problem.unit_windows):
        width = max(window[1] - window[0], 1e-9)
        result[index] = 1.0 / width
        if index in failed:
            result[index] += 100.0
        if index in local:
            result[index] += 50.0
        if index in predecessors:
            result[index] += 25.0
    return result


def _resource(
    problem: OperatorProblem,
    state: InitializerState,
    evaluation: TimedEvaluation,
) -> dict[int, float]:
    failed, _, local = _failure_context(problem, state, evaluation)
    failed_resources = {
        resource for index in failed for resource in problem.unit_resources[index]
    }
    frequency: dict[str, int] = {}
    for resources in problem.unit_resources:
        for resource in resources:
            frequency[resource] = frequency.get(resource, 0) + 1
    result = {}
    for index, resources in enumerate(problem.unit_resources):
        result[index] = float(sum(frequency[item] for item in resources))
        if index in failed:
            result[index] += 100.0
        if resources & failed_resources:
            result[index] += 60.0
        if index in local:
            result[index] += 20.0
    return result


def _related(
    problem: OperatorProblem,
    state: InitializerState,
    evaluation: TimedEvaluation,
    amount: int,
    rng: np.random.Generator,
) -> list[int]:
    failed = evaluation.diagnostic.failure_unit_indices
    seed = int(failed[0]) if failed else int(rng.integers(0, len(problem.units)))
    scores = {}
    linked = {edge for edge in problem.predecessor_edges if seed in edge}
    for index in range(len(problem.units)):
        distance = point_distance(problem.unit_midpoints[seed], problem.unit_midpoints[index])
        window_distance = abs(
            problem.unit_windows[seed][0] - problem.unit_windows[index][0]
        )
        scores[index] = (
            1.0 / (1.0 + distance)
            + 2.0 * bool(problem.unit_resources[seed] & problem.unit_resources[index])
            + 1.0 * (state.assignments[index] == state.assignments[seed])
            + 1.5 * any(index in edge for edge in linked)
            + 1.0 / (1.0 + window_distance)
            + 100.0 * (index == seed)
        )
    return _ranked(scores, amount)


def _compound(
    problem: OperatorProblem,
    state: InitializerState,
    evaluation: TimedEvaluation,
    amount: int,
    rng: np.random.Generator,
) -> list[int]:
    score_maps = (
        _precedence(problem, state, evaluation),
        _slack(problem, state, evaluation),
        _resource(problem, state, evaluation),
    )
    combined = {index: 0.0 for index in range(len(problem.units))}
    for scores in score_maps:
        ranking = _ranked(scores, len(problem.units))
        for rank, index in enumerate(ranking):
            combined[index] += len(ranking) - rank
    related = _related(problem, state, evaluation, len(problem.units), rng)
    for rank, index in enumerate(related):
        combined[index] += len(related) - rank
    return _ranked(combined, amount)


def select_destroy_set_v2(
    operator: str,
    instance: AllocationInstance,
    context: OracleContext,
    state: InitializerState,
    evaluation: TimedEvaluation,
    ratio: float,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    if operator not in DESTROY_OPERATORS:
        raise ValueError(f"unknown destroy operator: {operator}")
    problem = build_operator_problem(instance, context)
    amount = destroy_count(len(problem.units), ratio)
    if operator == "random_destroy":
        selected = rng.choice(len(problem.units), amount, replace=False).tolist()
    elif operator == "worst_cost_destroy":
        selected = _ranked(_worst_cost(problem, state), amount)
    elif operator == "load_imbalance_destroy":
        selected = _ranked(_load(problem, state), amount)
    elif operator == "precedence_chain_destroy":
        selected = _ranked(_precedence(problem, state, evaluation), amount)
    elif operator == "critical_slack_destroy":
        selected = _ranked(_slack(problem, state, evaluation), amount)
    elif operator == "shared_resource_conflict_destroy":
        selected = _ranked(_resource(problem, state, evaluation), amount)
    elif operator == "relatedness_destroy":
        selected = _related(problem, state, evaluation, amount, rng)
    else:
        selected = _compound(problem, state, evaluation, amount, rng)
    return validate_destroy_set(selected, len(problem.units), amount)

