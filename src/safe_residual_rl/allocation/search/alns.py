"""Unified ordinary LNS/ALNS engine for the A4b development protocol."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from ..oracle import OracleContext
from ..repair import InitializerState, evaluate_state
from ..repair.identical import StateEvaluation, canonicalize_state
from ..schema import AllocationInstance
from ..solvers.common import allocation_units, edge_mask_and_costs
from .anytime import InitializerOutcome
from .operators import (
    DESTROY_OPERATORS,
    HANDCRAFTED_OPERATORS,
    build_operator_problem,
    select_destroy_set,
)
from .trace import (
    IncumbentEvent,
    SearchStep,
    SearchTrace,
    canonical_hash,
    state_hash,
    violation_score,
)


@dataclass(frozen=True)
class AlnsConfig:
    protocol_id: str
    iterations: int
    maximum_end_to_end_time_s: float
    destroy_ratios: tuple[float, ...]
    repair_candidate_evaluation_budget: int
    random_seed: int
    objective_weights: Mapping[str, float]
    acceptance_rule: str = "verified_first_simulated_annealing_v1"
    initial_temperature_fraction: float = 0.05
    cooling_rate: float = 0.97
    reaction_factor: float = 0.20
    reward_global_best: float = 8.0
    reward_improving_accepted: float = 4.0
    reward_new_feasible: float = 2.0
    reward_accepted: float = 1.0
    reward_rejected: float = 0.0
    restart_no_improvement: int = 12
    allow_infeasible_current: bool = True

    def validate(self) -> None:
        if self.iterations < 1 or self.maximum_end_to_end_time_s <= 0:
            raise ValueError("search budgets must be positive")
        if not self.destroy_ratios or any(not 0 < item <= 1 for item in self.destroy_ratios):
            raise ValueError("destroy ratios must be in (0, 1]")
        if self.repair_candidate_evaluation_budget < 1:
            raise ValueError("repair candidate budget must be positive")
        if not 0 < self.reaction_factor <= 1 or not 0 < self.cooling_rate <= 1:
            raise ValueError("invalid ALNS update parameters")

    @property
    def sha256(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class RepairOutcome:
    state: InitializerState
    evaluation: StateEvaluation
    runtime_s: float
    candidate_evaluations: int
    budget_exhausted: bool
    edit_distance: int


@dataclass(frozen=True)
class SearchOutcome:
    trace: SearchTrace
    final_state: InitializerState
    final_evaluation: StateEvaluation
    best_state: InitializerState | None
    best_evaluation: StateEvaluation | None
    operator_weights: tuple[tuple[str, float], ...]


def _evaluation_hash(value: StateEvaluation) -> str:
    plan = None if value.plan is None else value.plan.to_dict()
    return canonical_hash(
        {
            "verified": value.verified,
            "objective": value.objective,
            "surrogate": value.surrogate,
            "failure_reason": value.failure_reason,
            "plan": plan,
        }
    )


def _plan_hash(value: StateEvaluation) -> str | None:
    return None if value.plan is None else canonical_hash(value.plan.to_dict())


def repair_destroyed_state(
    instance: AllocationInstance,
    context: OracleContext,
    state: InitializerState,
    destroy_set: Sequence[int],
    *,
    random_seed: int,
    weights: Mapping[str, float],
    candidate_evaluation_budget: int,
    clock: Callable[[], int] = time.monotonic_ns,
) -> RepairOutcome:
    """One unchanged regret/load reinsertion backend used by every A4b arm."""
    started = clock()
    rng = np.random.default_rng(random_seed)
    _, units, robots, costs = edge_mask_and_costs(instance, context)
    destroyed = tuple(int(item) for item in destroy_set)
    if len(set(destroyed)) != len(destroyed) or any(item < 0 or item >= len(units) for item in destroyed):
        raise ValueError("repair received a non-atomic or duplicate destroy set")

    assignment = list(state.assignments)
    orders = state.order_map()
    for index in destroyed:
        assignment[index] = None
        for order in orders.values():
            if index in order:
                order.remove(index)
    remaining = list(destroyed)
    evaluations = 0
    exhausted = False

    while remaining:
        best_per_unit: list[tuple[float, float, int, InitializerState, StateEvaluation]] = []
        for unit_index in remaining:
            choices: list[tuple[float, InitializerState, StateEvaluation]] = []
            robot_candidates = [
                robot for robot, cost in zip(robots, costs[unit_index]) if math.isfinite(cost)
            ]
            for robot in robot_candidates:
                positions = _candidate_positions(len(orders[robot]))
                for position in positions:
                    if evaluations >= candidate_evaluation_budget:
                        exhausted = True
                        break
                    trial_assignment = list(assignment)
                    trial_assignment[unit_index] = robot
                    trial_orders = {key: list(value) for key, value in orders.items()}
                    trial_orders[robot].insert(position, unit_index)
                    trial = canonicalize_state(
                        instance,
                        InitializerState(
                            tuple(trial_assignment),
                            tuple((key, tuple(trial_orders[key])) for key in robots),
                        ),
                    )
                    evaluated = evaluate_state(instance, context, trial, weights, method_id="a4b-shared-repair-v1")
                    evaluations += 1
                    score = float(evaluated.objective) if evaluated.verified else evaluated.surrogate
                    choices.append((score, trial, evaluated))
                if exhausted:
                    break
            if choices:
                choices.sort(key=lambda item: (not item[2].verified, item[0], state_hash(item[1])))
                first = choices[0]
                second_score = choices[1][0] if len(choices) > 1 else first[0]
                regret = second_score - first[0]
                best_per_unit.append((-regret, first[0], unit_index, first[1], first[2]))
            if exhausted:
                break
        if not best_per_unit:
            # Deterministic budget fallback remains the same for every method.
            unit_index = min(remaining)
            robot = min(
                (cost, robot) for robot, cost in zip(robots, costs[unit_index]) if math.isfinite(cost)
            )[1]
            assignment[unit_index] = robot
            orders[robot].append(unit_index)
            remaining.remove(unit_index)
            continue
        _, _, selected, trial, _ = min(best_per_unit)
        assignment = list(trial.assignments)
        orders = trial.order_map()
        remaining.remove(selected)

    final_state = canonicalize_state(
        instance,
        InitializerState(
            tuple(assignment), tuple((robot, tuple(orders[robot])) for robot in robots)
        ),
    )
    final_eval = evaluate_state(instance, context, final_state, weights, method_id="a4b-shared-repair-v1")
    runtime = (clock() - started) / 1e9
    edits = sum(left != right for left, right in zip(state.assignments, final_state.assignments))
    return RepairOutcome(final_state, final_eval, runtime, evaluations, exhausted, edits)


def _candidate_positions(length: int) -> tuple[int, ...]:
    values = {0, length}
    if length:
        values.add(length // 2)
    return tuple(sorted(values))


def update_operator_weight(weight: float, reward: float, reaction_factor: float) -> float:
    if weight <= 0 or reward < 0 or not 0 < reaction_factor <= 1:
        raise ValueError("invalid ALNS weight update")
    return (1.0 - reaction_factor) * weight + reaction_factor * max(reward, 1e-6)


def _select_operator(
    mode: str,
    iteration: int,
    rng: np.random.Generator,
    weights: Mapping[str, float],
    single_operator: str | None,
) -> str:
    if mode == "random_lns":
        return "random_destroy"
    if mode == "handcrafted_round_robin":
        return HANDCRAFTED_OPERATORS[(iteration - 1) % len(HANDCRAFTED_OPERATORS)]
    if mode == "single_operator":
        if single_operator not in HANDCRAFTED_OPERATORS:
            raise ValueError("single handcrafted operator is not registered")
        return str(single_operator)
    if mode == "adaptive_alns":
        operators = tuple(weights)
        probabilities = np.asarray([weights[item] for item in operators], dtype=float)
        probabilities /= probabilities.sum()
        return str(rng.choice(operators, p=probabilities))
    if mode == "oracle_destroy":
        return "oracle_destroy"
    raise ValueError(f"unknown search mode: {mode}")


def _accept(
    current: StateEvaluation,
    candidate: StateEvaluation,
    iteration: int,
    rng: np.random.Generator,
    config: AlnsConfig,
) -> bool:
    if candidate.verified and not current.verified:
        return True
    if current.verified and not candidate.verified:
        return False
    current_value = float(current.objective) if current.verified else current.surrogate
    candidate_value = float(candidate.objective) if candidate.verified else candidate.surrogate
    delta = candidate_value - current_value
    if delta <= 1e-12:
        return True
    scale = max(abs(current_value), 1.0)
    temperature = config.initial_temperature_fraction * scale * (config.cooling_rate ** (iteration - 1))
    return bool(rng.random() < math.exp(-min(delta / max(temperature, 1e-12), 50.0)))


def _reward(
    current_before: StateEvaluation,
    candidate: StateEvaluation,
    accepted: bool,
    global_best: bool,
    config: AlnsConfig,
) -> float:
    if global_best:
        return config.reward_global_best
    if accepted and candidate.verified and not current_before.verified:
        return config.reward_new_feasible
    if accepted:
        before = float(current_before.objective) if current_before.verified else current_before.surrogate
        after = float(candidate.objective) if candidate.verified else candidate.surrogate
        if after + 1e-12 < before:
            return config.reward_improving_accepted
        return config.reward_accepted
    return config.reward_rejected


def _incumbent_event(
    iteration: int,
    timestamp_ns: int,
    start_ns: int,
    state: InitializerState,
    evaluation: StateEvaluation,
) -> IncumbentEvent:
    assert evaluation.verified and evaluation.objective is not None and evaluation.plan is not None
    checked = verify_plan_placeholder(evaluation)
    return IncumbentEvent(
        iteration,
        timestamp_ns,
        (timestamp_ns - start_ns) / 1e9,
        float(evaluation.objective),
        state,
        state_hash(state),
        canonical_hash(evaluation.plan.to_dict()),
        checked,
    )


def verify_plan_placeholder(evaluation: StateEvaluation) -> str:
    """Hash the independent verifier-facing result already stored by evaluate_state."""
    return canonical_hash(
        {
            "verified": evaluation.verified,
            "failure_reason": evaluation.failure_reason,
            "objective": evaluation.objective,
            "plan": None if evaluation.plan is None else evaluation.plan.to_dict(),
        }
    )


def _static_features(instance: AllocationInstance, context: OracleContext):
    problem = build_operator_problem(instance, context)
    segment_by_id = {item.id: item for item in instance.segments}
    result = []
    for index, unit in enumerate(problem.units):
        result.append(
            {
                "unit_index": index,
                "segment_ids": list(unit),
                "atomic": True,
                "length_m": sum(segment_by_id[item].length_m for item in unit),
                "process_duration_s": sum(segment_by_id[item].process_duration_s for item in unit),
                "priority_max": max(segment_by_id[item].priority for item in unit),
                "time_window": list(problem.unit_windows[index]),
                "shared_resources": sorted(problem.unit_resources[index]),
                "predecessor_units": sorted(left for left, right in problem.predecessor_edges if right == index),
                "eligible_robots": [
                    robot for robot, cost in zip(problem.robots, problem.costs[index]) if math.isfinite(cost)
                ],
                "midpoint_m": list(problem.unit_midpoints[index]),
            }
        )
    return tuple(result)


def _dynamic_features(
    instance: AllocationInstance,
    state: InitializerState,
    evaluation: StateEvaluation,
) -> dict[str, object]:
    units = allocation_units(instance)
    segment_by_id = {item.id: item for item in instance.segments}
    schedule = {} if evaluation.plan is None else {item.segment_id: item for item in evaluation.plan.schedule}
    robot_state = []
    for robot, order in state.robot_orders:
        segments = [segment for index in order for segment in units[index]]
        completion = max((schedule[item].planned_end_s for item in segments if item in schedule), default=0.0)
        last = segments[-1] if segments else None
        location = None if last is None else list(segment_by_id[last].end_pose.position_m)
        robot_state.append(
            {
                "robot_id": robot,
                "assigned_units": list(order),
                "completion_proxy_s": completion,
                "last_location_m": location,
                "process_load_s": sum(segment_by_id[item].process_duration_s for item in segments),
            }
        )
    unit_state = []
    for index, unit in enumerate(units):
        predecessor_ids = {item for segment in unit for item in segment_by_id[segment].predecessor_ids}
        satisfied = all(item in schedule for item in predecessor_ids)
        end = max((schedule[item].planned_end_s for item in unit if item in schedule), default=None)
        slack = None if end is None else max(segment_by_id[item].time_window.end_s for item in unit) - end
        unit_state.append(
            {
                "unit_index": index,
                "robot_id": state.assignments[index],
                "precedence_satisfied": satisfied,
                "precedence_critical": bool(predecessor_ids) and (slack is None or slack <= 1.0),
                "time_window_slack_s": slack,
                "shared_resource_occupancy_proxy": sum(len(segment_by_id[item].shared_resource_ids) for item in unit),
            }
        )
    return {
        "current_assignment": list(state.assignments),
        "robot_local_orders": [[robot, list(order)] for robot, order in state.robot_orders],
        "verifier_feasible": evaluation.verified,
        "failure_reason": evaluation.failure_reason,
        "robots": robot_state,
        "units": unit_state,
    }


def run_search(
    instance: AllocationInstance,
    context: OracleContext,
    initializer: InitializerOutcome,
    config: AlnsConfig,
    *,
    mode: str,
    task_group_id: str,
    difficulty: str,
    split: str,
    single_operator: str | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> SearchOutcome:
    config.validate()
    start_ns = initializer.provenance.start_monotonic_ns
    if initializer.provenance.completion_monotonic_ns < start_ns:
        raise ValueError("initializer provenance is not monotonic")
    current_state = canonicalize_state(instance, initializer.state)
    current_eval = evaluate_state(instance, context, current_state, config.objective_weights)
    best_state = current_state if current_eval.verified else None
    best_eval = current_eval if current_eval.verified else None
    incumbents: list[IncumbentEvent] = []
    if current_eval.verified:
        incumbents.append(
            _incumbent_event(
                0,
                initializer.provenance.completion_monotonic_ns,
                start_ns,
                current_state,
                current_eval,
            )
        )
    weights = {operator: 1.0 for operator in DESTROY_OPERATORS}
    steps: list[SearchStep] = []
    no_improvement = 0
    termination = "iteration_budget"

    for iteration in range(1, config.iterations + 1):
        now = max(clock(), initializer.provenance.completion_monotonic_ns)
        if (now - start_ns) / 1e9 >= config.maximum_end_to_end_time_s:
            termination = "end_to_end_time_budget"
            break
        iteration_seed = int(
            np.random.SeedSequence([config.random_seed, iteration]).generate_state(1)[0]
        )
        rng = np.random.default_rng(iteration_seed)
        operator = _select_operator(mode, iteration, rng, weights, single_operator)
        ratio = config.destroy_ratios[(iteration - 1) % len(config.destroy_ratios)]
        weight_before = 1.0 if operator == "oracle_destroy" else weights[operator]
        before_state, before_eval = current_state, current_eval

        if operator == "oracle_destroy":
            candidates = []
            for op_index, candidate_operator in enumerate(DESTROY_OPERATORS):
                local_seed = int(np.random.SeedSequence([iteration_seed, op_index]).generate_state(1)[0])
                local_rng = np.random.default_rng(local_seed)
                destroyed = select_destroy_set(
                    candidate_operator, instance, context, before_state, before_eval, ratio, local_rng
                )
                repaired = repair_destroyed_state(
                    instance,
                    context,
                    before_state,
                    destroyed,
                    random_seed=local_seed,
                    weights=config.objective_weights,
                    candidate_evaluation_budget=config.repair_candidate_evaluation_budget,
                    clock=clock,
                )
                rank = (
                    not repaired.evaluation.verified,
                    float(repaired.evaluation.objective) if repaired.evaluation.verified else repaired.evaluation.surrogate,
                    candidate_operator,
                )
                candidates.append((rank, candidate_operator, destroyed, repaired))
            _, chosen_operator, destroyed, repaired = min(candidates, key=lambda item: item[0])
            recorded_operator = f"oracle_destroy:{chosen_operator}"
            candidate_evaluations = sum(item[3].candidate_evaluations for item in candidates)
            repair_runtime = sum(item[3].runtime_s for item in candidates)
        else:
            destroyed = select_destroy_set(operator, instance, context, before_state, before_eval, ratio, rng)
            repaired = repair_destroyed_state(
                instance,
                context,
                before_state,
                destroyed,
                random_seed=iteration_seed,
                weights=config.objective_weights,
                candidate_evaluation_budget=config.repair_candidate_evaluation_budget,
                clock=clock,
            )
            recorded_operator = operator
            candidate_evaluations = repaired.candidate_evaluations
            repair_runtime = repaired.runtime_s

        candidate_state, candidate_eval = repaired.state, repaired.evaluation
        candidate_ns = clock()
        accepted = _accept(before_eval, candidate_eval, iteration, rng, config)
        global_best = candidate_eval.verified and (
            best_eval is None or float(candidate_eval.objective) + 1e-12 < float(best_eval.objective)
        )
        if global_best:
            best_state, best_eval = candidate_state, candidate_eval
            incumbents.append(_incumbent_event(iteration, candidate_ns, start_ns, candidate_state, candidate_eval))
            no_improvement = 0
        else:
            no_improvement += 1
        if accepted:
            current_state, current_eval = candidate_state, candidate_eval
        reward = _reward(before_eval, candidate_eval, accepted, global_best, config)
        if operator != "oracle_destroy" and mode == "adaptive_alns":
            weights[operator] = update_operator_weight(weights[operator], reward, config.reaction_factor)
        weight_after = weight_before if operator == "oracle_destroy" else weights[operator]

        before_violation = violation_score(before_eval.verified, before_eval.failure_reason, before_eval.surrogate)
        after_violation = violation_score(candidate_eval.verified, candidate_eval.failure_reason, candidate_eval.surrogate)
        delta = None
        if before_eval.objective is not None and candidate_eval.objective is not None:
            delta = float(candidate_eval.objective) - float(before_eval.objective)
        steps.append(
            SearchStep(
                iteration=iteration,
                iteration_seed=iteration_seed,
                operator=recorded_operator,
                destroy_ratio=ratio,
                destroy_set=tuple(destroyed),
                before_state=before_state,
                candidate_state=candidate_state,
                before_verified=before_eval.verified,
                after_verified=candidate_eval.verified,
                before_failure_reason=before_eval.failure_reason,
                after_failure_reason=candidate_eval.failure_reason,
                current_objective=before_eval.objective,
                best_so_far_objective=None if best_eval is None else best_eval.objective,
                objective_delta=delta,
                violation_reduction=before_violation - after_violation,
                repair_runtime_s=repair_runtime,
                candidate_evaluations=candidate_evaluations,
                accepted=accepted,
                incumbent_updated=global_best,
                reward=reward,
                operator_weight_before=weight_before,
                operator_weight_after=weight_after,
                monotonic_ns=candidate_ns,
                elapsed_s=(candidate_ns - start_ns) / 1e9,
                before_state_sha256=state_hash(before_state),
                candidate_state_sha256=state_hash(candidate_state),
                plan_sha256=_plan_hash(candidate_eval),
                verifier_sha256=_evaluation_hash(candidate_eval),
                dynamic_features=_dynamic_features(instance, before_state, before_eval),
            )
        )
        if no_improvement >= config.restart_no_improvement:
            if best_state is not None and best_eval is not None:
                current_state, current_eval = best_state, best_eval
            else:
                current_state = initializer.state
                current_eval = evaluate_state(instance, context, current_state, config.objective_weights)
            no_improvement = 0

    returned = max(clock(), initializer.provenance.completion_monotonic_ns)
    trace = SearchTrace(
        version="a4b-search-trace-v1",
        protocol_id=config.protocol_id,
        instance_id=instance.instance_id,
        task_group_id=task_group_id,
        difficulty=difficulty,
        split=split,
        method=mode if single_operator is None else f"{mode}:{single_operator}",
        random_seed=config.random_seed,
        clock="time.monotonic_ns",
        start_monotonic_ns=start_ns,
        return_monotonic_ns=returned,
        initializer=initializer.provenance.to_dict(),
        initial_state=initializer.state,
        static_unit_features=_static_features(instance, context),
        config_sha256=config.sha256,
        steps=tuple(steps),
        incumbents=tuple(incumbents),
        termination_reason=termination,
    )
    return SearchOutcome(
        trace,
        current_state,
        current_eval,
        best_state,
        best_eval,
        tuple(sorted(weights.items())),
    )
