"""Deadline-safe ordinary LNS/ALNS engine for A4b recovery v2."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from ..oracle import OracleContext
from ..repair import InitializerState
from ..repair.identical import canonicalize_state
from ..schema import AllocationInstance
from ..solvers.common import allocation_units, edge_mask_and_costs
from .anytime import InitializerOutcome
from .diagnostics import TimedEvaluation, ViolationVector, analyze_state, evaluate_state_timed
from .operators import DESTROY_OPERATORS, HANDCRAFTED_OPERATORS
from .operators_v2 import select_destroy_set_v2
from .trace import canonical_hash, state_hash, state_to_dict


@dataclass(frozen=True)
class AlnsV2Config:
    protocol_id: str
    budget_mode: str
    iterations: int
    end_to_end_time_s: float | None
    safety_watchdog_s: float
    destroy_ratios: tuple[float, ...]
    repair_candidate_evaluation_budget: int
    random_seed: int
    objective_weights: Mapping[str, float]
    update_scheme: str = "segmented"
    segment_length: int = 8
    reaction_factor: float = 0.20
    initial_temperature_fraction: float = 0.02
    cooling_rate: float = 0.97
    reward_global_best: float = 8.0
    reward_new_feasible: float = 6.0
    reward_strict_improvement: float = 4.0
    reward_unseen_diversification: float = 1.0
    restart_no_improvement: int = 12
    allow_infeasible_current: bool = True

    def validate(self) -> None:
        if self.budget_mode not in {"fixed_time", "fixed_iterations"}:
            raise ValueError("unknown v2 budget mode")
        if self.budget_mode == "fixed_time" and (
            self.end_to_end_time_s is None or self.end_to_end_time_s <= 0
        ):
            raise ValueError("fixed-time mode requires a positive deadline")
        if self.iterations < 1 or self.safety_watchdog_s <= 0:
            raise ValueError("invalid search limits")
        if not self.destroy_ratios or any(not 0 < item <= 1 for item in self.destroy_ratios):
            raise ValueError("invalid destroy ratios")
        if self.repair_candidate_evaluation_budget < 1:
            raise ValueError("repair candidate budget must be positive")
        if self.update_scheme not in {"online", "segmented"}:
            raise ValueError("unknown ALNS update scheme")
        if self.segment_length < 1 or not 0 < self.reaction_factor <= 1:
            raise ValueError("invalid ALNS adaptation parameters")

    @property
    def sha256(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class RepairV2Outcome:
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


@dataclass(frozen=True)
class SearchV2Outcome:
    trace: dict[str, object]
    final_state: InitializerState
    final_evaluation: TimedEvaluation
    best_state: InitializerState | None
    best_evaluation: TimedEvaluation | None
    operator_weights: tuple[tuple[str, float], ...]


def update_segmented_weights(
    weights: Mapping[str, float],
    scores: Mapping[str, float],
    uses: Mapping[str, int],
    reaction_factor: float,
) -> dict[str, float]:
    if not 0 < reaction_factor <= 1:
        raise ValueError("invalid reaction factor")
    result = {}
    for operator, weight in weights.items():
        if weight <= 0:
            raise ValueError("operator weights must remain positive")
        usage = int(uses.get(operator, 0))
        observed = float(scores.get(operator, 0.0)) / usage if usage else weight
        result[operator] = max(
            1e-6, (1.0 - reaction_factor) * weight + reaction_factor * observed
        )
    return result


def _online_weight(weight: float, reward: float, reaction: float) -> float:
    return max(1e-6, (1.0 - reaction) * weight + reaction * reward)


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


def _fallback_insert(
    instance: AllocationInstance,
    context: OracleContext,
    assignments: list[str | None],
    orders: dict[str, list[int]],
    unit_index: int,
    robots: tuple[str, ...],
    costs: tuple[tuple[float, ...], ...],
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
            vector = analyze_state(instance, context, trial).vector
            key = (vector.rank(), edge_cost, robot, position, state_hash(trial))
            if best is None or key < best[0]:
                best = (key, trial)
    if best is None:
        raise RuntimeError("no edge-feasible fallback insertion")
    trial = best[1]
    return list(trial.assignments), trial.order_map()


def repair_destroyed_state_v2(
    instance: AllocationInstance,
    context: OracleContext,
    state: InitializerState,
    destroy_set: Sequence[int],
    *,
    weights: Mapping[str, float],
    candidate_evaluation_budget: int,
    deadline_ns: int | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> RepairV2Outcome:
    started = clock()
    _, units, robots, costs = edge_mask_and_costs(instance, context)
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
            assignments, orders = _fallback_insert(
                instance, context, assignments, orders, selected, robots, costs
            )
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
                    trial = _make_state(
                        instance, trial_assignments, trial_orders, robots
                    )
                    vector = analyze_state(instance, context, trial).vector
                    evaluations += 1
                    choices.append(
                        (
                            vector.rank(),
                            vector.scalar(),
                            state_hash(trial),
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
    final = evaluate_state_timed(instance, context, final_state, weights, clock=clock)
    returned = clock()
    assignment_edits, order_edits, total = _edit_counts(state, final_state)
    first_feasible = (returned - started) / 1e9 if final.evaluation.verified else None
    return RepairV2Outcome(
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
    )


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
            raise ValueError("single operator is not registered")
        return str(single_operator)
    if mode == "adaptive_alns":
        operators = tuple(weights)
        probabilities = np.asarray([weights[item] for item in operators], dtype=float)
        probabilities /= probabilities.sum()
        return str(rng.choice(operators, p=probabilities))
    raise ValueError(f"unknown v2 search mode: {mode}")


def _accept(
    current: TimedEvaluation,
    candidate: TimedEvaluation,
    iteration: int,
    rng: np.random.Generator,
    config: AlnsV2Config,
) -> bool:
    if candidate.evaluation.verified and not current.evaluation.verified:
        return True
    if current.evaluation.verified and not candidate.evaluation.verified:
        return False
    if not candidate.evaluation.verified and not config.allow_infeasible_current:
        return False
    if candidate.evaluation.verified:
        current_value = float(current.evaluation.objective)
        candidate_value = float(candidate.evaluation.objective)
    else:
        current_value = current.diagnostic.vector.scalar()
        candidate_value = candidate.diagnostic.vector.scalar()
    delta = candidate_value - current_value
    if delta <= 1e-12:
        return True
    temperature = (
        config.initial_temperature_fraction
        * max(abs(current_value), 1.0)
        * (config.cooling_rate ** (iteration - 1))
    )
    return bool(rng.random() < math.exp(-min(delta / max(temperature, 1e-12), 50.0)))


def _strict_improvement(before: TimedEvaluation, after: TimedEvaluation) -> bool:
    if after.evaluation.verified and not before.evaluation.verified:
        return True
    if after.evaluation.verified and before.evaluation.verified:
        return float(after.evaluation.objective) + 1e-12 < float(before.evaluation.objective)
    if not after.evaluation.verified and not before.evaluation.verified:
        return after.diagnostic.vector.rank() < before.diagnostic.vector.rank()
    return False


def _reward(
    before: TimedEvaluation,
    candidate: TimedEvaluation,
    *,
    accepted: bool,
    global_best: bool,
    unseen: bool,
    config: AlnsV2Config,
) -> float:
    if global_best:
        return config.reward_global_best
    if accepted and candidate.evaluation.verified and not before.evaluation.verified:
        return config.reward_new_feasible
    if accepted and _strict_improvement(before, candidate):
        return config.reward_strict_improvement
    if accepted and unseen:
        return config.reward_unseen_diversification
    return 0.0


def _event(
    iteration: int,
    timestamp_ns: int,
    start_ns: int,
    state: InitializerState,
    evaluation: TimedEvaluation,
) -> dict[str, object]:
    assert evaluation.evaluation.verified
    plan = evaluation.evaluation.plan
    assert plan is not None and evaluation.evaluation.objective is not None
    return {
        "iteration": iteration,
        "monotonic_ns": timestamp_ns,
        "elapsed_s": (timestamp_ns - start_ns) / 1e9,
        "objective": float(evaluation.evaluation.objective),
        "state": state_to_dict(state),
        "state_sha256": state_hash(state),
        "plan_sha256": canonical_hash(plan.to_dict()),
    }


def run_search_v2(
    instance: AllocationInstance,
    context: OracleContext,
    initializer: InitializerOutcome,
    config: AlnsV2Config,
    *,
    mode: str,
    task_group_id: str,
    difficulty: str,
    split: str,
    single_operator: str | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> SearchV2Outcome:
    config.validate()
    start_ns = initializer.provenance.start_monotonic_ns
    deadline_ns = (
        start_ns + int(float(config.end_to_end_time_s) * 1e9)
        if config.budget_mode == "fixed_time"
        else start_ns + int(config.safety_watchdog_s * 1e9)
    )
    current_state = canonicalize_state(instance, initializer.state)
    current = evaluate_state_timed(
        instance, context, current_state, config.objective_weights, clock=clock
    )
    if not current.evaluation.verified and not config.allow_infeasible_current:
        raise ValueError("infeasible initializer prohibited by configuration")
    best_state = current_state if current.evaluation.verified else None
    best = current if current.evaluation.verified else None
    incumbents = []
    if current.evaluation.verified:
        incumbents.append(
            _event(
                0,
                initializer.provenance.completion_monotonic_ns,
                start_ns,
                current_state,
                current,
            )
        )
    weights = {operator: 1.0 for operator in DESTROY_OPERATORS}
    segment_scores = {operator: 0.0 for operator in DESTROY_OPERATORS}
    segment_uses = {operator: 0 for operator in DESTROY_OPERATORS}
    visited = {state_hash(current_state)}
    steps = []
    started_iterations = 0
    completed_iterations = 0
    no_improvement = 0
    termination = "iteration_budget"

    for iteration in range(1, config.iterations + 1):
        now = max(clock(), initializer.provenance.completion_monotonic_ns)
        if now >= deadline_ns:
            termination = (
                "end_to_end_time_budget"
                if config.budget_mode == "fixed_time"
                else "safety_watchdog"
            )
            break
        started_iterations += 1
        iteration_seed = int(
            np.random.SeedSequence([config.random_seed, iteration]).generate_state(1)[0]
        )
        rng = np.random.default_rng(iteration_seed)
        guidance_started = clock()
        operator = _select_operator(mode, iteration, rng, weights, single_operator)
        ratio = config.destroy_ratios[(iteration - 1) % len(config.destroy_ratios)]
        destroyed = select_destroy_set_v2(
            operator, instance, context, current_state, current, ratio, rng
        )
        guidance_s = (clock() - guidance_started) / 1e9
        before_state, before = current_state, current
        repaired = repair_destroyed_state_v2(
            instance,
            context,
            before_state,
            destroyed,
            weights=config.objective_weights,
            candidate_evaluation_budget=config.repair_candidate_evaluation_budget,
            deadline_ns=deadline_ns,
            clock=clock,
        )
        candidate_ns = clock()
        completed_before_cutoff = candidate_ns <= deadline_ns
        candidate_state, candidate = repaired.state, repaired.evaluation
        candidate_hash = state_hash(candidate_state)
        unseen = candidate_hash not in visited
        accepted = False
        global_best = False
        reward = 0.0
        weight_before = weights[operator]
        if completed_before_cutoff:
            completed_iterations += 1
            accepted = _accept(before, candidate, iteration, rng, config)
            global_best = candidate.evaluation.verified and (
                best is None
                or float(candidate.evaluation.objective) + 1e-12
                < float(best.evaluation.objective)
            )
            if global_best:
                best_state, best = candidate_state, candidate
                incumbents.append(
                    _event(iteration, candidate_ns, start_ns, candidate_state, candidate)
                )
                no_improvement = 0
            else:
                no_improvement += 1
            if accepted:
                current_state, current = candidate_state, candidate
                visited.add(candidate_hash)
            reward = _reward(
                before,
                candidate,
                accepted=accepted,
                global_best=global_best,
                unseen=unseen,
                config=config,
            )
            if mode == "adaptive_alns":
                if config.update_scheme == "online":
                    weights[operator] = _online_weight(
                        weights[operator], reward, config.reaction_factor
                    )
                else:
                    segment_scores[operator] += reward
                    segment_uses[operator] += 1
                    if completed_iterations % config.segment_length == 0:
                        weights = update_segmented_weights(
                            weights,
                            segment_scores,
                            segment_uses,
                            config.reaction_factor,
                        )
                        segment_scores = {item: 0.0 for item in DESTROY_OPERATORS}
                        segment_uses = {item: 0 for item in DESTROY_OPERATORS}

        steps.append(
            {
                "iteration": iteration,
                "iteration_seed": iteration_seed,
                "operator": operator,
                "destroy_ratio": ratio,
                "destroy_set": list(destroyed),
                "before_state": state_to_dict(before_state),
                "before_state_sha256": state_hash(before_state),
                "candidate_state": state_to_dict(candidate_state),
                "candidate_state_sha256": candidate_hash,
                "before_verified": before.evaluation.verified,
                "after_verified": candidate.evaluation.verified,
                "before_failure_reason": before.evaluation.failure_reason,
                "after_failure_reason": candidate.evaluation.failure_reason,
                "before_violation": before.diagnostic.vector.to_dict(),
                "after_violation": candidate.diagnostic.vector.to_dict(),
                "current_objective": before.evaluation.objective,
                "candidate_objective": candidate.evaluation.objective,
                "best_so_far_objective": (
                    None if best is None else best.evaluation.objective
                ),
                "guidance_runtime_s": guidance_s,
                "repair_runtime_s": repaired.runtime_s,
                "repair_selection_runtime_s": repaired.selection_runtime_s,
                "scheduler_runtime_s": candidate.timing.scheduler_s,
                "verifier_runtime_s": candidate.timing.verifier_s,
                "candidate_evaluations": repaired.candidate_evaluations,
                "budget_exhausted": repaired.budget_exhausted,
                "deadline_exhausted": repaired.deadline_exhausted,
                "assignment_edits": repaired.assignment_edits,
                "order_edits": repaired.order_edits,
                "total_modified_units": repaired.total_modified_units,
                "accepted": accepted,
                "incumbent_updated": global_best,
                "unseen_state": unseen,
                "reward": reward,
                "operator_weight_before": weight_before,
                "operator_weight_after": weights[operator],
                "completed_before_cutoff": completed_before_cutoff,
                "monotonic_ns": candidate_ns,
                "elapsed_s": (candidate_ns - start_ns) / 1e9,
            }
        )
        if not completed_before_cutoff:
            termination = (
                "end_to_end_time_budget"
                if config.budget_mode == "fixed_time"
                else "safety_watchdog"
            )
            break
        if no_improvement >= config.restart_no_improvement:
            if best_state is not None and best is not None:
                current_state, current = best_state, best
            else:
                current_state = canonicalize_state(instance, initializer.state)
                current = evaluate_state_timed(
                    instance, context, current_state, config.objective_weights, clock=clock
                )
            no_improvement = 0

    returned_ns = max(clock(), initializer.provenance.completion_monotonic_ns)
    if (
        config.budget_mode == "fixed_iterations"
        and completed_iterations == config.iterations
    ):
        termination = "iteration_budget"
    fixed_complete = (
        config.budget_mode == "fixed_iterations"
        and completed_iterations == config.iterations
    )
    cutoff_ns = (
        start_ns + int(float(config.end_to_end_time_s) * 1e9)
        if config.budget_mode == "fixed_time"
        else None
    )
    trace = {
        "version": "a4b-search-trace-v2",
        "protocol_id": config.protocol_id,
        "instance_id": instance.instance_id,
        "task_group_id": task_group_id,
        "difficulty": difficulty,
        "split": split,
        "method": mode if single_operator is None else f"{mode}:{single_operator}",
        "random_seed": config.random_seed,
        "budget_mode": config.budget_mode,
        "clock": "time.monotonic_ns",
        "start_monotonic_ns": start_ns,
        "cutoff_monotonic_ns": cutoff_ns,
        "return_monotonic_ns": returned_ns,
        "return_elapsed_s": (returned_ns - start_ns) / 1e9,
        "cutoff_overrun_s": (
            0.0 if cutoff_ns is None else max(0.0, (returned_ns - cutoff_ns) / 1e9)
        ),
        "initializer": initializer.provenance.to_dict(),
        "initial_state": state_to_dict(initializer.state),
        "config_sha256": config.sha256,
        "iterations_started": started_iterations,
        "iterations_completed": completed_iterations,
        "fixed_iteration_complete": fixed_complete,
        "steps": steps,
        "incumbents": incumbents,
        "termination_reason": termination,
        "operator_weights": dict(sorted(weights.items())),
    }
    trace["trace_sha256"] = canonical_hash(trace)
    return SearchV2Outcome(
        trace,
        current_state,
        current,
        best_state,
        best,
        tuple(sorted(weights.items())),
    )

