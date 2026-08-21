"""Protocol-R search loop with the semantics-preserving prepared repair."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ..oracle import OracleContext
from ..repair.identical import canonicalize_state
from ..schema import AllocationInstance
from .alns_v2 import (
    AlnsV2Config,
    SearchV2Outcome,
    _accept,
    _online_weight,
    _reward,
    _select_operator,
    update_segmented_weights,
)
from .anytime import InitializerOutcome
from .operators import DESTROY_OPERATORS
from .operators_v2 import select_destroy_set_v2
from .prepared_repair import evaluate_state_timed_prepared, prepare_repair_problem
from .repair_protocol_r import RepairTraceCache, repair_destroyed_state_protocol_r
from .trace import canonical_hash, state_hash, state_to_dict

ProtocolRSearchConfig = AlnsV2Config


def _event(iteration, timestamp_ns, start_ns, state, evaluation):
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


def run_search_protocol_r(
    instance: AllocationInstance,
    context: OracleContext,
    initializer: InitializerOutcome,
    config: ProtocolRSearchConfig,
    *,
    mode: str,
    task_group_id: str,
    difficulty: str,
    split: str,
    single_operator: str | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> SearchV2Outcome:
    """Run v2-equivalent search while counting all preparation inside E2E time."""
    config.validate()
    start_ns = initializer.provenance.start_monotonic_ns
    deadline_ns = (
        start_ns + int(float(config.end_to_end_time_s) * 1e9)
        if config.budget_mode == "fixed_time"
        else start_ns + int(config.safety_watchdog_s * 1e9)
    )
    preparation_started = clock()
    prepared = prepare_repair_problem(instance, context)
    trace_cache = RepairTraceCache()
    preparation_runtime_s = (clock() - preparation_started) / 1e9
    current_state = canonicalize_state(instance, initializer.state)
    current = evaluate_state_timed_prepared(
        instance,
        context,
        prepared,
        current_state,
        config.objective_weights,
        clock=clock,
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
        repaired = repair_destroyed_state_protocol_r(
            instance,
            context,
            before_state,
            destroyed,
            weights=config.objective_weights,
            candidate_evaluation_budget=config.repair_candidate_evaluation_budget,
            prepared=prepared,
            cache=trace_cache,
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
                "candidate_sequence_sha256": repaired.candidate_sequence_sha256,
                "diagnostic_vector_sha256": repaired.diagnostic_vector_sha256,
                "repair_cache_hits": repaired.cache_hits,
                "repair_cache_misses": repaired.cache_misses,
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
                current = evaluate_state_timed_prepared(
                    instance,
                    context,
                    prepared,
                    current_state,
                    config.objective_weights,
                    clock=clock,
                )
            no_improvement = 0

    returned_ns = max(clock(), initializer.provenance.completion_monotonic_ns)
    if config.budget_mode == "fixed_iterations" and completed_iterations == config.iterations:
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
        "version": "a4b-protocol-r-search-trace-v1",
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
        "prepared_sha256": prepared.prepared_sha256,
        "preparation_runtime_s": preparation_runtime_s,
        "iterations_started": started_iterations,
        "iterations_completed": completed_iterations,
        "fixed_iteration_complete": fixed_complete,
        "steps": steps,
        "incumbents": incumbents,
        "termination_reason": termination,
        "operator_weights": dict(sorted(weights.items())),
        "repair_cache": {
            "scope": "same_instance_same_method_same_seed_same_trace_only",
            "hits": trace_cache.hits,
            "misses": trace_cache.misses,
        },
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


def transition_signature(trace: dict[str, object]) -> str:
    """Timing-independent transition signature shared with reference tests."""
    steps = trace.get("steps", [])
    assert isinstance(steps, list)
    return canonical_hash(
        [
            {
                "iteration": step["iteration"],
                "iteration_seed": step["iteration_seed"],
                "operator": step["operator"],
                "destroy_ratio": step["destroy_ratio"],
                "destroy_set": step["destroy_set"],
                "before_state_sha256": step["before_state_sha256"],
                "candidate_state_sha256": step["candidate_state_sha256"],
                "accepted": step["accepted"],
                "incumbent_updated": step["incumbent_updated"],
                "reward": step["reward"],
                "operator_weight_after": step["operator_weight_after"],
            }
            for step in steps
        ]
    )
