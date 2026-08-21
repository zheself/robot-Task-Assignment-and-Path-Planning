"""Sealed A3 final-evaluation primitives.

The caller controls when frozen files are opened.  This module keeps model,
baseline, aggregation and preregistered decision semantics in testable pure
functions so they can be exercised on fixtures and validation before sealing.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import wilcoxon

from .a3_protocol import PreparedA3Development
from .decoding import decode_masked_candidate
from .generation import canonical_instance_bytes
from .graphs import build_a3_graph
from .models import A3AllocationModel
from .oracle import OracleContext
from .paper_benchmark import PaperManifest, PaperRecord
from .schema import (
    AllocationInstance,
    allocation_instance_from_dict,
    allocation_plan_from_dict,
)
from .solvers import (
    SolverProtocol,
    solve_assignment_milp,
    solve_deterministic_lns,
    solve_greedy,
    solve_hungarian,
    solve_hybrid_assignment_milp,
    solve_hybrid_load_balanced,
    solve_load_balanced,
    solve_order_aware_lns,
)
from .training import state_dict_sha256
from .verifier import verify_plan
from .witness import ConstructiveWitness, verify_constructive_witness

EXPECTED_METHODS = (
    "greedy",
    "load_balanced",
    "hungarian",
    "assignment_milp",
    "deterministic_lns",
    "hybrid_load_balanced",
    "hybrid_assignment_milp",
    "order_aware_lns",
)
EXPECTED_FINAL_SPLITS = frozenset({"frozen_test", "stress"})


@dataclass(frozen=True)
class FinalEvaluationItem:
    record: PaperRecord
    instance: AllocationInstance


@dataclass(frozen=True)
class LockedModel:
    seed: int
    state_sha256: str
    checkpoint_sha256: str
    model: A3AllocationModel


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protocol_locks(
    project_root: str | Path, protocol: Mapping[str, Any]
) -> tuple[str, ...]:
    root = Path(project_root)
    failures: list[str] = []
    source_paths = {
        "training_py_sha256": "src/safe_residual_rl/allocation/training.py",
        "model_py_sha256": "src/safe_residual_rl/allocation/models/heterogeneous.py",
        "decoding_py_sha256": "src/safe_residual_rl/allocation/decoding.py",
        "scheduling_py_sha256": "src/safe_residual_rl/allocation/scheduling.py",
        "verifier_py_sha256": "src/safe_residual_rl/allocation/verifier.py",
        "heuristics_py_sha256": "src/safe_residual_rl/allocation/solvers/heuristics.py",
        "milp_py_sha256": "src/safe_residual_rl/allocation/solvers/milp.py",
        "lns_py_sha256": "src/safe_residual_rl/allocation/solvers/lns.py",
        "search_scheduling_py_sha256": "src/safe_residual_rl/allocation/search_scheduling.py",
        "oracle_config_sha256": "configs/allocation/oracle_proxy_v1.json",
    }
    for key, relative in source_paths.items():
        path = root / relative
        expected = str(protocol["source_code_locks"][key])
        if not path.is_file() or sha256_file(path) != expected:
            failures.append(f"SOURCE_LOCK_MISMATCH={relative}")
    source_locks = protocol["source_locks"]
    other_paths = {
        "benchmark_config_sha256": "configs/allocation/benchmark_v4.json",
        "a3_development_config_sha256": "configs/allocation/a3_development_v1.json",
        "a3_development_summary_sha256": "reports/phase1_allocation/a3_w10_development_v1_summary.json",
    }
    for key, relative in other_paths.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != str(source_locks[key]):
            failures.append(f"PROVENANCE_LOCK_MISMATCH={relative}")
    return tuple(failures)


def load_locked_models(
    project_root: str | Path,
    protocol: Mapping[str, Any],
    prepared: PreparedA3Development,
) -> tuple[LockedModel, ...]:
    root = Path(project_root)
    if prepared.vocabulary.sha256 != protocol["source_locks"]["vocabulary_sha256"]:
        raise RuntimeError("locked vocabulary hash mismatch")
    if prepared.normalizer.sha256 != protocol["source_locks"]["normalizer_sha256"]:
        raise RuntimeError("locked normalizer hash mismatch")
    model_config = json.loads(
        (root / "configs/allocation/a3_development_v1.json").read_text(
            encoding="utf-8"
        )
    )["models"]
    template = prepared.train_examples[0].graph
    locked: list[LockedModel] = []
    for item in protocol["selected_model"]["checkpoints"]:
        seed = int(item["seed"])
        checkpoint = (
            root
            / "outputs/phase1_allocation/a3_w10_development_v1/shards/edge_mlp"
            / f"seed_{seed:03d}/checkpoint.pt"
        )
        file_digest = sha256_file(checkpoint)
        if file_digest != item["checkpoint_file_sha256"]:
            raise RuntimeError(f"checkpoint file hash mismatch for seed {seed}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state_dict_sha256(state) != item["state_sha256"]:
            raise RuntimeError(f"checkpoint state hash mismatch for seed {seed}")
        model = A3AllocationModel(
            template,
            family="edge_mlp",
            hidden_dim=int(model_config["hidden_dim"]),
            layers=int(model_config["message_passing_layers"]),
            heads=int(model_config["attention_heads"]),
            dropout=float(model_config["dropout"]),
        )
        model.load_state_dict(state, strict=True)
        model.eval()
        locked.append(LockedModel(seed, str(item["state_sha256"]), file_digest, model))
    if tuple(item.seed for item in locked) != (17, 29, 43):
        raise RuntimeError("locked seed matrix changed")
    return tuple(locked)


def load_final_items(
    project_root: str | Path,
    manifest: PaperManifest,
    splits: Sequence[str],
) -> tuple[FinalEvaluationItem, ...]:
    requested = frozenset(str(item) for item in splits)
    if not requested or not requested <= EXPECTED_FINAL_SPLITS:
        raise PermissionError(f"sealed loader only permits final splits: {sorted(requested)}")
    root = Path(project_root).resolve()
    items: list[FinalEvaluationItem] = []
    for record in sorted(
        (item for item in manifest.records if item.split in requested),
        key=lambda item: (item.split, item.cell_id, item.instance_id),
    ):
        path = (root / record.relative_path).resolve()
        if root not in path.parents or record.split not in path.parts:
            raise PermissionError(f"record path escapes sealed split: {record.instance_id}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        instance = allocation_instance_from_dict(raw)
        digest = hashlib.sha256(canonical_instance_bytes(instance)).hexdigest()
        if digest != record.sha256 or instance.instance_id != record.instance_id:
            raise RuntimeError(f"instance hash/id mismatch: {record.instance_id}")
        items.append(FinalEvaluationItem(record, instance))
    return tuple(items)


def build_evaluation_graph(
    item: FinalEvaluationItem,
    context: OracleContext,
    prepared: PreparedA3Development,
):
    # Reuse the locked train/validation graph builder without weakening its
    # development access guard. Only the immutable graph split tag is replaced.
    graph = build_a3_graph(
        item.instance, context, prepared.vocabulary, split="validation"
    )
    graph = replace(graph, split=item.record.split)
    return prepared.normalizer.transform(graph)


def evaluate_items(
    items: Sequence[FinalEvaluationItem],
    models: Sequence[LockedModel],
    context: OracleContext,
    prepared: PreparedA3Development,
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weights = {
        key: float(value)
        for key, value in json.loads(
            (Path(__file__).resolve().parents[3] / "configs/allocation/benchmark_v4.json").read_text(
                encoding="utf-8"
            )
        )["objective_weights"].items()
    }
    registry = _baseline_registry(context, weights, protocol)
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for item in items:
        graph = build_evaluation_graph(item, context, prepared)
        for locked in models:
            started = time.perf_counter()
            with torch.no_grad():
                output = locked.model(graph)
                candidate = decode_masked_candidate(
                    graph,
                    item.instance,
                    context,
                    output.assignment_logits,
                    output.order_scores,
                    method_id=f"a3-edge-mlp-seed-{locked.seed}-final-v1",
                )
            check = (
                verify_plan(item.instance, candidate.plan, context)
                if candidate.plan is not None
                else None
            )
            elapsed = time.perf_counter() - started
            verified = bool(check and check.feasible)
            objectives = dict(check.objective_terms) if check else {}
            method = f"edge_mlp_seed_{locked.seed}"
            rows.append(
                _row(item, method, candidate.status, verified, elapsed, objectives, weights,
                     None, None, check)
            )
            raw.append(
                {
                    "instance_id": item.record.instance_id,
                    "method": method,
                    "seed": locked.seed,
                    "status": candidate.status,
                    "assignment": [list(value) for value in candidate.assignment],
                    "robot_orders": [
                        [robot, list(order)] for robot, order in candidate.robot_orders
                    ],
                    "diagnostics": list(candidate.diagnostics),
                    "plan": None if candidate.plan is None else candidate.plan.to_dict(),
                }
            )
        for method, solve in registry.items():
            started = time.perf_counter()
            result = solve(item.instance)
            check = (
                verify_plan(item.instance, result.plan, context)
                if result.plan is not None
                else None
            )
            elapsed = time.perf_counter() - started
            verified = bool(check and check.feasible)
            objectives = dict(check.objective_terms) if check else {}
            rows.append(
                _row(item, method, result.status, verified, elapsed, objectives, weights,
                     result.mip_gap, result.best_bound, check)
            )
            raw.append(
                {
                    "instance_id": item.record.instance_id,
                    "method": method,
                    "status": result.status,
                    "diagnostics": list(result.diagnostics),
                    "plan": None if result.plan is None else result.plan.to_dict(),
                }
            )
    return rows, raw


def audit_witnesses_after_prediction(
    project_root: str | Path,
    items: Sequence[FinalEvaluationItem],
    raw_predictions: Sequence[Mapping[str, Any]],
    context: OracleContext,
) -> tuple[list[str], dict[tuple[str, str], float]]:
    root = Path(project_root).resolve()
    learned_assignment = {
        (str(item["instance_id"]), str(item["method"])): dict(item["assignment"])
        for item in raw_predictions
        if str(item["method"]).startswith("edge_mlp_seed_")
    }
    failures: list[str] = []
    agreements: dict[tuple[str, str], float] = {}
    for item in items:
        record = item.record
        if record.witness_relative_path is None:
            continue
        path = (root / record.witness_relative_path).resolve()
        if root not in path.parents or record.split not in path.parts:
            failures.append(f"WITNESS_PATH_SCOPE={record.instance_id}")
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        plan = allocation_plan_from_dict(raw["plan"], item.instance)
        witness = ConstructiveWitness(
            item.instance,
            plan,
            str(raw["witness_sha256"]),
            tuple(str(value) for value in raw.get("diagnostics", ())),
        )
        issues = verify_constructive_witness(witness, context)
        if witness.witness_sha256 != record.witness_sha256:
            issues = issues + ("MANIFEST_WITNESS_HASH_MISMATCH",)
        failures.extend(f"{issue}={record.instance_id}" for issue in issues)
        teacher = {entry.segment_id: entry.robot_id for entry in plan.schedule}
        for method in (f"edge_mlp_seed_{seed}" for seed in (17, 29, 43)):
            predicted = learned_assignment.get((record.instance_id, method), {})
            agreements[(record.instance_id, method)] = (
                sum(predicted.get(key) == value for key, value in teacher.items())
                / len(teacher)
            )
    return failures, agreements


def method_cell_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["cell_id"]), str(row["method"]))].append(row)
    output: list[dict[str, Any]] = []
    for (split, cell, method), values in sorted(grouped.items()):
        by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for value in values:
            by_group[str(value["task_group_id"])].append(value)
        rates = [statistics.mean(float(bool(x["verified"])) for x in group) for group in by_group.values()]
        center, low, high = _bootstrap_mean(rates, resamples, confidence, _seed(seed, split, cell, method))
        verified = [value for value in values if value["verified"]]
        runtimes = sorted(float(value["runtime_s"]) for value in values)
        output.append(
            {
                "split": split,
                "cell_id": cell,
                "method": method,
                "groups": len(by_group),
                "instances": len(values),
                "verified_instances": len(verified),
                "group_verified_rate": center,
                "group_verified_ci_low": low,
                "group_verified_ci_high": high,
                "conditional_mean_score": _mean(value["weighted_proxy_score"] for value in verified),
                "median_runtime_s": statistics.median(runtimes),
                "runtime_q1_s": float(np.quantile(runtimes, 0.25)),
                "runtime_q3_s": float(np.quantile(runtimes, 0.75)),
            }
        )
    return output


def learned_group_rates(rows: Sequence[Mapping[str, Any]], split: str) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["split"] == split and str(row["method"]).startswith("edge_mlp_seed_"):
            grouped[(str(row["cell_id"]), str(row["task_group_id"]))].append(float(bool(row["verified"])))
    return {key: statistics.mean(values) for key, values in grouped.items()}


def baseline_group_rates(
    rows: Sequence[Mapping[str, Any]], split: str, method: str
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["split"] == split and row["method"] == method:
            grouped[(str(row["cell_id"]), str(row["task_group_id"]))].append(float(bool(row["verified"])))
    return {key: statistics.mean(values) for key, values in grouped.items()}


def strong_pairwise_statistics(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    learned = learned_group_rates(rows, "frozen_test")
    stats = protocol["statistics"]
    results: list[dict[str, Any]] = []
    for reference in protocol["baselines"]["strong_methods"]:
        baseline = baseline_group_rates(rows, "frozen_test", reference)
        keys = sorted(set(learned) & set(baseline))
        differences = [learned[key] - baseline[key] for key in keys]
        center, low, high = _bootstrap_mean(
            differences,
            int(stats["cluster_bootstrap_resamples"]),
            float(stats["confidence_level"]),
            _seed(int(stats["bootstrap_seed"]), "overall", reference, "coverage"),
        )
        nonzero = [value for value in differences if abs(value) > 1e-12]
        p_value = (
            float(wilcoxon(nonzero, alternative="greater", method="auto").pvalue)
            if len(nonzero) >= 5
            else None
        )
        cell_differences = {
            cell: statistics.mean(
                learned[key] - baseline[key] for key in keys if key[0] == cell
            )
            for cell in sorted({key[0] for key in keys})
        }
        quality = _paired_quality(rows, reference)
        results.append(
            {
                "reference_method": reference,
                "groups": len(keys),
                "coverage_difference": center,
                "coverage_ci_low": low,
                "coverage_ci_high": high,
                "coverage_wilcoxon_p_raw": p_value,
                "coverage_wilcoxon_p_holm": None,
                "cell_coverage_differences": cell_differences,
                **quality,
            }
        )
    _holm(results, "coverage_wilcoxon_p_raw", "coverage_wilcoxon_p_holm")
    return results


def classify_final_result(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    pairwise: Sequence[Mapping[str, Any]],
    integrity_checks: Mapping[str, bool],
) -> dict[str, Any]:
    if not all(integrity_checks.values()):
        return {"result_class": "A3_FINAL_INVALID", "a3_final_passed": False, "checks": dict(integrity_checks)}
    learned = learned_group_rates(rows, "frozen_test")
    cells = protocol["data_access"]["primary_cells"]
    learned_cell = {
        cell: statistics.mean(value for (name, _), value in learned.items() if name == cell)
        for cell in cells
    }
    learned_overall = statistics.mean(learned.values())
    context = protocol["baselines"]["context_methods"]
    context_rates = {
        method: baseline_group_rates(rows, "frozen_test", method) for method in context
    }
    context_overall = {
        method: statistics.mean(values.values()) for method, values in context_rates.items()
    }
    best_context_overall = max(context_overall.values())
    best_context_cell = {
        cell: max(
            statistics.mean(value for (name, _), value in values.items() if name == cell)
            for values in context_rates.values()
        )
        for cell in cells
    }
    absolute = protocol["difficulty_gates"]["absolute_mean_seed_coverage"]
    margin = float(protocol["difficulty_gates"]["relative_weak_baseline_margin"])
    gate_checks = {
        f"absolute_{cell}": learned_cell[cell] >= float(absolute[cell]) for cell in cells
    }
    gate_checks["overall_not_below_best_context"] = learned_overall >= best_context_overall
    gate_checks.update(
        {
            f"relative_context_{cell}": learned_cell[cell] >= best_context_cell[cell] - margin
            for cell in cells
        }
    )
    combined = {**dict(integrity_checks), **gate_checks}
    if not all(gate_checks.values()):
        result_class = "A3_FINAL_FAILED_BASELINE_FLOOR"
    else:
        strong_margin = float(protocol["decision_rules"]["strong_competitive_margin"])
        cell_margin = float(protocol["difficulty_gates"]["relative_weak_baseline_margin"])
        competitive = any(
            item["coverage_ci_low"] is not None
            and float(item["coverage_ci_low"]) >= -strong_margin
            and min(float(value) for value in item["cell_coverage_differences"].values()) >= -cell_margin
            for item in pairwise
        )
        superior = all(
            item["coverage_ci_low"] is not None
            and float(item["coverage_ci_low"]) > 0.0
            and item["coverage_wilcoxon_p_holm"] is not None
            and float(item["coverage_wilcoxon_p_holm"]) < float(protocol["statistics"]["alpha"])
            and min(float(value) for value in item["cell_coverage_differences"].values()) >= -cell_margin
            for item in pairwise
        )
        result_class = (
            "A3_FINAL_STRONG_ADVANTAGE"
            if superior
            else "A3_FINAL_COMPETITIVE_NOT_SUPERIOR"
            if competitive
            else "A3_FINAL_BASELINE_FLOOR_ONLY"
        )
    return {
        "result_class": result_class,
        "a3_final_passed": result_class not in {"A3_FINAL_INVALID", "A3_FINAL_FAILED_BASELINE_FLOOR"},
        "learned_overall_coverage": learned_overall,
        "learned_cell_coverage": learned_cell,
        "best_context_overall_coverage": best_context_overall,
        "best_context_cell_coverage": best_context_cell,
        "checks": combined,
        "required_wording": protocol["result_classes"][result_class],
    }


def _baseline_registry(context, weights, protocol):
    baseline = protocol["baselines"]
    solver_protocol = SolverProtocol(
        "a1-solver-protocol-v1",
        float(baseline["milp_time_limit_s"]),
        float(baseline["milp_relative_gap"]),
        0,
    )
    registry = {
        "greedy": lambda x: solve_greedy(x, context),
        "load_balanced": lambda x: solve_load_balanced(x, context),
        "hungarian": lambda x: solve_hungarian(x, context),
        "assignment_milp": lambda x: solve_assignment_milp(x, context, solver_protocol),
        "deterministic_lns": lambda x: solve_deterministic_lns(
            x, context, iterations=int(baseline["lns_iterations"]),
            seed=int(baseline["lns_seed"]), objective_weights=weights
        ),
        "hybrid_load_balanced": lambda x: solve_hybrid_load_balanced(x, context),
        "hybrid_assignment_milp": lambda x: solve_hybrid_assignment_milp(x, context, solver_protocol),
        "order_aware_lns": lambda x: solve_order_aware_lns(
            x, context, iterations=int(baseline["lns_iterations"]),
            seed=int(baseline["lns_seed"]), objective_weights=weights
        ),
    }
    if tuple(registry) != EXPECTED_METHODS:
        raise RuntimeError("baseline registry changed")
    return registry


def _row(item, method, status, verified, runtime, objectives, weights, gap, bound, check):
    return {
        "instance_id": item.record.instance_id,
        "split": item.record.split,
        "cell_id": item.record.cell_id,
        "paper_role": item.record.paper_role,
        "task_group_id": item.record.task_group_id,
        "variant_index": item.record.variant_index,
        "method": method,
        "status": status,
        "verified": verified,
        "runtime_s": runtime,
        "weighted_proxy_score": _score(objectives, weights) if verified else None,
        "makespan_s": objectives.get("makespan"),
        "load_variance_s2": objectives.get("load_variance"),
        "travel_setup_time_s": objectives.get("travel_setup_time"),
        "priority_tardiness": objectives.get("priority_tardiness"),
        "assignment_mip_gap": gap,
        "assignment_best_bound": bound,
        "violation_codes": "" if check is None else ";".join(sorted({value.code for value in check.violations})),
        "teacher_agreement": None,
    }


def _score(objectives, weights):
    return sum(float(weights.get(key, 0.0)) * float(value) for key, value in objectives.items())


def _paired_quality(rows, reference):
    by_instance = {(str(x["instance_id"]), str(x["method"])): x for x in rows if x["split"] == "frozen_test"}
    instance_group = {str(x["instance_id"]): str(x["task_group_id"]) for x in rows if x["split"] == "frozen_test"}
    grouped: dict[str, list[float]] = defaultdict(list)
    paired = 0
    for instance_id, group in sorted(instance_group.items()):
        right = by_instance.get((instance_id, reference))
        if right is None or not right["verified"]:
            continue
        for seed in (17, 29, 43):
            left = by_instance.get((instance_id, f"edge_mlp_seed_{seed}"))
            if left and left["verified"]:
                grouped[group].append(float(left["weighted_proxy_score"]) - float(right["weighted_proxy_score"]))
                paired += 1
    diffs = [statistics.mean(values) for values in grouped.values() if values]
    p = None
    nonzero = [value for value in diffs if abs(value) > 1e-12]
    if len(nonzero) >= 5:
        p = float(wilcoxon(nonzero, alternative="two-sided", method="auto").pvalue)
    return {
        "joint_quality_groups": len(diffs),
        "joint_quality_seed_variants": paired,
        "conditional_score_difference": _mean(diffs),
        "conditional_score_wilcoxon_p_raw": p,
    }


def _bootstrap_mean(values, resamples, confidence, seed):
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=float)
    center = float(np.mean(array))
    if len(array) == 1:
        return center, center, center
    rng = np.random.default_rng(seed)
    samples = np.mean(array[rng.integers(0, len(array), size=(resamples, len(array)))], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return center, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def _seed(seed, *parts):
    raw = "|".join([str(seed), *(str(value) for value in parts)]).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def _mean(values):
    selected = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.mean(selected) if selected else None


def _holm(rows, raw_key, adjusted_key):
    selected = sorted(
        ((index, float(row[raw_key])) for index, row in enumerate(rows) if row[raw_key] is not None),
        key=lambda value: value[1],
    )
    running = 0.0
    for rank, (index, p_value) in enumerate(selected):
        adjusted = min(1.0, (len(selected) - rank) * p_value)
        running = max(running, adjusted)
        rows[index][adjusted_key] = running
