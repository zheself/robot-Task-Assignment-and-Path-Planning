"""Preregistered group-clustered statistics and A2-v2 gate evaluation."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

EXPECTED_STATUSES = {
    "feasible",
    "optimal",
    "feasible_limit",
    "infeasible",
    "schedule_infeasible",
    "limit",
}


def method_cell_statistics(
    rows: Sequence[Mapping[str, Any]],
    resamples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["cell_id"]), str(row["method"]))].append(row)
    results: list[dict[str, Any]] = []
    for (split, cell_id, method), values in sorted(grouped.items()):
        by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in values:
            by_group[str(row["task_group_id"])].append(row)
        group_rates = [sum(bool(item["verified"]) for item in group) / len(group) for group in by_group.values()]
        rate, rate_low, rate_high = cluster_mean_ci(group_rates, resamples, confidence, _seed(seed, split, cell_id, method, "rate"))
        verified = [item for item in values if item["verified"]]
        runtimes = sorted(float(item["runtime_s"]) for item in values)
        results.append(
            {
                "split": split,
                "cell_id": cell_id,
                "method": method,
                "groups": len(by_group),
                "instances": len(values),
                "verified_instances": len(verified),
                "group_verified_rate": rate,
                "group_verified_ci_low": rate_low,
                "group_verified_ci_high": rate_high,
                "median_runtime_s": statistics.median(runtimes),
                "runtime_q1_s": _quantile(runtimes, 0.25),
                "runtime_q3_s": _quantile(runtimes, 0.75),
                "conditional_mean_score": _mean(item.get("weighted_proxy_score") for item in verified),
                "conditional_mean_makespan_s": _mean(item.get("makespan_s") for item in verified),
                "conditional_mean_load_variance_s2": _mean(item.get("load_variance_s2") for item in verified),
                "conditional_mean_travel_setup_time_s": _mean(item.get("travel_setup_time_s") for item in verified),
            }
        )
    return results


def frozen_pairwise_statistics(
    rows: Sequence[Mapping[str, Any]],
    reference_methods: Sequence[str],
    resamples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    frozen = [row for row in rows if row["split"] == "frozen_test"]
    cells = sorted({str(row["cell_id"]) for row in frozen})
    methods = sorted({str(row["method"]) for row in frozen})
    results: list[dict[str, Any]] = []
    for cell_id in cells:
        selected = [row for row in frozen if row["cell_id"] == cell_id]
        by_instance_method = {(str(row["instance_id"]), str(row["method"])): row for row in selected}
        group_by_instance = {str(row["instance_id"]): str(row["task_group_id"]) for row in selected}
        instances = sorted(group_by_instance)
        for reference in reference_methods:
            for method in methods:
                if method == reference:
                    continue
                feasibility_by_group: dict[str, list[float]] = defaultdict(list)
                score_by_group: dict[str, list[float]] = defaultdict(list)
                paired_variants = 0
                for instance_id in instances:
                    left = by_instance_method[(instance_id, method)]
                    right = by_instance_method[(instance_id, reference)]
                    group_id = group_by_instance[instance_id]
                    feasibility_by_group[group_id].append(float(bool(left["verified"])) - float(bool(right["verified"])))
                    if left["verified"] and right["verified"] and left.get("weighted_proxy_score") is not None and right.get("weighted_proxy_score") is not None:
                        score_by_group[group_id].append(float(left["weighted_proxy_score"]) - float(right["weighted_proxy_score"]))
                        paired_variants += 1
                feasibility_diffs = [statistics.mean(values) for values in feasibility_by_group.values()]
                score_diffs = [statistics.mean(values) for values in score_by_group.values() if values]
                feasibility = cluster_mean_ci(feasibility_diffs, resamples, confidence, _seed(seed, cell_id, method, reference, "feasibility"))
                score = cluster_mean_ci(score_diffs, resamples, confidence, _seed(seed, cell_id, method, reference, "score")) if score_diffs else (None, None, None)
                p_value = _wilcoxon_p(score_diffs)
                results.append(
                    {
                        "cell_id": cell_id,
                        "method": method,
                        "reference_method": reference,
                        "groups_total": len(feasibility_diffs),
                        "groups_jointly_verified": len(score_diffs),
                        "paired_variants": paired_variants,
                        "feasibility_rate_difference": feasibility[0],
                        "feasibility_ci_low": feasibility[1],
                        "feasibility_ci_high": feasibility[2],
                        "score_difference": score[0],
                        "score_ci_low": score[1],
                        "score_ci_high": score[2],
                        "wilcoxon_p_raw": p_value,
                        "wilcoxon_p_holm": None,
                    }
                )
    _holm_adjust(results)
    return results


def evaluate_acceptance(
    rows: Sequence[Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    audit: Mapping[str, int],
) -> dict[str, Any]:
    instance_methods: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        instance_methods[str(row["instance_id"])].append(row)
    meta = {instance_id: values[0] for instance_id, values in instance_methods.items()}

    def coverage(split: str, cell_id: str | None = None) -> float:
        ids = [instance_id for instance_id, row in meta.items() if row["split"] == split and (cell_id is None or row["cell_id"] == cell_id)]
        return sum(any(bool(item["verified"]) for item in instance_methods[item_id]) for item_id in ids) / len(ids) if ids else 0.0

    train_coverage = coverage("train")
    validation_coverage = coverage("validation")
    frozen_cells = sorted({str(row["cell_id"]) for row in meta.values() if row["split"] == "frozen_test"})
    frozen_coverage = {cell: coverage("frozen_test", cell) for cell in frozen_cells}
    negative_ids = [instance_id for instance_id, row in meta.items() if row["cell_id"] == "designed_edge_infeasible"]
    negative_detected = sum(
        all(item["status"] == "infeasible" and not item["verified"] for item in instance_methods[instance_id])
        for instance_id in negative_ids
    )
    negative_rate = negative_detected / len(negative_ids) if negative_ids else 0.0
    unexpected = [row for row in rows if str(row["status"]) not in EXPECTED_STATUSES]
    checks = {
        "zero_schema_failures": audit.get("schema_failures", 0) == 0,
        "zero_split_leakage": audit.get("split_leakage", 0) == 0,
        "zero_hash_failures": audit.get("hash_failures", 0) == 0,
        "zero_unexpected_solver_status": not unexpected,
        "minimum_train_candidate_coverage": train_coverage >= float(acceptance["minimum_train_candidate_coverage"]),
        "minimum_validation_candidate_coverage": validation_coverage >= float(acceptance["minimum_validation_candidate_coverage"]),
        "minimum_frozen_cell_candidate_coverage": all(value >= float(acceptance["minimum_frozen_cell_candidate_coverage"]) for value in frozen_coverage.values()),
        "designed_infeasible_detection_rate": negative_rate >= float(acceptance["designed_infeasible_detection_rate"]),
    }
    if "zero_witness_failures" in acceptance:
        checks["zero_witness_failures"] = audit.get("witness_failures", 0) == 0
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "train_candidate_coverage": train_coverage,
        "validation_candidate_coverage": validation_coverage,
        "frozen_cell_candidate_coverage": frozen_coverage,
        "designed_infeasible_detection_rate": negative_rate,
        "unexpected_status_count": len(unexpected),
    }


def cluster_mean_ci(
    values: Sequence[float], resamples: int, confidence: float, seed: int
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    data = np.asarray(values, dtype=float)
    observed = float(np.mean(data))
    if len(data) == 1:
        return observed, observed, observed
    rng = np.random.default_rng(seed)
    samples = np.mean(data[rng.integers(0, len(data), size=(resamples, len(data)))], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return observed, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def _wilcoxon_p(values: Sequence[float]) -> float | None:
    nonzero = [value for value in values if abs(value) > 1e-12]
    if len(nonzero) < 5:
        return None
    return float(wilcoxon(nonzero, alternative="two-sided", method="auto").pvalue)


def _holm_adjust(results: list[dict[str, Any]]) -> None:
    selected = sorted(
        ((index, float(item["wilcoxon_p_raw"])) for index, item in enumerate(results) if item["wilcoxon_p_raw"] is not None),
        key=lambda pair: pair[1],
    )
    total = len(selected)
    running = 0.0
    for rank, (index, p_value) in enumerate(selected):
        adjusted = min(1.0, (total - rank) * p_value)
        running = max(running, adjusted)
        results[index]["wilcoxon_p_holm"] = running


def _seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join([str(seed), *(str(item) for item in parts)]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _mean(values) -> float | None:
    selected = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.mean(selected) if selected else None


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))
