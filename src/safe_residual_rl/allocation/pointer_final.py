"""A3.5 sealed-final generator, evaluator and paired statistics.

The public entry points deliberately separate development evaluation, benchmark
generation, candidate prediction, witness audit and final aggregation.  The
runner controls the one-time access order.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .decoding import decode_masked_candidate
from .generation import BenchmarkConfig, canonical_instance_bytes, generate_instance, stable_seed
from .graphs import build_a3_graph
from .models import A3AllocationModel
from .oracle import OracleContext
from .pointer_decoder import FeasiblePairPointer
from .pointer_pilot import construct_pointer_compatible_witness, load_pointer_pilot_config
from .pointer_training import PreparedPointerPilot
from .schema import AllocationInstance, EvidenceLabel, allocation_instance_from_dict, allocation_plan_from_dict
from .solvers import SolverProtocol, solve_hybrid_assignment_milp, solve_hybrid_load_balanced, solve_order_aware_lns
from .training import state_dict_sha256
from .verifier import verify_plan

PROTOCOL_SHA256 = "8c4d3cb7cc6e61ee589e98786ab78e77958d09694afb102d7f806eda9c208368"
EXPECTED_CELLS = ("iid_small", "iid_medium", "dense_precedence", "resource_bottleneck", "tight_windows", "scale")
NEURAL_METHODS = ("pair_pointer", "static")
BASELINES = ("hybrid_assignment_milp", "order_aware_lns", "hybrid_load_balanced")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_final_protocol(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    if sha256_file(source) != PROTOCOL_SHA256:
        raise RuntimeError("A3.5 final protocol checksum changed")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("version") != "a3-5-sealed-final-v1":
        raise ValueError("unexpected A3.5 final protocol version")
    return raw


def verify_registered_locks(root: str | Path, protocol: Mapping[str, Any]) -> tuple[str, ...]:
    root = Path(root)
    failures: list[str] = []
    for relative, expected in protocol["source_locks"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            failures.append(f"SOURCE_LOCK_MISMATCH={relative}")
    training = protocol["training_lock"]
    provenance = {
        "pilot_config_sha256": "configs/allocation/a3_5_pointer_pilot_v1.json",
        "pilot_manifest_file_sha256": "outputs/phase1_allocation/a3_5_pointer_pilot_v1/manifest.json",
    }
    for key, relative in provenance.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != training[key]:
            failures.append(f"PROVENANCE_LOCK_MISMATCH={relative}")
    for method in protocol["fixed_neural_methods"].values():
        for seed, relative in method["checkpoint_files"].items():
            path = root / relative
            if not path.is_file() or sha256_file(path) != method["checkpoint_file_sha256"][seed]:
                failures.append(f"CHECKPOINT_FILE_MISMATCH={relative}")
            elif state_dict_sha256(torch.load(path, map_location="cpu", weights_only=True)) != method["state_sha256"][seed]:
                failures.append(f"CHECKPOINT_STATE_MISMATCH={relative}")
    return tuple(failures)


@dataclass(frozen=True)
class FixedModels:
    pointer: tuple[tuple[int, FeasiblePairPointer], ...]
    static: tuple[tuple[int, A3AllocationModel], ...]


def load_fixed_models(root: str | Path, protocol: Mapping[str, Any], prepared: PreparedPointerPilot) -> FixedModels:
    root = Path(root)
    lock = protocol["training_lock"]
    if prepared.access_sha256 != lock["train_validation_access_sha256"]:
        raise RuntimeError("train/validation access hash mismatch")
    if prepared.vocabulary.sha256 != lock["vocabulary_sha256"]:
        raise RuntimeError("vocabulary hash mismatch")
    if prepared.normalizer.sha256 != lock["normalizer_sha256"]:
        raise RuntimeError("normalizer hash mismatch")
    pilot = json.loads((root / "configs/allocation/a3_5_pointer_pilot_v1.json").read_text())
    model_cfg = pilot["models"]
    template = prepared.train_examples[0].graph
    pointer: list[tuple[int, FeasiblePairPointer]] = []
    static: list[tuple[int, A3AllocationModel]] = []
    for kind, destination in (("pair_pointer", pointer), ("matched_static", static)):
        item = protocol["fixed_neural_methods"][kind]
        for seed in item["seeds"]:
            state = torch.load(root / item["checkpoint_files"][str(seed)], map_location="cpu", weights_only=True)
            if kind == "pair_pointer":
                model = FeasiblePairPointer(template, encoder_family="hetero_gnn", hidden_dim=int(model_cfg["hidden_dim"]), layers=int(model_cfg["message_passing_layers"]), heads=int(model_cfg["attention_heads"]), dropout=float(model_cfg["dropout"]))
            else:
                model = A3AllocationModel(template, family="hetero_gnn", hidden_dim=int(model_cfg["hidden_dim"]), layers=int(model_cfg["message_passing_layers"]), heads=int(model_cfg["attention_heads"]), dropout=float(model_cfg["dropout"]))
            model.load_state_dict(state, strict=True)
            model.eval()
            destination.append((int(seed), model))
    return FixedModels(tuple(pointer), tuple(static))


def generate_final_benchmark(root: str | Path, protocol: Mapping[str, Any], output_root: str | Path, seal_sha256: str) -> Mapping[str, Any]:
    root = Path(root).resolve()
    output = Path(output_root)
    if not output.is_absolute():
        output = root / output
    if output.exists():
        raise FileExistsError("A3.5 final benchmark already exists")
    pilot_cfg = load_pointer_pilot_config(root / "configs/allocation/a3_5_pointer_pilot_v1.json")
    final = protocol["benchmark"]
    if tuple(item.cell_id for item in pilot_cfg.cells) != EXPECTED_CELLS:
        raise RuntimeError("registered difficulty cells changed")
    context = _load_context(root)
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for cell in pilot_cfg.cells:
            adapter = BenchmarkConfig(
                version="a3-5-sealed-final-generator-v1",
                manifest_version="a3-5-sealed-final-manifest-v1",
                master_seed=int(final["master_seed"]),
                evidence_label=EvidenceLabel.SIM_GEOMETRIC,
                coordinate_frame="synthetic_workcell_m",
                variants_per_group=int(final["variants_per_group"]),
                geometry=pilot_cfg.geometry,
                splits=(("frozen_test", replace(cell.spec, group_count=int(final["groups_per_cell"]))),),
                objective_weights=tuple(sorted(pilot_cfg.objective_weights.items())),
                baseline_protocol={},
                boundaries=("SIM_GEOMETRIC", "A3.5 sealed final; no physical/collision claim"),
            )
            for group_index in range(int(final["groups_per_cell"])):
                group_id = f"{final['id_prefix']}-frozen_test-{cell.cell_id}-group-{group_index:03d}"
                unique = stable_seed(int(final["master_seed"]), "frozen_test", cell.cell_id, group_index, "a3-5-sealed-final-group")
                for variant in range(int(final["variants_per_group"])):
                    generated = generate_instance(adapter, "frozen_test", group_id, unique, variant)
                    witness = construct_pointer_compatible_witness(
                        generated.instance,
                        context,
                        tight_pre_margin_duration=float(pilot_cfg.geometry["witness_tight_pre_margin_duration"]),
                        tight_post_margin_duration=float(pilot_cfg.geometry["witness_tight_post_margin_duration"]),
                        loose_pre_margin_s=float(pilot_cfg.geometry["witness_loose_pre_margin_s"]),
                    )
                    instance = witness.instance
                    if not instance.instance_id.startswith(str(final["id_prefix"])):
                        raise RuntimeError("final generator escaped ID namespace")
                    rel = Path("data/frozen_test") / cell.cell_id / f"{instance.instance_id}.json"
                    wrel = Path("witnesses/frozen_test") / cell.cell_id / f"{instance.instance_id}.json"
                    (output / rel).parent.mkdir(parents=True, exist_ok=True)
                    (output / wrel).parent.mkdir(parents=True, exist_ok=True)
                    (output / rel).write_text(json.dumps(instance.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
                    (output / wrel).write_text(json.dumps(witness.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
                    records.append({
                        "split": "frozen_test", "cell_id": cell.cell_id,
                        "task_group_id": group_id, "variant_index": variant,
                        "instance_id": instance.instance_id, "workpiece_id": instance.workpiece_id,
                        "layout_id": instance.layout_id,
                        "parent_curve_ids": sorted({item.parent_curve_id for item in instance.segments}),
                        "relative_path": rel.as_posix(),
                        "instance_sha256": hashlib.sha256(canonical_instance_bytes(instance)).hexdigest(),
                        "witness_relative_path": wrel.as_posix(),
                        "witness_sha256": witness.witness_sha256,
                        "evidence_label": "SIM_GEOMETRIC",
                    })
        _audit_records(records, protocol)
        payload = {
            "version": "a3-5-sealed-final-manifest-v1",
            "protocol_sha256": PROTOCOL_SHA256,
            "seal_sha256": seal_sha256,
            "evidence_label": "SIM_GEOMETRIC",
            "records": sorted(records, key=lambda x: x["instance_id"]),
        }
        payload["manifest_sha256"] = canonical_digest(payload)
        (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload
    except Exception:
        # The caller records the failed attempt; never treat a partial tree as a benchmark.
        raise


def load_final_items(root: str | Path, benchmark_root: str | Path) -> tuple[tuple[dict[str, Any], AllocationInstance], ...]:
    benchmark = Path(benchmark_root).resolve()
    manifest_path = benchmark / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.pop("manifest_sha256")
    if canonical_digest(manifest) != expected:
        raise RuntimeError("final manifest hash mismatch")
    manifest["manifest_sha256"] = expected
    items = []
    for record in sorted(manifest["records"], key=lambda x: x["instance_id"]):
        path = (benchmark / record["relative_path"]).resolve()
        if benchmark not in path.parents or "frozen_test" not in path.parts:
            raise PermissionError("final record escaped frozen_test")
        instance = allocation_instance_from_dict(json.loads(path.read_text()))
        if hashlib.sha256(canonical_instance_bytes(instance)).hexdigest() != record["instance_sha256"]:
            raise RuntimeError("final instance hash mismatch")
        items.append((record, instance))
    return tuple(items)


def validation_items(prepared: PreparedPointerPilot, limit_per_cell: int = 1):
    result = []
    counts: dict[str, int] = {}
    for example in prepared.validation_examples:
        if counts.get(example.cell_id, 0) >= limit_per_cell:
            continue
        counts[example.cell_id] = counts.get(example.cell_id, 0) + 1
        record = {"split": "validation", "cell_id": example.cell_id, "task_group_id": example.task_group_id, "variant_index": 0, "instance_id": example.instance.instance_id}
        result.append((record, example.instance, example.graph))
    return tuple(result)


def evaluate_candidates(items, models: FixedModels, prepared: PreparedPointerPilot, context: OracleContext, protocol: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weights = load_pointer_pilot_config(Path(__file__).resolve().parents[3] / "configs/allocation/a3_5_pointer_pilot_v1.json").objective_weights
    solver_protocol = SolverProtocol("a1-solver-protocol-v1", float(protocol["strong_baselines"]["milp_time_limit_s"]), float(protocol["strong_baselines"]["milp_relative_gap"]), 0)
    baselines = {
        "hybrid_assignment_milp": lambda x: solve_hybrid_assignment_milp(x, context, solver_protocol),
        "order_aware_lns": lambda x: solve_order_aware_lns(x, context, iterations=int(protocol["strong_baselines"]["lns_iterations"]), seed=int(protocol["strong_baselines"]["lns_seed"]), objective_weights=weights),
        "hybrid_load_balanced": lambda x: solve_hybrid_load_balanced(x, context),
    }
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for item in items:
        if len(item) == 3:
            record, instance, graph = item
        else:
            record, instance = item
            graph = prepared.normalizer.transform(build_a3_graph(instance, context, prepared.vocabulary, split="validation"))
        for seed, model in models.pointer:
            started = time.perf_counter()
            with torch.no_grad(): rollout = model.greedy_rollout(graph, instance, context)
            check = verify_plan(instance, rollout.plan, context) if rollout.plan is not None else None
            elapsed = time.perf_counter() - started
            method = f"pair_pointer_seed_{seed}"
            rows.append(_row(record, method, seed, rollout.status, check, elapsed, weights, rollout.hard_mask_violations, rollout.atomicity_violations, rollout.dead_end_step))
            raw.append({"instance_id": instance.instance_id, "method": method, "status": rollout.status, "actions": [x.to_dict() for x in rollout.actions], "plan": None if rollout.plan is None else rollout.plan.to_dict()})
        for seed, model in models.static:
            started = time.perf_counter()
            with torch.no_grad():
                output = model(graph)
                candidate = decode_masked_candidate(graph, instance, context, output.assignment_logits, output.order_scores, method_id=f"a3-5-static-seed-{seed}-final")
            check = verify_plan(instance, candidate.plan, context) if candidate.plan is not None else None
            elapsed = time.perf_counter() - started
            method = f"static_seed_{seed}"
            rows.append(_row(record, method, seed, candidate.status, check, elapsed, weights, 0, 0, None))
            raw.append({"instance_id": instance.instance_id, "method": method, "status": candidate.status, "assignment": [list(x) for x in candidate.assignment], "robot_orders": [[r, list(o)] for r, o in candidate.robot_orders], "plan": None if candidate.plan is None else candidate.plan.to_dict()})
        for method, solve in baselines.items():
            started = time.perf_counter(); result = solve(instance)
            check = verify_plan(instance, result.plan, context) if result.plan is not None else None
            elapsed = time.perf_counter() - started
            rows.append(_row(record, method, None, result.status, check, elapsed, weights, 0, 0, None))
            raw.append({"instance_id": instance.instance_id, "method": method, "status": result.status, "plan": None if result.plan is None else result.plan.to_dict()})
    return rows, raw


def audit_witnesses(benchmark_root: str | Path, items, raw_predictions, context: OracleContext) -> tuple[str, ...]:
    benchmark = Path(benchmark_root).resolve()
    predictions = {(x["instance_id"], x["method"]) for x in raw_predictions}
    failures: list[str] = []
    for record, instance in items:
        if not all((instance.instance_id, method) in predictions for method in _all_method_ids()):
            failures.append(f"PREDICTION_MATRIX_INCOMPLETE={instance.instance_id}")
        path = (benchmark / record["witness_relative_path"]).resolve()
        if benchmark not in path.parents or "witnesses" not in path.parts:
            failures.append(f"WITNESS_PATH_ESCAPE={instance.instance_id}"); continue
        raw = json.loads(path.read_text())
        if raw.get("witness_sha256") != record["witness_sha256"]:
            failures.append(f"WITNESS_HASH_MISMATCH={instance.instance_id}"); continue
        plan = allocation_plan_from_dict(raw["plan"])
        if not verify_plan(instance, plan, context).feasible:
            failures.append(f"WITNESS_INVALID={instance.instance_id}")
    return tuple(failures)


def aggregate_final(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], integrity_failures: Sequence[str]) -> Mapping[str, Any]:
    expected = int(protocol["benchmark"]["instances_total"]) * 9
    checks = {
        "complete_method_matrix": len(rows) == expected,
        "all_metrics_finite": all(math.isfinite(float(x["runtime_s"])) for x in rows),
        "zero_hard_mask_violations": sum(int(x["hard_mask_violations"]) for x in rows) == 0,
        "zero_atomicity_violations": sum(int(x["atomicity_violations"]) for x in rows) == 0,
        "zero_integrity_failures": not integrity_failures,
    }
    differences = _group_differences(rows, "pair_pointer", "static")
    analysis = protocol["primary_analysis"]
    observed = float(np.mean(list(differences.values())))
    values = np.asarray(list(differences.values()), dtype=float)
    rng = np.random.default_rng(int(analysis["randomization_seed"]))
    extreme = 0; remaining = int(analysis["randomization_draws"])
    while remaining:
        count = min(10000, remaining)
        null = (rng.choice((-1.0, 1.0), size=(count, len(values))) * values).mean(axis=1)
        extreme += int(np.sum(null >= observed)); remaining -= count
    p_value = (extreme + 1) / (int(analysis["randomization_draws"]) + 1)
    rng = np.random.default_rng(int(analysis["bootstrap_seed"]))
    indices = rng.integers(0, len(values), size=(int(analysis["bootstrap_draws"]), len(values)))
    boot = values[indices].mean(axis=1)
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    seed_differences = {}
    for seed in (101, 211, 307):
        d = _group_differences(rows, f"pair_pointer_seed_{seed}", f"static_seed_{seed}", exact=True)
        seed_differences[str(seed)] = float(np.mean(list(d.values())))
    cell_differences = {cell: float(np.mean([value for group, value in differences.items() if f"-{cell}-" in group])) for cell in EXPECTED_CELLS}
    primary = all(checks.values()) and observed > 0 and p_value < float(analysis["alpha"]) and ci[0] > 0 and sum(v > 0 for v in seed_differences.values()) >= 2
    robustness = all(value >= -float(protocol["secondary_analysis"]["noninferiority_margin"]) * 6 for value in cell_differences.values())
    if not all(checks.values()): result_class = "A3_5_FINAL_INVALID"
    elif not primary: result_class = "A3_5_DECODER_HYPOTHESIS_NOT_SUPPORTED"
    elif not robustness: result_class = "A3_5_DECODER_HYPOTHESIS_SUPPORTED_HETEROGENEOUS"
    else: result_class = "A3_5_DECODER_HYPOTHESIS_SUPPORTED"
    baselines = {method: _method_summary(rows, method) for method in BASELINES}
    method_summary = {method: _method_summary(rows, method) for method in ("pair_pointer", "static")}
    secondary = {}
    for baseline in BASELINES:
        diff = _group_differences(rows, "pair_pointer", baseline)
        arr = np.asarray(list(diff.values()))
        rng = np.random.default_rng(int(analysis["bootstrap_seed"]) + BASELINES.index(baseline) + 1)
        idx = rng.integers(0, len(arr), size=(int(analysis["bootstrap_draws"]), len(arr)))
        bs = arr[idx].mean(axis=1)
        secondary[baseline] = {"mean_difference": float(arr.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    if primary and all(value["ci95"][0] > 0 for value in secondary.values()): result_class = "A3_5_STRONG_BASELINE_ADVANTAGE"
    return {
        "result_class": result_class, "checks": checks,
        "integrity_failures": list(integrity_failures),
        "primary": {"mean_paired_coverage_difference": observed, "one_sided_randomization_p": p_value, "cluster_bootstrap_ci95": ci, "seed_differences": seed_differences, "cell_differences": cell_differences, "supported": primary, "robustness_flag": robustness},
        "methods": method_summary, "strong_baselines": baselines, "secondary_pairwise": secondary,
        "required_wording": protocol["result_classes"][result_class],
    }


def _audit_records(records, protocol):
    expected = int(protocol["benchmark"]["instances_total"])
    if len(records) != expected or len({x["instance_id"] for x in records}) != expected:
        raise RuntimeError("final record count or ID uniqueness failure")
    if len({x["task_group_id"] for x in records}) != int(protocol["benchmark"]["groups_total"]):
        raise RuntimeError("final group count mismatch")
    if {x["cell_id"] for x in records} != set(EXPECTED_CELLS):
        raise RuntimeError("final cell mismatch")
    prefix = str(protocol["benchmark"]["id_prefix"])
    if any(not x["instance_id"].startswith(prefix) or not x["task_group_id"].startswith(prefix) for x in records):
        raise RuntimeError("final namespace mismatch")


def _row(record, method, seed, status, check, runtime, weights, hard_mask, atomicity, dead_end):
    verified = bool(check and check.feasible); terms = {} if check is None else dict(check.objective_terms)
    return {
        "instance_id": record["instance_id"], "split": record["split"], "cell_id": record["cell_id"], "task_group_id": record["task_group_id"], "variant_index": int(record["variant_index"]),
        "method": method, "seed": seed, "status": status, "verified": verified,
        "runtime_s": float(runtime), "hard_mask_violations": int(hard_mask), "atomicity_violations": int(atomicity), "dead_end_step": dead_end,
        "weighted_proxy_score": sum(float(weights.get(k, 0)) * float(v) for k, v in terms.items()) if verified else None,
        "makespan_s": terms.get("makespan"), "load_variance_s2": terms.get("load_variance"), "travel_setup_time_s": terms.get("travel_setup_time"),
        "failure_class": None if verified else _failure_class(status, check),
    }


def _failure_class(status, check):
    if status == "decoder_dead_end": return "decoder_dead_end"
    codes = set() if check is None else {x.code for x in check.violations}
    if "SEGMENT_COVERAGE" in codes: return "incomplete_assignment"
    if "PRECEDENCE" in codes or "PARENT_SEGMENT_ORDER" in codes: return "precedence_failure"
    if "SEGMENT_TIME_WINDOW" in codes or any("TIME" in x for x in codes): return "time_window_failure"
    if "RESOURCE_CAPACITY" in codes or any("RESOURCE" in x for x in codes): return "shared_resource_failure"
    return "schedule_infeasible"


def _all_method_ids():
    return tuple(f"pair_pointer_seed_{s}" for s in (101,211,307)) + tuple(f"static_seed_{s}" for s in (101,211,307)) + BASELINES


def _group_differences(rows, left, right, exact=False):
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        method = str(row["method"])
        family = method if exact else ("pair_pointer" if method.startswith("pair_pointer_seed_") else "static" if method.startswith("static_seed_") else method)
        grouped.setdefault((str(row["task_group_id"]), family), []).append(float(bool(row["verified"])))
    groups = sorted({key[0] for key in grouped})
    return {group: float(np.mean(grouped[(group,left)]) - np.mean(grouped[(group,right)])) for group in groups}


def _method_summary(rows, method):
    selected = [x for x in rows if x["method"] == method or (method == "pair_pointer" and str(x["method"]).startswith("pair_pointer_seed_")) or (method == "static" and str(x["method"]).startswith("static_seed_"))]
    successful = [x for x in selected if x["verified"]]
    return {
        "rows": len(selected), "coverage": float(np.mean([bool(x["verified"]) for x in selected])),
        "median_runtime_s": float(statistics.median(float(x["runtime_s"]) for x in selected)),
        "runtime_iqr_s": [float(np.quantile([x["runtime_s"] for x in selected], .25)), float(np.quantile([x["runtime_s"] for x in selected], .75))],
        "total_runtime_s": float(sum(float(x["runtime_s"]) for x in selected)),
        "conditional_weighted_proxy_score": None if not successful else float(np.mean([x["weighted_proxy_score"] for x in successful])),
    }


def _load_context(root):
    from .oracle import load_oracle_context
    return load_oracle_context(root / "configs/allocation/oracle_proxy_v1.json")
