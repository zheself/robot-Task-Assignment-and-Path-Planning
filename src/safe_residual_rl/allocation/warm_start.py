"""A4a data, immutable initializer and evaluation utilities.

This module rejects every frozen/stress path.  A4a is development-only and
does not import or invoke either sealed evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .decoding import decode_masked_candidate
from .generation import BenchmarkConfig, canonical_instance_bytes, generate_instance, stable_seed
from .graphs import A3Graph, FeatureNormalizer, FeatureVocabulary, build_a3_graph, fit_feature_normalizer, fit_feature_vocabulary
from .models import A3AllocationModel
from .oracle import OracleContext
from .pointer_decoder import FeasiblePairPointer
from .pointer_pilot import construct_pointer_compatible_witness
from .repair import InitializerState, evaluate_state, identical_repair, state_from_plan
from .schema import AllocationInstance, EvidenceLabel, allocation_instance_from_dict
from .solvers import SolverProtocol, solve_hybrid_assignment_milp, solve_hybrid_load_balanced, solve_order_aware_lns
from .solvers.common import allocation_units, edge_mask_and_costs
from .verifier import verify_plan

A4_VERSION = "a4-warm-start-pilot-v1"
FORBIDDEN = frozenset({"frozen_test", "stress", "a35f1", "a3_5_sealed_final_v1", "benchmark_v4"})


def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_a4_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = json.loads(path.read_text())
    if raw.get("version") != A4_VERSION or raw.get("status") != "PREREGISTERED_BEFORE_A4_DATA_GENERATION":
        raise ValueError("unexpected or unfrozen A4a protocol")
    if raw["data"]["splits"] != ["train", "validation"] or raw["data"]["forbidden_splits"] != ["frozen_test", "stress"]:
        raise ValueError("A4a split guard changed")
    raw["config_sha256"] = sha256_file(path)
    return raw


def export_locked_preprocessor(
    project_root: str | Path,
    legacy_pilot_root: str | Path,
    output_path: str | Path,
    context: OracleContext,
    expected_vocabulary_sha256: str,
    expected_normalizer_sha256: str,
) -> dict[str, Any]:
    """Recover missing inference metadata from A3.5 *train only*.

    Older checkpoints stored model weights but not the fitted feature objects.
    This one-time export is a checkpoint-completion operation, not A4 fitting.
    Any validation/frozen/stress record aborts the export.
    """
    source = Path(legacy_pilot_root).resolve()
    if any(token in source.parts for token in ("frozen_test", "stress")):
        raise PermissionError("forbidden preprocessing source")
    manifest = json.loads((source / "manifest.json").read_text())
    records = [item for item in manifest["records"] if item["split"] == "train"]
    if len(records) != 96 or any(item["split"] != "train" for item in records):
        raise RuntimeError("locked preprocessor export must use exactly A3.5 train")
    instances = []
    accessed = []
    for item in sorted(records, key=lambda x: x["instance_id"]):
        relative = Path(item["relative_path"])
        if set(relative.parts) & FORBIDDEN or "validation" in relative.parts:
            raise PermissionError("non-train preprocessing record rejected")
        path = (Path(project_root).resolve() / relative).resolve()
        instance = allocation_instance_from_dict(json.loads(path.read_text()))
        instances.append(instance)
        accessed.append({"split": "train", "instance_id": instance.instance_id, "path": relative.as_posix()})
    vocabulary = fit_feature_vocabulary(instances, split="train")
    graphs = [build_a3_graph(instance, context, vocabulary, split="train") for instance in instances]
    normalizer = fit_feature_normalizer(graphs, split="train")
    if vocabulary.sha256 != expected_vocabulary_sha256 or normalizer.sha256 != expected_normalizer_sha256:
        raise RuntimeError("recovered frozen preprocessing hash mismatch")
    payload = {
        "version": "a3-5-immutable-inference-preprocessing-v1",
        "purpose": "complete immutable checkpoint inference metadata; never A4 fitting",
        "accessed_splits": ["train"],
        "forbidden_splits_accessed": [],
        "vocabulary": vocabulary.to_dict(),
        "normalizer": normalizer.to_dict(),
        "access_log_sha256": canonical_digest(accessed),
        "access_count": len(accessed),
    }
    payload["artifact_sha256"] = canonical_digest(payload)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def load_locked_preprocessor(path: str | Path) -> tuple[FeatureVocabulary, FeatureNormalizer, str]:
    raw = json.loads(Path(path).read_text())
    expected = raw.pop("artifact_sha256")
    if canonical_digest(raw) != expected or raw["accessed_splits"] != ["train"] or raw["forbidden_splits_accessed"]:
        raise RuntimeError("immutable preprocessing artifact integrity failure")
    v = raw["vocabulary"]
    vocabulary = FeatureVocabulary(tuple(v["capabilities"]), tuple(v["tools"]), tuple(v["kinematic_models"]), v["fit_split"], int(v["fit_instance_count"]), v["sha256"])
    n = raw["normalizer"]
    normalizer = FeatureNormalizer(n["version"], n["fit_split"], int(n["fit_graph_count"]), tuple(sorted((k, tuple(float(x) for x in values)) for k, values in n["means"].items())), tuple(sorted((k, tuple(float(x) for x in values)) for k, values in n["scales"].items())), float(n["epsilon"]), n["sha256"])
    return vocabulary, normalizer, expected


def generate_a4_data(root: str | Path, config: Mapping[str, Any], output_root: str | Path, context: OracleContext) -> dict[str, Any]:
    root, output = Path(root).resolve(), Path(output_root).resolve()
    _guard_path(output)
    if output.exists():
        raise FileExistsError("A4a data already exist")
    from .pointer_pilot import load_pointer_pilot_config
    pilot = load_pointer_pilot_config(root / "configs/allocation/a3_5_pointer_pilot_v1.json")
    data = config["data"]
    output.mkdir(parents=True)
    records = []
    try:
        for split in ("train", "validation"):
            groups = int(data[f"{split}_groups_per_cell"])
            for cell in pilot.cells:
                adapter = BenchmarkConfig(
                    version="a4-warm-start-geometric-generator-v1",
                    manifest_version="a4-warm-start-manifest-v1",
                    master_seed=int(data["master_seed"]),
                    evidence_label=EvidenceLabel.SIM_GEOMETRIC,
                    coordinate_frame="synthetic_workcell_m",
                    variants_per_group=int(data["variants_per_group"]),
                    geometry=pilot.geometry,
                    splits=((split, replace(cell.spec, group_count=groups)),),
                    objective_weights=tuple(sorted(config["repair"]["objective_weights"].items())),
                    baseline_protocol={},
                    boundaries=("SIM_GEOMETRIC", "A4a development only"),
                )
                for group_index in range(groups):
                    group_id = f"{data['id_prefix']}-{split}-{cell.cell_id}-group-{group_index:03d}"
                    seed = stable_seed(int(data["master_seed"]), split, cell.cell_id, group_index, "a4-warm-start-group")
                    for variant in range(int(data["variants_per_group"])):
                        generated = generate_instance(adapter, split, group_id, seed, variant)
                        witness = construct_pointer_compatible_witness(
                            generated.instance, context,
                            tight_pre_margin_duration=float(pilot.geometry["witness_tight_pre_margin_duration"]),
                            tight_post_margin_duration=float(pilot.geometry["witness_tight_post_margin_duration"]),
                            loose_pre_margin_s=float(pilot.geometry["witness_loose_pre_margin_s"]),
                        )
                        instance = witness.instance
                        if not instance.instance_id.startswith(data["id_prefix"]):
                            raise RuntimeError("A4a namespace escape")
                        rel = Path("data") / split / cell.cell_id / f"{instance.instance_id}.json"
                        wrel = Path("witnesses") / split / cell.cell_id / f"{instance.instance_id}.json"
                        (output / rel).parent.mkdir(parents=True, exist_ok=True)
                        (output / wrel).parent.mkdir(parents=True, exist_ok=True)
                        (output / rel).write_text(json.dumps(instance.to_dict(), indent=2, sort_keys=True))
                        (output / wrel).write_text(json.dumps(witness.to_dict(), indent=2, sort_keys=True))
                        records.append({
                            "split": split, "cell_id": cell.cell_id, "task_group_id": group_id,
                            "variant_index": variant, "instance_id": instance.instance_id,
                            "workpiece_id": instance.workpiece_id, "layout_id": instance.layout_id,
                            "parent_curve_ids": sorted({x.parent_curve_id for x in instance.segments}),
                            "relative_path": rel.as_posix(), "witness_relative_path": wrel.as_posix(),
                            "instance_sha256": hashlib.sha256(canonical_instance_bytes(instance)).hexdigest(),
                            "witness_sha256": witness.witness_sha256, "evidence_label": "SIM_GEOMETRIC",
                        })
        _audit_new_records(records, config)
        payload = {"version": "a4-warm-start-manifest-v1", "config_sha256": config["config_sha256"], "records": sorted(records, key=lambda x: x["instance_id"])}
        payload["manifest_sha256"] = canonical_digest(payload)
        (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        return payload
    except Exception:
        raise


def load_a4_items(data_root: str | Path, split: str, context: OracleContext, *, audit_witness: bool = True):
    if split not in {"train", "validation"}:
        raise PermissionError(f"A4a forbids split {split}")
    root = Path(data_root).resolve()
    _guard_path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = manifest.pop("manifest_sha256")
    if canonical_digest(manifest) != expected:
        raise RuntimeError("A4a manifest hash mismatch")
    rows = []
    for record in manifest["records"]:
        if record["split"] != split:
            continue
        path = (root / record["relative_path"]).resolve()
        _guard_path(path)
        instance = allocation_instance_from_dict(json.loads(path.read_text()))
        if hashlib.sha256(canonical_instance_bytes(instance)).hexdigest() != record["instance_sha256"]:
            raise RuntimeError("A4a instance hash mismatch")
        if audit_witness:
            witness = json.loads((root / record["witness_relative_path"]).read_text())
            if witness.get("witness_sha256") != record["witness_sha256"]:
                raise RuntimeError("A4a witness hash mismatch")
            from .schema import allocation_plan_from_dict
            if not verify_plan(instance, allocation_plan_from_dict(witness["plan"], instance), context).feasible:
                raise RuntimeError("A4a witness verifier failure")
        rows.append((record, instance))
    if not rows:
        raise ValueError(f"no A4a {split} data")
    return tuple(sorted(rows, key=lambda x: x[0]["instance_id"])), expected


@dataclass(frozen=True)
class FixedInitializers:
    pointer: tuple[tuple[int, FeasiblePairPointer], ...]
    static: tuple[tuple[int, A3AllocationModel], ...]


def load_fixed_initializers(root: str | Path, checkpoint_registry: Mapping[str, Any], template: A3Graph) -> FixedInitializers:
    root = Path(root)
    pilot = json.loads((root / "configs/allocation/a3_5_pointer_pilot_v1.json").read_text())
    cfg = pilot["models"]
    pointer, static = [], []
    for key, destination in (("pair_pointer", pointer), ("matched_static", static)):
        item = checkpoint_registry["fixed_neural_methods"][key]
        for seed in item["seeds"]:
            path = root / item["checkpoint_files"][str(seed)]
            if sha256_file(path) != item["checkpoint_file_sha256"][str(seed)]:
                raise RuntimeError("checkpoint file hash mismatch")
            if key == "pair_pointer":
                model = FeasiblePairPointer(template, encoder_family="hetero_gnn", hidden_dim=int(cfg["hidden_dim"]), layers=int(cfg["message_passing_layers"]), heads=int(cfg["attention_heads"]), dropout=float(cfg["dropout"]))
            else:
                model = A3AllocationModel(template, family="hetero_gnn", hidden_dim=int(cfg["hidden_dim"]), layers=int(cfg["message_passing_layers"]), heads=int(cfg["attention_heads"]), dropout=float(cfg["dropout"]))
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            model.eval()
            destination.append((int(seed), model))
    return FixedInitializers(tuple(pointer), tuple(static))


def initialize_all(instance, graph, models, context, config):
    units = allocation_units(instance)
    outputs = {}
    for seed, model in models.pointer:
        started = time.perf_counter()
        with torch.no_grad():
            rollout = model.greedy_rollout(graph, instance, context)
        state = _state_from_pointer_actions(instance, rollout.actions)
        outputs[f"pair_pointer_seed_{seed}"] = (state, time.perf_counter() - started, rollout.status)
    for seed, model in models.static:
        started = time.perf_counter()
        with torch.no_grad():
            out = model(graph)
            candidate = decode_masked_candidate(graph, instance, context, out.assignment_logits, out.order_scores, method_id=f"a4-static-{seed}")
        state = _state_from_static(instance, candidate.assignment, candidate.robot_orders)
        outputs[f"static_seed_{seed}"] = (state, time.perf_counter() - started, candidate.status)
    started = time.perf_counter(); result = solve_hybrid_load_balanced(instance, context)
    outputs["hybrid_load_balanced"] = ((state_from_plan(instance, result.plan) if result.plan else _load_balanced_state(instance, context)), time.perf_counter() - started, result.status)
    started = time.perf_counter(); result = solve_hybrid_assignment_milp(instance, context, SolverProtocol("a1-solver-protocol-v1", float(config["repair"]["milp_initializer_time_limit_s"]), 0.0, 0))
    outputs["hybrid_assignment_milp"] = ((state_from_plan(instance, result.plan) if result.plan else _load_balanced_state(instance, context)), time.perf_counter() - started, result.status)
    robots = tuple(sorted(x.id for x in instance.robots))
    outputs["cold_start"] = (InitializerState(tuple(None for _ in units), tuple((r, ()) for r in robots)), 0.0, "empty")
    return outputs


def evaluate_raw_references(instance, graph, models, context, config, weights):
    rows = []
    initializers = initialize_all(instance, graph, models, context, config)
    for name, (state, runtime, status) in initializers.items():
        if name == "cold_start":
            continue
        evaluation = evaluate_state(instance, context, state, weights, method_id=f"a4-{name}-raw")
        rows.append(_raw_row(name, status, runtime, evaluation))
    started = time.perf_counter()
    lns = solve_order_aware_lns(instance, context, iterations=int(config["repair"]["original_lns_iterations"]), seed=0, objective_weights=weights)
    runtime = time.perf_counter() - started
    checked = verify_plan(instance, lns.plan, context) if lns.plan else None
    rows.append({"initializer": "order_aware_lns", "status": lns.status, "verified": bool(checked and checked.feasible), "objective": lns.objective_value, "initializer_runtime_s": runtime, "failure_reason": None if checked and checked.feasible else "schedule_infeasible"})
    return initializers, rows


def dependency_versions():
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "scipy": scipy.__version__, "platform": platform.platform(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}


def _state_from_pointer_actions(instance, actions):
    units = allocation_units(instance)
    lookup = {"+".join(unit): i for i, unit in enumerate(units)}
    assignments: list[str | None] = [None] * len(units)
    orders = {r.id: [] for r in sorted(instance.robots, key=lambda x: x.id)}
    for action in actions:
        index = lookup[action.unit_id]
        assignments[index] = action.robot_id
        orders[action.robot_id].append(index)
    return InitializerState(tuple(assignments), tuple((r, tuple(orders[r])) for r in sorted(orders)))


def _state_from_static(instance, assignment, robot_orders):
    units = allocation_units(instance)
    mapping = dict(assignment)
    values = tuple(mapping.get(unit[0]) if len({mapping.get(x) for x in unit}) == 1 else None for unit in units)
    segment_to_unit = {s: i for i, unit in enumerate(units) for s in unit}
    orders = []
    for robot, segments in robot_orders:
        order = []
        for segment in segments:
            index = segment_to_unit[segment]
            if index not in order:
                order.append(index)
        orders.append((robot, tuple(order)))
    return InitializerState(values, tuple(orders))


def _load_balanced_state(instance, context):
    _, units, robots, costs = edge_mask_and_costs(instance, context)
    loads = {r: 0.0 for r in robots}; assignment = []; orders = {r: [] for r in robots}
    for i, row in enumerate(costs):
        _, robot, value = min((loads[r] + row[j], r, row[j]) for j, r in enumerate(robots) if math.isfinite(row[j]))
        assignment.append(robot); loads[robot] += value; orders[robot].append(i)
    return InitializerState(tuple(assignment), tuple((r, tuple(orders[r])) for r in robots))


def _raw_row(name, status, runtime, evaluation):
    return {"initializer": name, "status": status, "verified": evaluation.verified, "objective": evaluation.objective, "initializer_runtime_s": runtime, "failure_reason": evaluation.failure_reason}


def _audit_new_records(records, config):
    data = config["data"]
    if len(records) != int(data["train_instances_total"]) + int(data["validation_instances_total"]):
        raise RuntimeError("A4a record count mismatch")
    if len({x["instance_id"] for x in records}) != len(records):
        raise RuntimeError("A4a duplicate instance ID")
    if len({x["task_group_id"] for x in records}) != int(data["train_groups_total"]) + int(data["validation_groups_total"]):
        raise RuntimeError("A4a group count mismatch")
    ids = [x["instance_id"] for x in records] + [x["task_group_id"] for x in records]
    if any(not x.startswith(data["id_prefix"]) or any(old in x for old in ("a2v2", "a2v3", "a2v4", "a35p1", "a35f1")) for x in ids):
        raise RuntimeError("A4a namespace overlap")
    train_groups = {x["task_group_id"] for x in records if x["split"] == "train"}
    validation_groups = {x["task_group_id"] for x in records if x["split"] == "validation"}
    if train_groups & validation_groups:
        raise RuntimeError("A4a split leakage")


def _guard_path(path: Path):
    lowered = {x.lower() for x in path.parts}
    if lowered & FORBIDDEN:
        raise PermissionError(f"A4a forbidden path token: {sorted(lowered & FORBIDDEN)}")
