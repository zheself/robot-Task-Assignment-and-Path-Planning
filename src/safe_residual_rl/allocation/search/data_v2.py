"""One-shot, development-only corpus contract for A4b ordinary-LNS v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..generation import BenchmarkConfig, canonical_instance_bytes, generate_instance, stable_seed
from ..oracle import OracleContext
from ..pointer_pilot import construct_pointer_compatible_witness, load_pointer_pilot_config
from ..schema import EvidenceLabel, allocation_instance_from_dict, allocation_plan_from_dict
from ..verifier import verify_plan
from .trace import canonical_hash

A4B_V2_VERSION = "a4b-ordinary-lns-dev-v2"
ALLOWED_SPLITS = frozenset({"train", "development"})
FORBIDDEN_TOKENS = frozenset(
    {
        "validation", "frozen_test", "stress", "a2v2", "a2v3", "a2v4",
        "a35p1", "a35f1", "a4wsp1", "a4bnlsd1", "benchmark_v2",
        "benchmark_v3", "benchmark_v4", "a4_warm_start_pilot_v1",
        "a4b_neural_lns_dev_v1",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_a4b_v2_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    raw = json.loads(source.read_text())
    if raw.get("version") != A4B_V2_VERSION:
        raise ValueError("wrong A4b v2 protocol version")
    if raw.get("status") != "PREREGISTERED_BEFORE_A4B_V2_DATA_GENERATION":
        raise ValueError("A4b v2 protocol was not frozen before generation")
    if raw["data"]["splits"] != ["train", "development"]:
        raise ValueError("A4b v2 split contract changed")
    if raw["data"]["frozen_or_stress_generation_allowed"]:
        raise ValueError("A4b v2 may not generate frozen or stress data")
    if raw["data"]["id_prefix"] != "a4blnsd2":
        raise ValueError("A4b v2 namespace changed")
    raw["config_sha256"] = sha256_file(source)
    return raw


def guard_a4b_v2_path(path: str | Path) -> None:
    lowered = {item.lower() for item in Path(path).resolve().parts}
    overlap = lowered & FORBIDDEN_TOKENS
    if overlap:
        raise PermissionError(f"A4b v2 forbidden path token: {sorted(overlap)}")


def generate_a4b_v2_data(
    project_root: str | Path,
    config: Mapping[str, Any],
    output_root: str | Path,
    context: OracleContext,
) -> dict[str, Any]:
    project = Path(project_root).resolve()
    output = Path(output_root).resolve()
    guard_a4b_v2_path(output)
    if output.exists():
        raise FileExistsError("A4b v2 corpus generation is immutable and one-shot")

    # This reads generator geometry/cell definitions only. It never opens an
    # A3.5 instance, witness, output, evaluator, or result.
    generator = load_pointer_pilot_config(
        project / "configs/allocation/a3_5_pointer_pilot_v1.json"
    )
    data = config["data"]
    records: list[dict[str, Any]] = []
    output.mkdir(parents=True)
    for split in ("train", "development"):
        generator_split = "validation" if split == "development" else split
        groups = int(data[f"{split}_groups_per_cell"])
        for cell in generator.cells:
            adapter = BenchmarkConfig(
                version="a4b-ordinary-lns-geometric-generator-v2",
                manifest_version="a4b-ordinary-lns-development-manifest-v2",
                master_seed=int(data["master_seed"]),
                evidence_label=EvidenceLabel.SIM_GEOMETRIC,
                coordinate_frame="synthetic_workcell_m",
                variants_per_group=int(data["variants_per_group"]),
                geometry=generator.geometry,
                splits=((generator_split, replace(cell.spec, group_count=groups)),),
                objective_weights=tuple(sorted(config["search"]["objective_weights"].items())),
                baseline_protocol={},
                boundaries=("SIM_GEOMETRIC", "A4b v2 development only", "NO_FROZEN_OR_STRESS"),
            )
            for group_index in range(groups):
                group_id = f"{data['id_prefix']}-{split}-{cell.cell_id}-group-{group_index:03d}"
                seed = stable_seed(
                    int(data["master_seed"]), split, cell.cell_id, group_index,
                    "a4b-ordinary-lns-v2-group",
                )
                for variant in range(int(data["variants_per_group"])):
                    generated = generate_instance(adapter, generator_split, group_id, seed, variant)
                    witness = construct_pointer_compatible_witness(
                        generated.instance,
                        context,
                        tight_pre_margin_duration=float(generator.geometry["witness_tight_pre_margin_duration"]),
                        tight_post_margin_duration=float(generator.geometry["witness_tight_post_margin_duration"]),
                        loose_pre_margin_s=float(generator.geometry["witness_loose_pre_margin_s"]),
                    )
                    instance = witness.instance
                    relative = Path("data") / split / cell.cell_id / f"{instance.instance_id}.json"
                    witness_relative = Path("witnesses") / split / cell.cell_id / f"{instance.instance_id}.json"
                    (output / relative).parent.mkdir(parents=True, exist_ok=True)
                    (output / witness_relative).parent.mkdir(parents=True, exist_ok=True)
                    (output / relative).write_text(json.dumps(instance.to_dict(), indent=2, sort_keys=True))
                    (output / witness_relative).write_text(json.dumps(witness.to_dict(), indent=2, sort_keys=True))
                    records.append(
                        {
                            "split": split,
                            "cell_id": cell.cell_id,
                            "task_group_id": group_id,
                            "variant_index": variant,
                            "instance_id": instance.instance_id,
                            "workpiece_id": instance.workpiece_id,
                            "layout_id": instance.layout_id,
                            "parent_curve_ids": sorted({item.parent_curve_id for item in instance.segments}),
                            "relative_path": relative.as_posix(),
                            "witness_relative_path": witness_relative.as_posix(),
                            "instance_sha256": hashlib.sha256(canonical_instance_bytes(instance)).hexdigest(),
                            "witness_sha256": witness.witness_sha256,
                            "evidence_label": "SIM_GEOMETRIC",
                        }
                    )
    audit_a4b_v2_records(records, config)
    payload = {
        "version": "a4b-ordinary-lns-development-manifest-v2",
        "config_sha256": config["config_sha256"],
        "records": sorted(records, key=lambda item: item["instance_id"]),
        "forbidden_splits_generated": [],
    }
    payload["manifest_sha256"] = canonical_hash(payload)
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def audit_a4b_v2_records(records, config: Mapping[str, Any]) -> None:
    data = config["data"]
    expected = int(data["train_instances_total"]) + int(data["development_instances_total"])
    if len(records) != expected or len({item["instance_id"] for item in records}) != expected:
        raise RuntimeError("A4b v2 instance count or uniqueness failure")
    expected_groups = int(data["train_groups_total"]) + int(data["development_groups_total"])
    if len({item["task_group_id"] for item in records}) != expected_groups:
        raise RuntimeError("A4b v2 task-group count failure")
    for item in records:
        if item["split"] not in ALLOWED_SPLITS or item["evidence_label"] != "SIM_GEOMETRIC":
            raise RuntimeError("A4b v2 split or evidence-label escape")
        identifiers = [
            item["instance_id"], item["task_group_id"], item["workpiece_id"],
            item["layout_id"], *item["parent_curve_ids"],
        ]
        if any(data["id_prefix"] not in identifier for identifier in identifiers):
            raise RuntimeError("A4b v2 identifier namespace escape")
        if any(old in identifier for old in data["disjoint_id_prefixes"] for identifier in identifiers):
            raise RuntimeError("A4b v2 old benchmark identifier overlap")
    for field in ("task_group_id", "workpiece_id", "layout_id"):
        train = {item[field] for item in records if item["split"] == "train"}
        development = {item[field] for item in records if item["split"] == "development"}
        if train & development:
            raise RuntimeError(f"A4b v2 split leakage in {field}")
    train_parents = {p for item in records if item["split"] == "train" for p in item["parent_curve_ids"]}
    development_parents = {p for item in records if item["split"] == "development" for p in item["parent_curve_ids"]}
    if train_parents & development_parents:
        raise RuntimeError("A4b v2 parent-curve split leakage")


def load_a4b_v2_items(
    data_root: str | Path,
    split: str,
    context: OracleContext,
    *,
    audit_witness: bool = True,
):
    if split not in ALLOWED_SPLITS:
        raise PermissionError(f"A4b v2 forbids split {split}")
    root = Path(data_root).resolve()
    guard_a4b_v2_path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = manifest.pop("manifest_sha256")
    if canonical_hash(manifest) != expected or manifest["forbidden_splits_generated"]:
        raise RuntimeError("A4b v2 manifest integrity failure")
    rows = []
    for record in manifest["records"]:
        if record["split"] != split:
            continue
        path = (root / record["relative_path"]).resolve()
        guard_a4b_v2_path(path)
        instance = allocation_instance_from_dict(json.loads(path.read_text()))
        if hashlib.sha256(canonical_instance_bytes(instance)).hexdigest() != record["instance_sha256"]:
            raise RuntimeError("A4b v2 instance hash mismatch")
        if audit_witness:
            witness_path = (root / record["witness_relative_path"]).resolve()
            guard_a4b_v2_path(witness_path)
            witness = json.loads(witness_path.read_text())
            if witness["witness_sha256"] != record["witness_sha256"]:
                raise RuntimeError("A4b v2 witness hash mismatch")
            plan = allocation_plan_from_dict(witness["plan"], instance)
            if not verify_plan(instance, plan, context).feasible:
                raise RuntimeError("A4b v2 witness verifier failure")
        rows.append((record, instance))
    if not rows:
        raise ValueError(f"no A4b v2 records for split {split}")
    return tuple(sorted(rows, key=lambda item: item[0]["instance_id"])), expected
