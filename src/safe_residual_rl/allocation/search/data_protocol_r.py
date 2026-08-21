"""Fresh, guarded Protocol-R corpus and deterministic challenge construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..generation import (
    BenchmarkConfig,
    SplitGenerationSpec,
    canonical_instance_bytes,
    generate_instance,
    stable_seed,
)
from ..oracle import OracleContext
from ..pointer_pilot import construct_pointer_compatible_witness
from ..schema import (
    AllocationInstance,
    EvidenceLabel,
    TimeWindow,
    allocation_instance_from_dict,
    allocation_plan_from_dict,
)
from ..solvers.common import allocation_units
from ..verifier import verify_plan
from .anytime import build_hybrid_load_balanced_initializer
from .diagnostics import evaluate_state_timed
from .trace import canonical_hash

PROTOCOL_R_VERSION = "a4b-protocol-r-freeze-candidate-v1"
ALLOWED_SPLITS = frozenset({"train", "development"})
FORBIDDEN_TOKENS = frozenset(
    {
        "validation",
        "frozen_test",
        "stress",
        "a2v2",
        "a2v3",
        "a2v4",
        "a35p1",
        "a35f1",
        "a4wsp1",
        "a4bnlsd1",
        "a4blnsd2",
        "a4_warm_start_pilot_v1",
        "a4b_neural_lns_dev_v1",
        "a4b_ordinary_lns_dev_v2",
    }
)


@dataclass(frozen=True)
class ChallengeVariant:
    instance: AllocationInstance
    witness_plan: object
    eligible: bool
    failure_reason: str | None
    tightened_unit_indices: tuple[int, ...]
    attempt: int
    audit: Mapping[str, object]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol_r_config(
    path: str | Path, *, allow_draft: bool = False
) -> dict[str, Any]:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("version") != PROTOCOL_R_VERSION:
        raise ValueError("unexpected Protocol-R version")
    if raw.get("protocol_id") != "a4b_ordinary_search_recovery_v3":
        raise ValueError("Protocol-R namespace changed")
    if raw.get("formal_action") != "HOLD_A4B_LEARNED_DESTROY_TRAINING":
        raise ValueError("Protocol-R HOLD boundary changed")
    if raw.get("status") != "DRAFT_FOR_REVIEW_NOT_FROZEN_NOT_AUTHORIZED":
        if raw.get("status") != "FROZEN_BEFORE_PROTOCOL_R_EXECUTION":
            raise ValueError("unknown Protocol-R status")
    elif not allow_draft:
        raise PermissionError("draft Protocol R cannot generate data or execute search")
    data = raw["data"]
    if data["id_prefix"] != "a4blnsr3" or data["splits"] != ["train", "development"]:
        raise ValueError("Protocol-R data contract changed")
    if data["frozen_or_stress_generation_allowed"]:
        raise ValueError("Protocol R cannot generate frozen or stress data")
    generator = data["generator"]
    parameter_source = source.parents[2] / generator["parameter_source_path"]
    if sha256_file(parameter_source) != generator["parameter_source_sha256"]:
        raise RuntimeError("generator parameter source hash drift")
    expected_cells = set(data["cells"])
    if set(generator["cell_specs"]) != expected_cells:
        raise ValueError("generator cell matrix is incomplete")
    if any(value is not None for value in raw["seal_time_required_fields"].values()):
        if raw["status"] != "FROZEN_BEFORE_PROTOCOL_R_EXECUTION":
            raise ValueError("draft Protocol R may not contain partial seal values")
    raw["config_sha256"] = sha256_file(source)
    return raw


def require_execution_ready(config: Mapping[str, Any]) -> None:
    if config.get("status") != "FROZEN_BEFORE_PROTOCOL_R_EXECUTION":
        raise PermissionError("Protocol R is not frozen; execution is fail-closed")
    seals = config.get("seal_time_required_fields", {})
    if not seals or any(value in (None, "") for value in seals.values()):
        raise PermissionError("Protocol R seal is incomplete")


def guard_protocol_r_path(path: str | Path) -> None:
    lowered = {item.lower() for item in Path(path).resolve().parts}
    overlap = lowered & FORBIDDEN_TOKENS
    if overlap:
        raise PermissionError(f"Protocol-R forbidden path token: {sorted(overlap)}")


def _spec(raw: Mapping[str, Any], group_count: int) -> SplitGenerationSpec:
    return SplitGenerationSpec(
        group_count=group_count,
        robot_count=tuple(int(item) for item in raw["robots"]),  # type: ignore[arg-type]
        segment_count=tuple(int(item) for item in raw["segments"]),  # type: ignore[arg-type]
        max_segments_per_curve=int(raw["max_segments_per_curve"]),
        precedence_probability=float(raw["precedence_probability"]),
        shared_resource_probability=float(raw["shared_resource_probability"]),
        tight_window_probability=float(raw["tight_window_probability"]),
    )


def _adapter(
    config: Mapping[str, Any], cell: str, split: str, group_count: int
) -> BenchmarkConfig:
    data = config["data"]
    generator = data["generator"]
    generator_split = "validation" if split == "development" else "train"
    return BenchmarkConfig(
        version="a4b-protocol-r-geometric-generator-v1",
        manifest_version="a4b-protocol-r-development-manifest-v1",
        master_seed=int(data["master_seed"]),
        evidence_label=EvidenceLabel.SIM_GEOMETRIC,
        coordinate_frame=str(generator["coordinate_frame"]),
        variants_per_group=int(data["variants_per_group"]),
        geometry=dict(generator["geometry"]),
        splits=((generator_split, _spec(generator["cell_specs"][cell], group_count)),),
        objective_weights=tuple(
            sorted((str(k), float(v)) for k, v in config["search"]["objective_weights"].items())
        ),
        baseline_protocol={},
        boundaries=("SIM_GEOMETRIC", "PROTOCOL_R_DEVELOPMENT_ONLY", "NO_FROZEN_OR_STRESS"),
    )


def _normalize_ids(
    instance: AllocationInstance, config: Mapping[str, Any], group_id: str, variant: int
) -> AllocationInstance:
    templates = config["data"]["id_templates"]
    expected_instance = templates["instance"].format(
        task_group_id=group_id, variant=variant
    )
    if instance.instance_id != expected_instance:
        raise RuntimeError("generator instance ID escaped Protocol-R template")
    return replace(
        instance,
        workpiece_id=templates["workpiece"].format(task_group_id=group_id),
        layout_id=templates["layout"].format(task_group_id=group_id),
    )


def build_regular_variant(
    config: Mapping[str, Any], cell: str, split: str, group_index: int, variant: int
) -> AllocationInstance:
    if split not in ALLOWED_SPLITS:
        raise PermissionError(f"Protocol R forbids split {split}")
    data = config["data"]
    template = data["id_templates"][
        "regular_train_group" if split == "train" else "development_group"
    ]
    group_id = template.format(cell=cell, index=group_index)
    adapter = _adapter(config, cell, split, 1)
    generator_split = "validation" if split == "development" else "train"
    unique_index = stable_seed(
        int(data["master_seed"]), split, cell, group_index, "protocol-r-regular-group"
    )
    generated = generate_instance(
        adapter, generator_split, group_id, unique_index, variant
    )
    return _normalize_ids(generated.instance, config, group_id, variant)


def _finish_by_unit(instance: AllocationInstance, plan) -> dict[int, float]:
    units = allocation_units(instance)
    scheduled = {item.segment_id: item for item in plan.schedule}
    return {
        index: max(scheduled[segment].planned_end_s for segment in unit)
        for index, unit in enumerate(units)
    }


def make_initializer_failure_challenge(
    instance: AllocationInstance,
    witness_plan,
    context: OracleContext,
    weights: Mapping[str, float],
    *,
    attempt: int = 0,
) -> ChallengeVariant:
    """Tighten only window ends until the unchanged initializer fails."""
    witness_check = verify_plan(instance, witness_plan, context)
    if not witness_check.feasible:
        return ChallengeVariant(
            instance, witness_plan, False, "witness_infeasible", (), attempt, {}
        )
    initial = build_hybrid_load_balanced_initializer(instance, context, weights)
    if not initial.provenance.verifier_feasible:
        eligible = initial.provenance.verifier_failure_reason == "time_window_failure"
        return ChallengeVariant(
            instance,
            witness_plan,
            eligible,
            initial.provenance.verifier_failure_reason,
            (),
            attempt,
            {
                "base_already_failed": True,
                "initializer": initial.provenance.to_dict(),
                "witness_plan_sha256": canonical_hash(witness_plan.to_dict()),
            },
        )
    evaluated = evaluate_state_timed(instance, context, initial.state, weights)
    hybrid_plan = evaluated.evaluation.plan
    if hybrid_plan is None:
        return ChallengeVariant(
            instance, witness_plan, False, evaluated.evaluation.failure_reason, (), attempt, {}
        )
    witness_finish = _finish_by_unit(instance, witness_plan)
    hybrid_finish = _finish_by_unit(instance, hybrid_plan)
    units = allocation_units(instance)
    ranked = sorted(
        (
            (hybrid_finish[index] - witness_finish[index], units[index][0], index)
            for index in range(len(units))
            if hybrid_finish[index] - witness_finish[index] > 0.0
        ),
        key=lambda item: (-item[0], item[1]),
    )
    original_segments = {item.id: item for item in instance.segments}
    for length in range(1, len(ranked) + 1):
        selected = tuple(item[2] for item in ranked[:length])
        selected_segments = {
            segment for unit_index in selected for segment in units[unit_index]
        }
        changed = []
        for segment in instance.segments:
            if segment.id not in selected_segments:
                changed.append(segment)
                continue
            unit_index = next(
                index for index in selected if segment.id in units[index]
            )
            midpoint = 0.5 * (
                witness_finish[unit_index] + hybrid_finish[unit_index]
            )
            end = min(original_segments[segment.id].time_window.end_s, midpoint)
            end = max(witness_finish[unit_index], end)
            if end < segment.time_window.start_s:
                break
            changed.append(
                replace(segment, time_window=TimeWindow(segment.time_window.start_s, end))
            )
        if len(changed) != len(instance.segments):
            continue
        candidate = replace(instance, segments=tuple(changed))
        if not verify_plan(candidate, witness_plan, context).feasible:
            continue
        outcome = build_hybrid_load_balanced_initializer(candidate, context, weights)
        if (
            not outcome.provenance.verifier_feasible
            and outcome.provenance.verifier_failure_reason == "time_window_failure"
        ):
            return ChallengeVariant(
                candidate,
                witness_plan,
                True,
                "time_window_failure",
                selected,
                attempt,
                {
                    "base_already_failed": False,
                    "ranked_unit_indices": [item[2] for item in ranked],
                    "tightened_unit_indices": list(selected),
                    "witness_plan_sha256": canonical_hash(witness_plan.to_dict()),
                    "hybrid_plan_sha256": canonical_hash(hybrid_plan.to_dict()),
                    "initializer": outcome.provenance.to_dict(),
                },
            )
    return ChallengeVariant(
        instance,
        witness_plan,
        False,
        "challenge_not_constructed",
        (),
        attempt,
        {"ranked_unit_indices": [item[2] for item in ranked]},
    )


def build_challenge_group(
    config: Mapping[str, Any],
    cell: str,
    group_slot: int,
    context: OracleContext,
) -> tuple[ChallengeVariant, ...]:
    """Return the first two-variant eligible attempt or fail after the cap."""
    data = config["data"]
    challenge = config["challenge_generation"]
    if cell not in data["challenge_cells"]:
        raise ValueError("challenge requested for a non-challenge cell")
    group_id = data["id_templates"]["challenge_train_group"].format(
        cell=cell, index=group_slot
    )
    max_attempts = int(challenge["max_base_attempts_per_group"])
    geometry = data["generator"]["geometry"]
    weights = config["search"]["objective_weights"]
    for attempt in range(max_attempts):
        group_attempt_seed = stable_seed(
            int(data["master_seed"]), "challenge", cell, group_slot, attempt
        )
        adapter = _adapter(config, cell, "train", 1)
        adapter = replace(adapter, master_seed=group_attempt_seed)
        variants = []
        for variant in range(int(data["variants_per_group"])):
            generated = generate_instance(adapter, "train", group_id, 0, variant)
            base = _normalize_ids(generated.instance, config, group_id, variant)
            witness = construct_pointer_compatible_witness(
                base,
                context,
                tight_pre_margin_duration=float(
                    geometry["witness_tight_pre_margin_duration"]
                ),
                tight_post_margin_duration=float(
                    geometry["witness_tight_post_margin_duration"]
                ),
                loose_pre_margin_s=float(geometry["witness_loose_pre_margin_s"]),
            )
            variants.append(
                make_initializer_failure_challenge(
                    witness.instance,
                    witness.plan,
                    context,
                    weights,
                    attempt=attempt,
                )
            )
        if len(variants) == 2 and all(item.eligible for item in variants):
            return tuple(variants)
    raise RuntimeError(
        f"Protocol-R challenge attempt cap exhausted: {cell}/{group_slot}"
    )


def audit_protocol_r_records(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> None:
    data = config["data"]
    expected_instances = int(data["train_instances_total"]) + int(
        data["development_instances_total"]
    )
    if len(records) != expected_instances:
        raise RuntimeError("Protocol-R instance count failure")
    identities = [str(item["instance_id"]) for item in records]
    if len(set(identities)) != expected_instances:
        raise RuntimeError("Protocol-R duplicate instance identity")
    expected_groups = int(data["train_groups_total"]) + int(
        data["development_groups_total"]
    )
    if len({item["task_group_id"] for item in records}) != expected_groups:
        raise RuntimeError("Protocol-R task-group count failure")
    for item in records:
        if item["split"] not in ALLOWED_SPLITS or item["evidence_label"] != "SIM_GEOMETRIC":
            raise RuntimeError("Protocol-R split or evidence-label escape")
        identifiers = [
            item["instance_id"],
            item["task_group_id"],
            item["workpiece_id"],
            item["layout_id"],
            *item["parent_curve_ids"],
        ]
        if any(data["id_prefix"] not in str(identifier) for identifier in identifiers):
            raise RuntimeError("Protocol-R identifier namespace escape")
        if any(
            old in str(identifier)
            for old in data["disjoint_id_prefixes"]
            for identifier in identifiers
        ):
            raise RuntimeError("Protocol-R old namespace overlap")
    for field in ("task_group_id", "instance_id", "workpiece_id", "layout_id"):
        train = {item[field] for item in records if item["split"] == "train"}
        development = {
            item[field] for item in records if item["split"] == "development"
        }
        if train & development:
            raise RuntimeError(f"Protocol-R split leakage in {field}")
    train_parents = {
        parent
        for item in records
        if item["split"] == "train"
        for parent in item["parent_curve_ids"]
    }
    development_parents = {
        parent
        for item in records
        if item["split"] == "development"
        for parent in item["parent_curve_ids"]
    }
    if train_parents & development_parents:
        raise RuntimeError("Protocol-R parent-curve split leakage")


def generate_protocol_r_data(
    config: Mapping[str, Any],
    output_root: str | Path,
    context: OracleContext,
    *,
    execution_authorized: bool = False,
) -> dict[str, Any]:
    """One-shot materialization, unreachable until a future frozen execution."""
    require_execution_ready(config)
    if not execution_authorized:
        raise PermissionError("explicit Protocol-R execution authorization missing")
    output = Path(output_root).resolve()
    guard_protocol_r_path(output)
    if output.exists():
        raise FileExistsError("Protocol-R corpus generation is immutable and one-shot")
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    data = config["data"]

    def save(split, cell, group_id, variant, instance, witness_plan, challenge):
        relative = Path("data") / split / cell / f"{instance.instance_id}.json"
        witness_relative = (
            Path("witnesses") / split / cell / f"{instance.instance_id}.json"
        )
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        (output / witness_relative).parent.mkdir(parents=True, exist_ok=True)
        (output / relative).write_text(
            json.dumps(instance.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        witness_payload = {
            "plan": witness_plan.to_dict(),
            "plan_sha256": canonical_hash(witness_plan.to_dict()),
        }
        (output / witness_relative).write_text(
            json.dumps(witness_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        records.append(
            {
                "split": split,
                "cell_id": cell,
                "task_group_id": group_id,
                "variant_index": variant,
                "instance_id": instance.instance_id,
                "workpiece_id": instance.workpiece_id,
                "layout_id": instance.layout_id,
                "parent_curve_ids": sorted(
                    {item.parent_curve_id for item in instance.segments}
                ),
                "relative_path": relative.as_posix(),
                "witness_relative_path": witness_relative.as_posix(),
                "instance_sha256": hashlib.sha256(
                    canonical_instance_bytes(instance)
                ).hexdigest(),
                "witness_sha256": witness_payload["plan_sha256"],
                "challenge": challenge,
                "evidence_label": "SIM_GEOMETRIC",
            }
        )

    geometry = data["generator"]["geometry"]
    for cell in data["cells"]:
        for index in range(int(data["regular_train_groups_per_cell"])):
            group_id = data["id_templates"]["regular_train_group"].format(
                cell=cell, index=index
            )
            for variant in range(int(data["variants_per_group"])):
                base = build_regular_variant(config, cell, "train", index, variant)
                witness = construct_pointer_compatible_witness(
                    base,
                    context,
                    tight_pre_margin_duration=float(
                        geometry["witness_tight_pre_margin_duration"]
                    ),
                    tight_post_margin_duration=float(
                        geometry["witness_tight_post_margin_duration"]
                    ),
                    loose_pre_margin_s=float(geometry["witness_loose_pre_margin_s"]),
                )
                save("train", cell, group_id, variant, witness.instance, witness.plan, False)
        if cell in data["challenge_cells"]:
            for slot in range(int(data["challenge_train_groups_per_challenge_cell"])):
                group_id = data["id_templates"]["challenge_train_group"].format(
                    cell=cell, index=slot
                )
                for variant, challenge in enumerate(
                    build_challenge_group(config, cell, slot, context)
                ):
                    save(
                        "train",
                        cell,
                        group_id,
                        variant,
                        challenge.instance,
                        challenge.witness_plan,
                        True,
                    )
        for index in range(int(data["development_groups_per_cell"])):
            group_id = data["id_templates"]["development_group"].format(
                cell=cell, index=index
            )
            for variant in range(int(data["variants_per_group"])):
                base = build_regular_variant(config, cell, "development", index, variant)
                witness = construct_pointer_compatible_witness(
                    base,
                    context,
                    tight_pre_margin_duration=float(
                        geometry["witness_tight_pre_margin_duration"]
                    ),
                    tight_post_margin_duration=float(
                        geometry["witness_tight_post_margin_duration"]
                    ),
                    loose_pre_margin_s=float(geometry["witness_loose_pre_margin_s"]),
                )
                save(
                    "development",
                    cell,
                    group_id,
                    variant,
                    witness.instance,
                    witness.plan,
                    False,
                )
    audit_protocol_r_records(records, config)
    payload = {
        "version": "a4b-protocol-r-development-manifest-v1",
        "config_sha256": config["config_sha256"],
        "records": sorted(records, key=lambda item: item["instance_id"]),
        "forbidden_splits_generated": [],
    }
    payload["manifest_sha256"] = canonical_hash(payload)
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def load_protocol_r_items(
    data_root: str | Path, split: str, context: OracleContext
):
    if split not in ALLOWED_SPLITS:
        raise PermissionError(f"Protocol R forbids split {split}")
    root = Path(data_root).resolve()
    guard_protocol_r_path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest.pop("manifest_sha256")
    if canonical_hash(manifest) != expected or manifest["forbidden_splits_generated"]:
        raise RuntimeError("Protocol-R manifest integrity failure")
    rows = []
    for record in manifest["records"]:
        if record["split"] != split:
            continue
        path = (root / record["relative_path"]).resolve()
        guard_protocol_r_path(path)
        instance = allocation_instance_from_dict(json.loads(path.read_text()))
        if hashlib.sha256(canonical_instance_bytes(instance)).hexdigest() != record["instance_sha256"]:
            raise RuntimeError("Protocol-R instance hash mismatch")
        witness_path = (root / record["witness_relative_path"]).resolve()
        guard_protocol_r_path(witness_path)
        witness = json.loads(witness_path.read_text())
        if witness["plan_sha256"] != record["witness_sha256"]:
            raise RuntimeError("Protocol-R witness hash mismatch")
        plan = allocation_plan_from_dict(witness["plan"], instance)
        if not verify_plan(instance, plan, context).feasible:
            raise RuntimeError("Protocol-R witness verifier failure")
        rows.append((record, instance))
    if not rows:
        raise ValueError(f"no Protocol-R records for split {split}")
    return tuple(sorted(rows, key=lambda item: item[0]["instance_id"])), expected
