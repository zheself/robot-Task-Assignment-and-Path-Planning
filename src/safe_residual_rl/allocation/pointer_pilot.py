"""A3.5 train/validation-only data and teacher protocol.

This module never loads an A2 v2/v3/v4 instance or witness.  It reuses the
programmatic generator, A1 oracle, solvers, scheduler and verifier to create a
new development-only corpus with a disjoint ID namespace.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .generation import (
    BenchmarkConfig,
    SplitGenerationSpec,
    canonical_instance_bytes,
    generate_instance,
    stable_seed,
)
from .oracle import OracleContext
from .schema import AllocationInstance, AllocationPlan, EvidenceLabel, TimeWindow
from .scheduling import build_schedule
from .solvers import (
    SolverProtocol,
    solve_hybrid_assignment_milp,
    solve_order_aware_lns,
)
from .solvers.common import allocation_units
from .verifier import verify_plan
from .witness import ConstructiveWitness, construct_feasible_witness

PILOT_VERSION = "a3-5-feasible-pair-pointer-pilot-v1"
MANIFEST_VERSION = "a3-5-pointer-pilot-manifest-v1"
ALLOWED_SPLITS = frozenset({"train", "validation"})
FORBIDDEN_SPLITS = frozenset({"frozen_test", "stress"})


@dataclass(frozen=True)
class PointerAction:
    unit_id: str
    unit: tuple[str, ...]
    robot_id: str

    def to_dict(self) -> dict[str, object]:
        return {"unit_id": self.unit_id, "unit": list(self.unit), "robot_id": self.robot_id}


@dataclass(frozen=True)
class PilotCell:
    cell_id: str
    train_groups: int
    validation_groups: int
    variants: int
    spec: SplitGenerationSpec


@dataclass(frozen=True)
class PointerPilotConfig:
    raw: Mapping[str, Any]
    sha256: str
    master_seed: int
    id_prefix: str
    geometry: Mapping[str, Any]
    cells: tuple[PilotCell, ...]
    objective_weights: Mapping[str, float]


@dataclass(frozen=True)
class PilotRecord:
    split: str
    cell_id: str
    task_group_id: str
    variant_index: int
    instance_id: str
    workpiece_id: str
    layout_id: str
    parent_curve_ids: tuple[str, ...]
    relative_path: str
    instance_sha256: str
    teacher_relative_path: str
    teacher_sha256: str
    teacher_method: str
    teacher_solver_status: str
    teacher_objective: float | None
    teacher_bound: float | None
    teacher_gap: float | None
    teacher_runtime_s: float
    teacher_fallback: bool

    def to_dict(self) -> dict[str, object]:
        value = dict(self.__dict__)
        value["parent_curve_ids"] = list(self.parent_curve_ids)
        return value


@dataclass(frozen=True)
class PilotManifest:
    config_sha256: str
    records: tuple[PilotRecord, ...]
    source_hashes: tuple[tuple[str, str], ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": MANIFEST_VERSION,
            "evidence_label": "SIM_GEOMETRIC",
            "allowed_splits": ["train", "validation"],
            "config_sha256": self.config_sha256,
            "source_hashes": dict(self.source_hashes),
            "records": [item.to_dict() for item in self.records],
            "manifest_sha256": self.manifest_sha256,
        }


def load_pointer_pilot_config(path: str | Path) -> PointerPilotConfig:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("version") != PILOT_VERSION:
        raise ValueError("unexpected A3.5 pilot version")
    guard = raw["access_guard"]
    if guard["allowed_splits"] != ["train", "validation"]:
        raise ValueError("A3.5 allowed split contract changed")
    if guard["forbidden_splits"] != ["frozen_test", "stress"]:
        raise ValueError("A3.5 forbidden split contract changed")
    cells = tuple(
        PilotCell(
            str(item["cell_id"]),
            int(item["train_groups"]),
            int(item["validation_groups"]),
            int(item["variants"]),
            SplitGenerationSpec(
                group_count=max(int(item["train_groups"]), int(item["validation_groups"])),
                robot_count=_pair(item["robots"]),
                segment_count=_pair(item["segments"]),
                max_segments_per_curve=int(item["max_segments_per_curve"]),
                precedence_probability=float(item["precedence_probability"]),
                shared_resource_probability=float(item["shared_resource_probability"]),
                tight_window_probability=float(item["tight_window_probability"]),
            ),
        )
        for item in raw["cells"]
    )
    if len({item.cell_id for item in cells}) != 6:
        raise ValueError("A3.5 requires six unique development cells")
    if any(min(item.train_groups, item.validation_groups, item.variants) < 1 for item in cells):
        raise ValueError("A3.5 cells cannot be empty")
    return PointerPilotConfig(
        raw,
        hashlib.sha256(source.read_bytes()).hexdigest(),
        int(raw["master_seed"]),
        str(raw["id_prefix"]),
        dict(raw["geometry"]),
        cells,
        {str(key): float(value) for key, value in raw["objective_weights"].items()},
    )


def materialize_pointer_pilot(
    config: PointerPilotConfig,
    context: OracleContext,
    project_root: str | Path,
    data_relative: str | Path,
) -> PilotManifest:
    root = Path(project_root).resolve()
    data_relative = Path(data_relative)
    _guard_development_path(root / data_relative)
    records: list[PilotRecord] = []
    teacher_raw = config.raw["teacher"]
    protocol = SolverProtocol(
        "a1-solver-protocol-v1",
        float(teacher_raw["milp_time_limit_s"]),
        float(teacher_raw["milp_relative_gap"]),
        int(teacher_raw["lns_seed"]),
    )
    for split in ("train", "validation"):
        for cell in config.cells:
            groups = cell.train_groups if split == "train" else cell.validation_groups
            adapter = BenchmarkConfig(
                version="a2-geometric-benchmark-v1",
                manifest_version="a2-split-manifest-v1",
                master_seed=config.master_seed,
                evidence_label=EvidenceLabel.SIM_GEOMETRIC,
                coordinate_frame="synthetic_workcell_m",
                variants_per_group=cell.variants,
                geometry=config.geometry,
                splits=((split, cell.spec),),
                objective_weights=tuple(sorted(config.objective_weights.items())),
                baseline_protocol={},
                boundaries=tuple(str(item) for item in config.raw["boundaries"]),
            )
            for group_index in range(groups):
                group_id = f"{config.id_prefix}-{split}-{cell.cell_id}-group-{group_index:03d}"
                unique_index = stable_seed(
                    config.master_seed, split, cell.cell_id, group_index, "a3-5-pointer-group"
                )
                for variant in range(cell.variants):
                    generated = generate_instance(adapter, split, group_id, unique_index, variant)
                    if not generated.instance.instance_id.startswith(config.id_prefix):
                        raise RuntimeError("A3.5 generator escaped its ID namespace")
                    witness = construct_pointer_compatible_witness(
                        generated.instance,
                        context,
                        tight_pre_margin_duration=float(config.geometry["witness_tight_pre_margin_duration"]),
                        tight_post_margin_duration=float(config.geometry["witness_tight_post_margin_duration"]),
                        loose_pre_margin_s=float(config.geometry["witness_loose_pre_margin_s"]),
                    )
                    teacher, metadata = select_canonical_teacher(
                        witness,
                        context,
                        config.objective_weights,
                        protocol,
                        int(teacher_raw["lns_iterations"]),
                        int(teacher_raw["lns_seed"]),
                    )
                    instance = teacher.instance
                    relative = data_relative / split / cell.cell_id / f"{instance.instance_id}.json"
                    teacher_relative = data_relative / "witnesses" / split / cell.cell_id / f"{instance.instance_id}.json"
                    destination = root / relative
                    teacher_destination = root / teacher_relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    teacher_destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(instance.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
                    teacher_payload = teacher.to_dict()
                    teacher_payload["teacher_metadata"] = metadata
                    teacher_payload["canonical_actions"] = [
                        item.to_dict() for item in canonical_teacher_actions(instance, teacher.plan)
                    ]
                    teacher_destination.write_text(
                        json.dumps(teacher_payload, indent=2, sort_keys=True), encoding="utf-8"
                    )
                    records.append(
                        PilotRecord(
                            split,
                            cell.cell_id,
                            group_id,
                            variant,
                            instance.instance_id,
                            instance.workpiece_id,
                            instance.layout_id,
                            tuple(sorted({item.parent_curve_id for item in instance.segments})),
                            relative.as_posix(),
                            hashlib.sha256(canonical_instance_bytes(instance)).hexdigest(),
                            teacher_relative.as_posix(),
                            teacher.witness_sha256,
                            str(metadata["selected_method"]),
                            str(metadata["solver_status"]),
                            _optional_float(metadata["objective"]),
                            _optional_float(metadata["best_bound"]),
                            _optional_float(metadata["mip_gap"]),
                            float(metadata["runtime_s"]),
                            bool(metadata["constructive_fallback"]),
                        )
                    )
    _assert_internal_disjointness(records)
    source_hashes = source_hashes_for_config(config, root)
    payload = _manifest_payload(config.sha256, records, source_hashes)
    digest = _digest(payload)
    return PilotManifest(config.sha256, tuple(records), tuple(sorted(source_hashes.items())), digest)


def construct_pointer_compatible_witness(
    instance: AllocationInstance,
    context: OracleContext,
    *,
    tight_pre_margin_duration: float,
    tight_post_margin_duration: float,
    loose_pre_margin_s: float,
) -> ConstructiveWitness:
    """Create a verified witness whose robot orders are atomic-unit blocks.

    The generic A2 witness may interleave members of an atomic unit with other
    units.  One unit-level pointer action cannot encode that segment-level
    interleaving.  This A3.5-specific constructor first schedules under the
    registered horizon, canonicalises to unit blocks, then reconstructs the
    original tight/loose window pattern around that representable schedule.
    """
    horizon = min(
        [item.availability.end_s for item in instance.robots]
        + [item.availability.end_s for item in instance.resources]
    )
    relaxed = replace(
        instance,
        segments=tuple(replace(item, time_window=TimeWindow(0.0, horizon)) for item in instance.segments),
    )
    base = construct_feasible_witness(
        relaxed,
        context,
        tight_pre_margin_duration=tight_pre_margin_duration,
        tight_post_margin_duration=tight_post_margin_duration,
        loose_pre_margin_s=loose_pre_margin_s,
    )
    canonical = canonicalize_teacher_plan(relaxed, base.plan, context)
    if canonical is None:
        raise RuntimeError("cannot construct a pointer-representable relaxed witness")
    scheduled = {item.segment_id: item for item in canonical.schedule}
    calibrated_segments = []
    tight_count = 0
    for segment in instance.segments:
        item = scheduled[segment.id]
        tight = segment.time_window.end_s - segment.time_window.start_s < 0.5 * horizon
        if tight:
            tight_count += 1
            start = max(0.0, item.planned_start_s - tight_pre_margin_duration * segment.process_duration_s)
            end = min(horizon, item.planned_end_s + tight_post_margin_duration * segment.process_duration_s)
        else:
            start = max(0.0, item.planned_start_s - loose_pre_margin_s)
            end = horizon
        calibrated_segments.append(replace(segment, time_window=TimeWindow(start, end)))
    calibrated = replace(instance, segments=tuple(calibrated_segments))
    if not verify_plan(calibrated, canonical, context).feasible:
        raise RuntimeError("pointer-compatible calibrated witness failed verification")
    return ConstructiveWitness(
        calibrated,
        canonical,
        _teacher_digest(calibrated, canonical),
        (
            "A3_5_POINTER_COMPATIBLE_CONSTRUCTIVE_WITNESS",
            f"TIGHT_WINDOWS_RECALIBRATED={tight_count}",
            "ATOMIC_UNIT_BLOCK_ORDER",
            "NOT_GLOBAL_OPTIMUM_OR_REAL_EXPERT",
        ),
    )


def select_canonical_teacher(
    witness: ConstructiveWitness,
    context: OracleContext,
    weights: Mapping[str, float],
    protocol: SolverProtocol,
    lns_iterations: int,
    lns_seed: int,
) -> tuple[ConstructiveWitness, dict[str, object]]:
    instance = witness.instance
    candidates = [
        solve_hybrid_assignment_milp(instance, context, protocol),
        solve_order_aware_lns(
            instance,
            context,
            iterations=lns_iterations,
            seed=lns_seed,
            objective_weights=weights,
        ),
    ]
    viable: list[tuple[float, AllocationPlan, dict[str, object]]] = []
    for result in candidates:
        if result.plan is None or not verify_plan(instance, result.plan, context).feasible:
            continue
        canonical = canonicalize_teacher_plan(instance, result.plan, context)
        if canonical is None:
            continue
        viable.append(
            (
                _score(canonical, weights),
                canonical,
                {
                    "selected_method": result.method_id,
                    "solver_status": result.status,
                    "objective": result.objective_value,
                    "best_bound": result.best_bound,
                    "mip_gap": result.mip_gap,
                    "runtime_s": result.runtime_s,
                    "constructive_fallback": False,
                    "diagnostics": list(result.diagnostics),
                },
            )
        )
    witness_canonical = canonicalize_teacher_plan(instance, witness.plan, context)
    if witness_canonical is not None:
        viable.append(
            (
                _score(witness_canonical, weights),
                witness_canonical,
                {
                    "selected_method": "constructive-witness-v1",
                    "solver_status": "feasible",
                    "objective": _score(witness_canonical, weights),
                    "best_bound": None,
                    "mip_gap": None,
                    "runtime_s": 0.0,
                    "constructive_fallback": True,
                    "diagnostics": list(witness.diagnostics),
                },
            )
        )
    if not viable:
        raise RuntimeError(f"no representable verified teacher for {instance.instance_id}")
    score, plan, metadata = min(viable, key=lambda item: (item[0], item[1].method_id))
    metadata["canonical_weighted_proxy_score"] = score
    metadata["teacher_is_globally_optimal"] = False
    digest = _teacher_digest(instance, plan)
    selected = ConstructiveWitness(
        instance,
        plan,
        digest,
        (
            "A3_5_CANONICAL_UNIT_BLOCK_TEACHER",
            f"SOURCE={metadata['selected_method']}",
            "VERIFIED_A1_PROXY_INCUMBENT_NOT_GLOBAL_OPTIMUM_OR_REAL_EXPERT",
        ),
    )
    if verify_plan(instance, plan, context).feasible is not True:
        raise RuntimeError("selected A3.5 teacher failed final verification")
    validate_teacher_prefixes(instance, canonical_teacher_actions(instance, plan))
    replayed = replay_pointer_actions(instance, canonical_teacher_actions(instance, plan), context)
    if replayed is None or _assignment_order_signature(replayed) != _assignment_order_signature(plan):
        raise RuntimeError("canonical teacher sequence does not replay exactly")
    return selected, metadata


def canonical_teacher_actions(
    instance: AllocationInstance, plan: AllocationPlan
) -> tuple[PointerAction, ...]:
    units = allocation_units(instance)
    scheduled = {item.segment_id: item for item in plan.schedule}
    if set(scheduled) != {item.id for item in instance.segments}:
        raise ValueError("teacher does not cover all segments")
    actions: list[PointerAction] = []
    for unit in units:
        robots = {scheduled[item].robot_id for item in unit}
        if len(robots) != 1:
            raise ValueError("teacher splits an atomic unit")
        actions.append(PointerAction(_unit_id(unit), tuple(unit), next(iter(robots))))
    dependencies = unit_dependencies(instance, units)
    remaining = {item.unit_id: item for item in actions}
    completed: set[str] = set()
    result: list[PointerAction] = []
    while remaining:
        ready = [item for item in remaining.values() if dependencies[item.unit_id] <= completed]
        if not ready:
            raise ValueError("teacher unit dependency graph is cyclic")
        chosen = min(
            ready,
            key=lambda item: (
                min(scheduled[segment].planned_start_s for segment in item.unit),
                item.robot_id,
                item.unit_id,
            ),
        )
        result.append(chosen)
        completed.add(chosen.unit_id)
        remaining.pop(chosen.unit_id)
    return tuple(result)


def validate_teacher_prefixes(
    instance: AllocationInstance, actions: Sequence[PointerAction]
) -> None:
    units = allocation_units(instance)
    expected = {_unit_id(unit): tuple(unit) for unit in units}
    dependencies = unit_dependencies(instance, units)
    completed: set[str] = set()
    robots = {item.id for item in instance.robots}
    for action in actions:
        if action.unit_id not in expected or action.unit != expected[action.unit_id]:
            raise ValueError("teacher action references an unknown unit")
        if action.unit_id in completed:
            raise ValueError("teacher repeats an atomic unit")
        if action.robot_id not in robots:
            raise ValueError("teacher action references an unknown robot")
        if not dependencies[action.unit_id] <= completed:
            raise ValueError("teacher selects a unit before its predecessors")
        completed.add(action.unit_id)
    if completed != set(expected):
        raise ValueError("teacher action sequence is incomplete")


def replay_pointer_actions(
    instance: AllocationInstance,
    actions: Sequence[PointerAction],
    context: OracleContext,
    *,
    method_id: str = "a3-5-pointer-replay-v1",
) -> AllocationPlan | None:
    validate_teacher_prefixes(instance, actions)
    segment_by_id = {item.id: item for item in instance.segments}
    orders = {item.id: [] for item in instance.robots}
    for action in actions:
        members = sorted(action.unit, key=lambda item: (segment_by_id[item].segment_index, item))
        orders[action.robot_id].extend(members)
    built = build_schedule(instance, orders, context, method_id)
    return built.plan


def canonicalize_teacher_plan(
    instance: AllocationInstance, plan: AllocationPlan, context: OracleContext
) -> AllocationPlan | None:
    try:
        actions = canonical_teacher_actions(instance, plan)
        return replay_pointer_actions(instance, actions, context, method_id=f"canonical-{plan.method_id}")
    except ValueError:
        return None


def unit_dependencies(
    instance: AllocationInstance, units: Sequence[Sequence[str]] | None = None
) -> dict[str, set[str]]:
    selected = tuple(tuple(item) for item in (units or allocation_units(instance)))
    owner = {segment: _unit_id(unit) for unit in selected for segment in unit}
    dependencies = {_unit_id(unit): set() for unit in selected}
    segments = {item.id: item for item in instance.segments}
    parent_groups: dict[str, list[object]] = {}
    for segment in instance.segments:
        parent_groups.setdefault(segment.parent_curve_id, []).append(segment)
    edges = [(predecessor, item.id) for item in instance.segments for predecessor in item.predecessor_ids]
    for group in parent_groups.values():
        ordered = sorted(group, key=lambda item: item.segment_index)
        edges.extend((left.id, right.id) for left, right in zip(ordered, ordered[1:]))
    for predecessor, successor in edges:
        left, right = owner[predecessor], owner[successor]
        if left != right:
            dependencies[right].add(left)
    return dependencies


def write_pointer_manifest(manifest: PilotManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_pointer_manifest(path: str | Path) -> PilotManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("version") != MANIFEST_VERSION or set(raw.get("allowed_splits", ())) != ALLOWED_SPLITS:
        raise ValueError("invalid A3.5 manifest contract")
    records = tuple(PilotRecord(
        split=str(item["split"]), cell_id=str(item["cell_id"]), task_group_id=str(item["task_group_id"]),
        variant_index=int(item["variant_index"]), instance_id=str(item["instance_id"]),
        workpiece_id=str(item["workpiece_id"]), layout_id=str(item["layout_id"]),
        parent_curve_ids=tuple(str(value) for value in item["parent_curve_ids"]),
        relative_path=str(item["relative_path"]), instance_sha256=str(item["instance_sha256"]),
        teacher_relative_path=str(item["teacher_relative_path"]), teacher_sha256=str(item["teacher_sha256"]),
        teacher_method=str(item["teacher_method"]), teacher_solver_status=str(item["teacher_solver_status"]),
        teacher_objective=_optional_float(item.get("teacher_objective")), teacher_bound=_optional_float(item.get("teacher_bound")),
        teacher_gap=_optional_float(item.get("teacher_gap")), teacher_runtime_s=float(item["teacher_runtime_s"]),
        teacher_fallback=bool(item["teacher_fallback"]),
    ) for item in raw["records"])
    source_hashes = tuple(sorted((str(key), str(value)) for key, value in raw["source_hashes"].items()))
    manifest = PilotManifest(str(raw["config_sha256"]), records, source_hashes, str(raw["manifest_sha256"]))
    expected = _digest(_manifest_payload(manifest.config_sha256, records, dict(source_hashes)))
    if expected != manifest.manifest_sha256:
        raise ValueError("A3.5 manifest hash mismatch")
    _assert_internal_disjointness(records)
    return manifest


def audit_manifest_overlap(
    manifest: PilotManifest, historical_manifest_paths: Sequence[str | Path]
) -> dict[str, list[str]]:
    pilot_groups = {item.task_group_id for item in manifest.records}
    pilot_instances = {item.instance_id for item in manifest.records}
    overlap_groups: set[str] = set()
    overlap_instances: set[str] = set()
    for path in historical_manifest_paths:
        source = Path(path)
        if source.name not in {
            "a2_paper_manifest_v2.json",
            "a2_paper_manifest_v3.json",
            "a2_paper_manifest_v4.json",
        }:
            raise PermissionError("A3.5 overlap audit accepts historical manifest metadata only")
        raw = json.loads(source.read_text(encoding="utf-8"))
        for item in raw.get("records", ()):
            if str(item.get("task_group_id")) in pilot_groups:
                overlap_groups.add(str(item["task_group_id"]))
            if str(item.get("instance_id")) in pilot_instances:
                overlap_instances.add(str(item["instance_id"]))
    return {"task_group_ids": sorted(overlap_groups), "instance_ids": sorted(overlap_instances)}


def source_hashes_for_config(config: PointerPilotConfig, root: Path) -> dict[str, str]:
    result = {}
    for relative in config.raw["source_files_to_hash"]:
        path = root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"A3.5 registered source missing: {relative}")
        result[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _manifest_payload(config_sha: str, records: Sequence[PilotRecord], source_hashes: Mapping[str, str]) -> dict[str, object]:
    return {
        "version": MANIFEST_VERSION,
        "evidence_label": "SIM_GEOMETRIC",
        "allowed_splits": ["train", "validation"],
        "config_sha256": config_sha,
        "source_hashes": dict(sorted(source_hashes.items())),
        "records": [item.to_dict() for item in records],
    }


def _assert_internal_disjointness(records: Sequence[PilotRecord]) -> None:
    if {item.split for item in records} != ALLOWED_SPLITS:
        raise ValueError("A3.5 manifest must contain train and validation only")
    for name, accessor in (
        ("task_group", lambda item: {item.task_group_id}),
        ("workpiece", lambda item: {item.workpiece_id}),
        ("layout", lambda item: {item.layout_id}),
        ("parent_curve", lambda item: set(item.parent_curve_ids)),
    ):
        train = set().union(*(accessor(item) for item in records if item.split == "train"))
        validation = set().union(*(accessor(item) for item in records if item.split == "validation"))
        if train & validation:
            raise ValueError(f"A3.5 train/validation {name} leakage")
    ids = [item.instance_id for item in records]
    if len(ids) != len(set(ids)) or any(not item.startswith("a35p1") for item in ids):
        raise ValueError("A3.5 instance IDs are duplicate or outside namespace")


def _guard_development_path(path: Path) -> None:
    parts = set(path.resolve().parts)
    if parts & FORBIDDEN_SPLITS:
        raise PermissionError("A3.5 path contains forbidden split")
    if any(token in parts for token in ("a2_paper_v2", "a2_paper_v3", "a2_paper_v4")):
        raise PermissionError("A3.5 path points into an A2 corpus")


def _assignment_order_signature(plan: AllocationPlan) -> tuple[tuple[str, str, int], ...]:
    return tuple(sorted((item.segment_id, item.robot_id, item.order_index) for item in plan.schedule))


def _teacher_digest(instance: AllocationInstance, plan: AllocationPlan) -> str:
    return _digest({"instance": instance.to_dict(), "plan": plan.to_dict()})


def _score(plan: AllocationPlan, weights: Mapping[str, float]) -> float:
    return sum(float(weights.get(key, 0.0)) * float(value) for key, value in plan.objective_terms)


def _unit_id(unit: Sequence[str]) -> str:
    return "+".join(unit)


def _digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _pair(value: Sequence[object]) -> tuple[int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 2:
        raise ValueError("range must have two endpoints")
    return result  # type: ignore[return-value]


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
