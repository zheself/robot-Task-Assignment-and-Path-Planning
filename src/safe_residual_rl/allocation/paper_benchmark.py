"""Paper-scale A2 v2 generation, proxy-admissibility and frozen manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .generation import (
    BenchmarkConfig,
    GeneratedInstance,
    SplitGenerationSpec,
    canonical_instance_bytes,
    generate_instance,
    stable_seed,
)
from .masks import build_edge_mask
from .oracle import OracleContext, load_oracle_context
from .schema import AllocationInstance, EvidenceLabel, allocation_instance_from_dict, allocation_plan_from_dict, validate_instance
from .solvers.common import allocation_units
from .witness import ConstructiveWitness, construct_feasible_witness, verify_constructive_witness

PAPER_CONFIG_VERSION = "a2-paper-geometric-benchmark-v2"
PAPER_MANIFEST_VERSION = "a2-paper-split-manifest-v2"
PAPER_CONFIG_VERSIONS = {
    "a2-paper-geometric-benchmark-v2",
    "a2-paper-geometric-benchmark-v3",
    "a2-paper-geometric-benchmark-v4",
}
PAPER_MANIFEST_VERSIONS = {
    "a2-paper-split-manifest-v2",
    "a2-paper-split-manifest-v3",
    "a2-paper-split-manifest-v4",
}


@dataclass(frozen=True)
class PaperCell:
    split: str
    cell_id: str
    paper_role: str
    groups: int
    variants: int
    generation_spec: SplitGenerationSpec
    feasibility_policy: str


@dataclass(frozen=True)
class PaperBenchmarkConfig:
    version: str
    manifest_version: str
    generator_version: str
    benchmark_tier: str
    master_seed: int
    evidence_label: EvidenceLabel
    coordinate_frame: str
    geometry: Mapping[str, Any]
    cells: tuple[PaperCell, ...]
    objective_weights: tuple[tuple[str, float], ...]
    baseline_protocol: Mapping[str, Any]
    statistics: Mapping[str, Any]
    acceptance: Mapping[str, Any]
    boundaries: tuple[str, ...]
    config_sha256: str


@dataclass(frozen=True)
class PaperGeneratedInstance:
    split: str
    cell_id: str
    paper_role: str
    task_group_id: str
    variant_index: int
    seed: int
    feasibility_policy: str
    instance: AllocationInstance


@dataclass(frozen=True)
class ProxyAdmissibility:
    admissible: bool
    uncovered_unit_ids: tuple[str, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class PaperRecord:
    instance_id: str
    split: str
    cell_id: str
    paper_role: str
    task_group_id: str
    variant_index: int
    seed: int
    feasibility_policy: str
    proxy_admissible: bool
    workpiece_id: str
    layout_id: str
    parent_curve_ids: tuple[str, ...]
    evidence_label: str
    robot_count: int
    segment_count: int
    relative_path: str
    sha256: str
    witness_relative_path: str | None = None
    witness_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["parent_curve_ids"] = list(self.parent_curve_ids)
        if self.witness_relative_path is None:
            result.pop("witness_relative_path")
        if self.witness_sha256 is None:
            result.pop("witness_sha256")
        return result


@dataclass(frozen=True)
class PaperManifest:
    version: str
    generator_version: str
    config_sha256: str
    records: tuple[PaperRecord, ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generator_version": self.generator_version,
            "config_sha256": self.config_sha256,
            "records": [item.to_dict() for item in self.records],
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class PaperAuditIssue:
    code: str
    key: str
    splits: tuple[str, ...]


def load_paper_config(path: str | Path) -> PaperBenchmarkConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = []
    for item in raw["cells"]:
        cells.append(
            PaperCell(
                split=str(item["split"]),
                cell_id=str(item["cell_id"]),
                paper_role=str(item["paper_role"]),
                groups=int(item["groups"]),
                variants=int(item["variants"]),
                generation_spec=SplitGenerationSpec(
                    group_count=int(item["groups"]),
                    robot_count=_pair(item["robots"]),
                    segment_count=_pair(item["segments"]),
                    max_segments_per_curve=int(item["max_segments_per_curve"]),
                    precedence_probability=float(item["precedence_probability"]),
                    shared_resource_probability=float(item["shared_resource_probability"]),
                    tight_window_probability=float(item["tight_window_probability"]),
                ),
                feasibility_policy=str(item["feasibility_policy"]),
            )
        )
    canonical = _canonical(raw)
    config = PaperBenchmarkConfig(
        version=str(raw["version"]),
        manifest_version=str(raw["manifest_version"]),
        generator_version=str(raw["generator_version"]),
        benchmark_tier=str(raw["benchmark_tier"]),
        master_seed=int(raw["master_seed"]),
        evidence_label=EvidenceLabel(raw["evidence_label"]),
        coordinate_frame=str(raw["coordinate_frame"]),
        geometry=dict(raw["geometry"]),
        cells=tuple(cells),
        objective_weights=tuple(sorted((str(key), float(value)) for key, value in raw["objective_weights"].items())),
        baseline_protocol=dict(raw["baseline_protocol"]),
        statistics=dict(raw["statistics"]),
        acceptance=dict(raw["acceptance"]),
        boundaries=tuple(str(item) for item in raw["boundaries"]),
        config_sha256=hashlib.sha256(canonical).hexdigest(),
    )
    _validate_paper_config(config)
    return config


def generate_paper_benchmark(config: PaperBenchmarkConfig) -> tuple[PaperGeneratedInstance, ...]:
    result: list[PaperGeneratedInstance] = []
    for cell in config.cells:
        adapter = BenchmarkConfig(
            version="a2-geometric-benchmark-v1",
            manifest_version="a2-split-manifest-v1",
            master_seed=config.master_seed,
            evidence_label=config.evidence_label,
            coordinate_frame=config.coordinate_frame,
            variants_per_group=cell.variants,
            geometry=config.geometry,
            splits=((cell.split, cell.generation_spec),),
            objective_weights=config.objective_weights,
            baseline_protocol=config.baseline_protocol,
            boundaries=config.boundaries,
        )
        for group_index in range(cell.groups):
            version_tag = config.version.rsplit("-", 1)[-1]
            group_prefix = "" if version_tag == "v2" else f"{version_tag}-"
            group_id = f"{group_prefix}{cell.split}-{cell.cell_id}-group-{group_index:03d}"
            seed_tag = "paper-v2-group" if version_tag == "v2" else f"paper-{version_tag}-group"
            unique_index = stable_seed(config.master_seed, cell.split, cell.cell_id, group_index, seed_tag)
            for variant_index in range(cell.variants):
                base = generate_instance(adapter, cell.split, group_id, unique_index, variant_index)
                instance = base.instance
                if cell.feasibility_policy == "designed_edge_infeasible":
                    first = instance.segments[0]
                    injected = replace(first, required_capabilities=first.required_capabilities + ("unavailable-negative-control",))
                    instance = replace(instance, segments=(injected,) + instance.segments[1:])
                if validate_instance(instance):
                    raise ValueError(f"schema failure in {instance.instance_id}")
                result.append(
                    PaperGeneratedInstance(
                        split=cell.split,
                        cell_id=cell.cell_id,
                        paper_role=cell.paper_role,
                        task_group_id=group_id,
                        variant_index=variant_index,
                        seed=base.seed,
                        feasibility_policy=cell.feasibility_policy,
                        instance=instance,
                    )
                )
    return tuple(result)


def proxy_admissibility(
    instance: AllocationInstance, context: OracleContext
) -> ProxyAdmissibility:
    mask = build_edge_mask(instance, context)
    uncovered: list[str] = []
    for unit in allocation_units(instance):
        common = [
            robot_id
            for robot_id in mask.robot_ids
            if all(mask.is_allowed(segment_id, robot_id) for segment_id in unit)
        ]
        if not common:
            uncovered.append("+".join(unit))
    return ProxyAdmissibility(
        admissible=not uncovered,
        uncovered_unit_ids=tuple(uncovered),
        diagnostics=(
            "EDGE_MASK_AND_COUPLED_UNIT_COVER_ONLY",
            "NOT_A_JOINT_SCHEDULE_FEASIBILITY_PROOF",
        ),
    )


def materialize_paper_benchmark(
    config: PaperBenchmarkConfig,
    context: OracleContext,
    project_root: str | Path,
    instance_directory: str | Path = "outputs/phase1_allocation/a2_paper_v2/instances",
) -> tuple[PaperManifest, tuple[PaperGeneratedInstance, ...]]:
    root = Path(project_root)
    generated = generate_paper_benchmark(config)
    effective_generated: list[PaperGeneratedInstance] = []
    records: list[PaperRecord] = []
    for item in generated:
        witness = None
        if item.feasibility_policy == "constructive_witness_required":
            witness = construct_feasible_witness(
                item.instance,
                context,
                tight_pre_margin_duration=float(
                    config.geometry.get("witness_tight_pre_margin_duration", 0.25)
                ),
                tight_post_margin_duration=float(
                    config.geometry.get("witness_tight_post_margin_duration", 0.75)
                ),
                loose_pre_margin_s=float(
                    config.geometry.get("witness_loose_pre_margin_s", 30.0)
                ),
            )
            item = replace(item, instance=witness.instance)
        effective_generated.append(item)
        admissibility = proxy_admissibility(item.instance, context)
        expected = item.feasibility_policy in {"admissible_required", "constructive_witness_required"}
        if admissibility.admissible != expected:
            raise ValueError(
                f"feasibility-policy mismatch for {item.instance.instance_id}: "
                f"expected proxy_admissible={expected}"
            )
        relative = Path(instance_directory) / item.split / item.cell_id / f"{item.instance.instance_id}.json"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(item.instance.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        witness_relative = None
        witness_sha = None
        if witness is not None:
            witness_relative = (Path(instance_directory) / "witnesses" / item.split / item.cell_id / f"{item.instance.instance_id}.json")
            witness_destination = root / witness_relative
            witness_destination.parent.mkdir(parents=True, exist_ok=True)
            witness_destination.write_text(json.dumps(witness.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            witness_sha = witness.witness_sha256
        records.append(
            PaperRecord(
                instance_id=item.instance.instance_id,
                split=item.split,
                cell_id=item.cell_id,
                paper_role=item.paper_role,
                task_group_id=item.task_group_id,
                variant_index=item.variant_index,
                seed=item.seed,
                feasibility_policy=item.feasibility_policy,
                proxy_admissible=admissibility.admissible,
                workpiece_id=item.instance.workpiece_id,
                layout_id=item.instance.layout_id,
                parent_curve_ids=tuple(sorted({segment.parent_curve_id for segment in item.instance.segments})),
                evidence_label=item.instance.evidence_label.value,
                robot_count=len(item.instance.robots),
                segment_count=len(item.instance.segments),
                relative_path=relative.as_posix(),
                sha256=hashlib.sha256(canonical_instance_bytes(item.instance)).hexdigest(),
                witness_relative_path=None if witness_relative is None else witness_relative.as_posix(),
                witness_sha256=witness_sha,
            )
        )
    payload = {
        "version": config.manifest_version,
        "generator_version": config.generator_version,
        "config_sha256": config.config_sha256,
        "records": [item.to_dict() for item in records],
    }
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    manifest = PaperManifest(config.manifest_version, config.generator_version, config.config_sha256, tuple(records), digest)
    issues = audit_paper_leakage(manifest.records)
    if issues:
        raise ValueError("paper benchmark leakage: " + ";".join(item.code for item in issues))
    return manifest, tuple(effective_generated)


def write_paper_manifest(manifest: PaperManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_paper_manifest(path: str | Path) -> PaperManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = tuple(
        PaperRecord(
            instance_id=str(item["instance_id"]),
            split=str(item["split"]),
            cell_id=str(item["cell_id"]),
            paper_role=str(item["paper_role"]),
            task_group_id=str(item["task_group_id"]),
            variant_index=int(item["variant_index"]),
            seed=int(item["seed"]),
            feasibility_policy=str(item["feasibility_policy"]),
            proxy_admissible=bool(item["proxy_admissible"]),
            workpiece_id=str(item["workpiece_id"]),
            layout_id=str(item["layout_id"]),
            parent_curve_ids=tuple(str(value) for value in item["parent_curve_ids"]),
            evidence_label=str(item["evidence_label"]),
            robot_count=int(item["robot_count"]),
            segment_count=int(item["segment_count"]),
            relative_path=str(item["relative_path"]),
            sha256=str(item["sha256"]),
            witness_relative_path=None if item.get("witness_relative_path") is None else str(item["witness_relative_path"]),
            witness_sha256=None if item.get("witness_sha256") is None else str(item["witness_sha256"]),
        )
        for item in raw["records"]
    )
    manifest = PaperManifest(str(raw["version"]), str(raw["generator_version"]), str(raw["config_sha256"]), records, str(raw["manifest_sha256"]))
    if manifest.version not in PAPER_MANIFEST_VERSIONS or paper_manifest_digest(manifest) != manifest.manifest_sha256:
        raise ValueError("paper manifest version or digest mismatch")
    return manifest


def paper_manifest_digest(manifest: PaperManifest) -> str:
    payload = {
        "version": manifest.version,
        "generator_version": manifest.generator_version,
        "config_sha256": manifest.config_sha256,
        "records": [item.to_dict() for item in manifest.records],
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def audit_paper_leakage(records: Sequence[PaperRecord]) -> tuple[PaperAuditIssue, ...]:
    issues: list[PaperAuditIssue] = []
    for code, accessor in (
        ("WORKPIECE_LEAKAGE", lambda item: (item.workpiece_id,)),
        ("LAYOUT_LEAKAGE", lambda item: (item.layout_id,)),
        ("TASK_GROUP_LEAKAGE", lambda item: (item.task_group_id,)),
        ("PARENT_CURVE_LEAKAGE", lambda item: item.parent_curve_ids),
    ):
        mapping: dict[str, set[str]] = {}
        for record in records:
            for key in accessor(record):
                mapping.setdefault(key, set()).add(record.split)
        issues.extend(PaperAuditIssue(code, key, tuple(sorted(splits))) for key, splits in sorted(mapping.items()) if len(splits) > 1)
    ids = [item.instance_id for item in records]
    if len(ids) != len(set(ids)):
        issues.append(PaperAuditIssue("DUPLICATE_INSTANCE_ID", "*", ()))
    return tuple(issues)


def verify_paper_instances(
    manifest: PaperManifest, project_root: str | Path
) -> tuple[str, ...]:
    root = Path(project_root)
    package_root = Path(__file__).resolve().parents[3]
    context = load_oracle_context(
        package_root / "configs/allocation/oracle_proxy_v1.json"
    )
    failures: list[str] = []
    for record in manifest.records:
        path = root / record.relative_path
        if not path.is_file():
            failures.append(f"MISSING={record.instance_id}")
            continue
        instance = allocation_instance_from_dict(json.loads(path.read_text(encoding="utf-8")))
        digest = hashlib.sha256(canonical_instance_bytes(instance)).hexdigest()
        if digest != record.sha256:
            failures.append(f"HASH_MISMATCH={record.instance_id}")
        if record.witness_relative_path is not None:
            witness_path = root / record.witness_relative_path
            if not witness_path.is_file():
                failures.append(f"WITNESS_MISSING={record.instance_id}")
                continue
            raw = json.loads(witness_path.read_text(encoding="utf-8"))
            plan = allocation_plan_from_dict(raw["plan"], instance)
            witness = ConstructiveWitness(
                instance,
                plan,
                str(raw["witness_sha256"]),
                tuple(str(item) for item in raw.get("diagnostics", ())),
            )
            issues = verify_constructive_witness(witness, context)
            if record.witness_sha256 != witness.witness_sha256:
                issues = issues + ("MANIFEST_WITNESS_HASH_MISMATCH",)
            failures.extend(f"{issue}={record.instance_id}" for issue in issues)
    return tuple(failures)


def paper_split_counts(records: Sequence[PaperRecord]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in records:
        result[item.split] = result.get(item.split, 0) + 1
    return dict(sorted(result.items()))


def _pair(value: Sequence[Any]) -> tuple[int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 2:
        raise ValueError("range must contain two values")
    return result  # type: ignore[return-value]


def _validate_paper_config(config: PaperBenchmarkConfig) -> None:
    if config.version not in PAPER_CONFIG_VERSIONS or config.manifest_version not in PAPER_MANIFEST_VERSIONS:
        raise ValueError("unsupported paper benchmark version")
    if config.version.rsplit("-", 1)[-1] != config.manifest_version.rsplit("-", 1)[-1]:
        raise ValueError("paper config and manifest versions must match")
    if config.benchmark_tier != "PAPER_SCALE_FROZEN" or config.evidence_label is not EvidenceLabel.SIM_GEOMETRIC:
        raise ValueError("paper benchmark tier/evidence mismatch")
    if {cell.split for cell in config.cells} != {"train", "validation", "frozen_test", "stress"}:
        raise ValueError("paper benchmark requires all four splits")
    seen: set[tuple[str, str]] = set()
    for cell in config.cells:
        key = (cell.split, cell.cell_id)
        if key in seen or cell.groups < 1 or cell.variants < 1:
            raise ValueError("duplicate/empty difficulty cell")
        seen.add(key)
        if cell.feasibility_policy not in {
            "admissible_required",
            "constructive_witness_required",
            "designed_edge_infeasible",
        }:
            raise ValueError("unknown feasibility policy")
        if cell.feasibility_policy == "designed_edge_infeasible" and cell.paper_role != "negative_control":
            raise ValueError("designed infeasibility is restricted to negative controls")


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
