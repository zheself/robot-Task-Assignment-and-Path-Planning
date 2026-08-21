"""A2 benchmark materialisation, immutable manifests and leakage audits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .generation import BenchmarkConfig, GeneratedInstance, canonical_instance_bytes, generate_benchmark
from .schema import AllocationInstance, allocation_instance_from_dict

MANIFEST_VERSION = "a2-split-manifest-v1"


@dataclass(frozen=True)
class BenchmarkRecord:
    instance_id: str
    split: str
    family: str
    task_group_id: str
    variant_index: int
    seed: int
    workpiece_id: str
    layout_id: str
    parent_curve_ids: tuple[str, ...]
    evidence_label: str
    robot_count: int
    segment_count: int
    relative_path: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["parent_curve_ids"] = list(self.parent_curve_ids)
        return result


@dataclass(frozen=True)
class LeakageIssue:
    code: str
    key: str
    splits: tuple[str, ...]


class SplitAccessError(ValueError):
    """Raised when a caller attempts to use a split for the wrong purpose."""


@dataclass(frozen=True)
class BenchmarkManifest:
    version: str
    generator_version: str
    config_sha256: str
    records: tuple[BenchmarkRecord, ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generator_version": self.generator_version,
            "config_sha256": self.config_sha256,
            "records": [item.to_dict() for item in self.records],
            "manifest_sha256": self.manifest_sha256,
        }


def materialize_benchmark(
    config: BenchmarkConfig,
    project_root: str | Path,
    instance_directory: str | Path = "outputs/phase1_allocation/a2_benchmark/instances",
) -> tuple[BenchmarkManifest, tuple[GeneratedInstance, ...]]:
    root = Path(project_root)
    output = root / instance_directory
    generated = generate_benchmark(config)
    records: list[BenchmarkRecord] = []
    for item in generated:
        relative_path = Path(instance_directory) / item.split / f"{item.instance.instance_id}.json"
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(item.instance.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        records.append(
            BenchmarkRecord(
                instance_id=item.instance.instance_id,
                split=item.split,
                family=item.family,
                task_group_id=item.task_group_id,
                variant_index=item.variant_index,
                seed=item.seed,
                workpiece_id=item.instance.workpiece_id,
                layout_id=item.instance.layout_id,
                parent_curve_ids=tuple(sorted({segment.parent_curve_id for segment in item.instance.segments})),
                evidence_label=item.instance.evidence_label.value,
                robot_count=len(item.instance.robots),
                segment_count=len(item.instance.segments),
                relative_path=relative_path.as_posix(),
                sha256=hashlib.sha256(canonical_instance_bytes(item.instance)).hexdigest(),
            )
        )
    config_hash = hashlib.sha256(_canonical_config(config)).hexdigest()
    payload = {
        "version": MANIFEST_VERSION,
        "generator_version": "a2-continuous-generator-v1",
        "config_sha256": config_hash,
        "records": [item.to_dict() for item in records],
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    manifest = BenchmarkManifest(MANIFEST_VERSION, payload["generator_version"], config_hash, tuple(records), digest)
    issues = audit_split_leakage(manifest.records)
    if issues:
        raise ValueError("generated split leakage: " + ";".join(item.code for item in issues))
    return manifest, generated


def write_manifest(manifest: BenchmarkManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: str | Path) -> BenchmarkManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = tuple(
        BenchmarkRecord(
            instance_id=str(item["instance_id"]),
            split=str(item["split"]),
            family=str(item["family"]),
            task_group_id=str(item["task_group_id"]),
            variant_index=int(item["variant_index"]),
            seed=int(item["seed"]),
            workpiece_id=str(item["workpiece_id"]),
            layout_id=str(item["layout_id"]),
            parent_curve_ids=tuple(str(value) for value in item["parent_curve_ids"]),
            evidence_label=str(item["evidence_label"]),
            robot_count=int(item["robot_count"]),
            segment_count=int(item["segment_count"]),
            relative_path=str(item["relative_path"]),
            sha256=str(item["sha256"]),
        )
        for item in data["records"]
    )
    manifest = BenchmarkManifest(str(data["version"]), str(data["generator_version"]), str(data["config_sha256"]), records, str(data["manifest_sha256"]))
    if manifest.version != MANIFEST_VERSION or manifest.manifest_sha256 != manifest_digest(manifest):
        raise ValueError("manifest version or digest mismatch")
    return manifest


def manifest_digest(manifest: BenchmarkManifest) -> str:
    payload = {
        "version": manifest.version,
        "generator_version": manifest.generator_version,
        "config_sha256": manifest.config_sha256,
        "records": [item.to_dict() for item in manifest.records],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def verify_materialized_instances(
    manifest: BenchmarkManifest, project_root: str | Path
) -> tuple[str, ...]:
    failures: list[str] = []
    root = Path(project_root)
    for record in manifest.records:
        path = root / record.relative_path
        if not path.is_file():
            failures.append(f"MISSING={record.instance_id}")
            continue
        instance = allocation_instance_from_dict(json.loads(path.read_text(encoding="utf-8")))
        digest = hashlib.sha256(canonical_instance_bytes(instance)).hexdigest()
        if digest != record.sha256:
            failures.append(f"HASH_MISMATCH={record.instance_id}")
    return tuple(failures)


def audit_split_leakage(records: Sequence[BenchmarkRecord]) -> tuple[LeakageIssue, ...]:
    issues: list[LeakageIssue] = []
    _audit_key(records, "WORKPIECE_LEAKAGE", lambda item: (item.workpiece_id,), issues)
    _audit_key(records, "LAYOUT_LEAKAGE", lambda item: (item.layout_id,), issues)
    _audit_key(records, "TASK_GROUP_LEAKAGE", lambda item: (item.task_group_id,), issues)
    _audit_key(records, "PARENT_CURVE_LEAKAGE", lambda item: item.parent_curve_ids, issues)
    instance_ids = [item.instance_id for item in records]
    if len(instance_ids) != len(set(instance_ids)):
        issues.append(LeakageIssue("DUPLICATE_INSTANCE_ID", "*", ()))
    return tuple(issues)


def split_counts(records: Iterable[BenchmarkRecord]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in records:
        result[item.split] = result.get(item.split, 0) + 1
    return dict(sorted(result.items()))


def select_records(
    manifest: BenchmarkManifest, split: str, purpose: str
) -> tuple[BenchmarkRecord, ...]:
    allowed = {
        "train_fit": "train",
        "validation_select": "validation",
        "frozen_evaluate": "frozen_test",
        "stress_evaluate": "stress",
    }
    if allowed.get(purpose) != split:
        raise SplitAccessError(f"purpose={purpose} cannot access split={split}")
    return tuple(item for item in manifest.records if item.split == split)


def _audit_key(records, code, keys, issues) -> None:
    split_by_key: dict[str, set[str]] = {}
    for record in records:
        for key in keys(record):
            split_by_key.setdefault(key, set()).add(record.split)
    for key, splits in sorted(split_by_key.items()):
        if len(splits) > 1:
            issues.append(LeakageIssue(code, key, tuple(sorted(splits))))


def _canonical_config(config: BenchmarkConfig) -> bytes:
    data = {
        "version": config.version,
        "manifest_version": config.manifest_version,
        "master_seed": config.master_seed,
        "evidence_label": config.evidence_label.value,
        "coordinate_frame": config.coordinate_frame,
        "variants_per_group": config.variants_per_group,
        "geometry": dict(config.geometry),
        "splits": {name: spec.__dict__ for name, spec in config.splits},
        "objective_weights": dict(config.objective_weights),
        "baseline_protocol": dict(config.baseline_protocol),
        "boundaries": list(config.boundaries),
    }
    return _canonical_json(data)


def _canonical_json(data: Mapping[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
