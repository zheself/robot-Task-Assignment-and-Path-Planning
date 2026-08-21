"""Split-guarded A3 instance/witness access."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..oracle import OracleContext
from ..schema import AllocationInstance, AllocationPlan, allocation_instance_from_dict, allocation_plan_from_dict
from ..witness import ConstructiveWitness, verify_constructive_witness

ALLOWED_DEVELOPMENT_SPLITS = frozenset({"train", "validation"})
FORBIDDEN_SPLITS = frozenset({"frozen_test", "stress"})


@dataclass(frozen=True)
class A3GraphRecord:
    split: str
    cell_id: str
    instance_path: Path
    witness_path: Path
    instance: AllocationInstance
    teacher_plan: AllocationPlan
    teacher_sha256: str
    record_sha256: str


def discover_a3_records(
    instance_root: str | Path,
    split: str,
    context: OracleContext,
) -> tuple[A3GraphRecord, ...]:
    if split not in ALLOWED_DEVELOPMENT_SPLITS:
        raise PermissionError(f"A3 development forbids split access: {split}")
    root = Path(instance_root).resolve()
    split_root = (root / split).resolve()
    witness_root = (root / "witnesses" / split).resolve()
    _require_split_path(split_root, split)
    _require_split_path(witness_root, split)
    records = []
    for instance_path in sorted(split_root.glob("*/*.json")):
        cell_id = instance_path.parent.name
        witness_path = witness_root / cell_id / instance_path.name
        records.append(load_a3_record(instance_path, witness_path, split, cell_id, context))
    if not records:
        raise ValueError(f"no A3 {split} records found under {root}")
    return tuple(records)


def load_a3_record(
    instance_path: str | Path,
    witness_path: str | Path,
    split: str,
    cell_id: str,
    context: OracleContext,
) -> A3GraphRecord:
    if split not in ALLOWED_DEVELOPMENT_SPLITS:
        raise PermissionError(f"A3 development forbids split access: {split}")
    instance_path = Path(instance_path).resolve()
    witness_path = Path(witness_path).resolve()
    _require_split_path(instance_path, split)
    _require_split_path(witness_path, split)
    if not witness_path.is_file():
        raise FileNotFoundError(f"missing A3 teacher witness for {instance_path.name}")
    instance_raw = json.loads(instance_path.read_text(encoding="utf-8"))
    witness_raw = json.loads(witness_path.read_text(encoding="utf-8"))
    instance = allocation_instance_from_dict(instance_raw)
    if witness_raw.get("instance_id") != instance.instance_id:
        raise ValueError("teacher witness instance ID mismatch")
    instance_digest = hashlib.sha256(
        json.dumps(instance.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if witness_raw.get("instance_sha256") != instance_digest:
        raise ValueError("teacher witness instance hash mismatch")
    plan = allocation_plan_from_dict(witness_raw["plan"], instance)
    teacher_sha256 = str(witness_raw["witness_sha256"])
    witness = ConstructiveWitness(
        instance,
        plan,
        teacher_sha256,
        tuple(str(item) for item in witness_raw.get("diagnostics", ())),
    )
    issues = verify_constructive_witness(witness, context)
    if issues:
        raise ValueError("teacher witness verification failed: " + ",".join(issues))
    record_payload = {
        "split": split,
        "cell_id": cell_id,
        "instance_id": instance.instance_id,
        "instance_sha256": instance_digest,
        "teacher_sha256": teacher_sha256,
    }
    record_sha256 = hashlib.sha256(
        json.dumps(record_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return A3GraphRecord(
        split,
        cell_id,
        instance_path,
        witness_path,
        instance,
        plan,
        teacher_sha256,
        record_sha256,
    )


def _require_split_path(path: Path, split: str) -> None:
    parts = set(path.parts)
    if split not in parts:
        raise PermissionError(f"path is not explicitly scoped to {split}: {path}")
    forbidden = parts & FORBIDDEN_SPLITS
    if forbidden:
        raise PermissionError(f"forbidden split token in path: {sorted(forbidden)}")
