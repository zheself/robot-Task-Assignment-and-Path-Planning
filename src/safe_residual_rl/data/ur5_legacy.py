"""Read-only adapter for the group's heterogeneous UR5 static CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .synthetic import MeasurementDataset


@dataclass(frozen=True)
class UR5LegacyLoadResult:
    dataset: MeasurementDataset
    audit: dict


def _joint_columns(fieldnames: list[str]) -> list[str] | None:
    for prefix in ("θ", "a"):
        names = [f"{prefix}{index}" for index in range(1, 7)]
        if all(name in fieldnames for name in names):
            return names
    return None


def load_ur5_static_csvs(root: Path, *, joint_unit: str, length_unit: str) -> UR5LegacyLoadResult:
    """Load static positions while preserving file-level groups.

    Units are mandatory runtime assertions because the source files do not
    carry reliable machine-readable unit metadata.
    """
    if joint_unit not in ("degree", "radian") or length_unit not in ("mm", "m"):
        raise ValueError("joint_unit must be degree/radian and length_unit must be mm/m")
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    q_rows, nominal_rows, measured_rows = [], [], []
    sessions, paths, dates = [], [], []
    files = []
    exclusions = []
    for path in sorted(root.rglob("*.csv")):
        relative = path.relative_to(root).as_posix()
        if path.name == "data_all.csv":
            exclusions.append({"file": relative, "reason": "merged_duplicate_risk"})
            continue
        if path.name == "data08.csv":
            exclusions.append({"file": relative, "reason": "real_xyz_header_order_suspect"})
            continue
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [name for name in (reader.fieldnames or []) if name]
            joints = _joint_columns(fieldnames)
            required = ["x", "y", "z", "x-real", "y-real", "z-real"]
            if joints is None or not all(name in fieldnames for name in required):
                exclusions.append({"file": relative, "reason": "unsupported_or_missing_required_columns"})
                continue
            accepted, rejected = 0, 0
            for row in reader:
                try:
                    q = np.array([float(row[name]) for name in joints], dtype=float)
                    nominal = np.array([float(row[name]) for name in ("x", "y", "z")], dtype=float)
                    measured = np.array([float(row[name]) for name in ("x-real", "y-real", "z-real")], dtype=float)
                except (TypeError, ValueError, KeyError):
                    rejected += 1
                    continue
                if not np.isfinite(q).all() or not np.isfinite(nominal).all() or not np.isfinite(measured).all():
                    rejected += 1
                    continue
                if joint_unit == "degree":
                    q = np.deg2rad(q)
                if length_unit == "mm":
                    nominal, measured = nominal / 1000.0, measured / 1000.0
                q_rows.append(q)
                nominal_rows.append(nominal)
                measured_rows.append(measured)
                session_id = f"ur5::{relative}"
                sessions.append(session_id)
                paths.append(session_id)
                parent = path.parent.name
                dates.append(parent if len(parent) == 8 and parent.isdigit() else "unverified_date")
                accepted += 1
            files.append(
                {
                    "file": relative,
                    "accepted_rows": accepted,
                    "rejected_rows": rejected,
                    "joint_columns": joints,
                    "date_id": dates[-1] if accepted else "unverified_date",
                    "sequence_semantics": "ordered_rows_not_assumed_to_be_transitions",
                }
            )
    if not q_rows:
        raise ValueError(f"no usable UR5 rows under {root}")
    dataset = MeasurementDataset(
        q_rad=np.vstack(q_rows), x_nominal_m=np.vstack(nominal_rows), x_measured_m=np.vstack(measured_rows),
        session_id=np.asarray(sessions), path_id=np.asarray(paths), date_id=np.asarray(dates),
        evidence_level="REAL_STATIC_UNVERIFIED_METADATA",
    )
    dataset.validate()
    error_norm_mm = np.linalg.norm(dataset.error_m, axis=1) * 1000.0
    audit = {
        "robot_id": "ur5",
        "evidence_level": dataset.evidence_level,
        "source_root": str(root.resolve()),
        "runtime_unit_assertions": {"joint": joint_unit, "length": length_unit, "verification": "unverified"},
        "included_files": files,
        "excluded_files": exclusions,
        "total_rows": len(dataset.q_rad),
        "session_count": len(set(dataset.session_id)),
        "date_ids": sorted(set(dataset.date_id)),
        "position_error_norm_mm": {
            "mean": float(np.mean(error_norm_mm)),
            "median": float(np.median(error_norm_mm)),
            "p95": float(np.percentile(error_norm_mm, 95)),
            "max": float(np.max(error_norm_mm)),
        },
        "transition_status": "not_an_offline_rl_dataset",
    }
    return UR5LegacyLoadResult(dataset=dataset, audit=audit)
