"""Quality analysis and provisional grouping for UR5 REAL_STATIC data."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from safe_residual_rl.kinematics import UR5Kinematics

from .synthetic import MeasurementDataset


def metric_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def per_file_quality(dataset: MeasurementDataset, kinematics: UR5Kinematics) -> list[dict]:
    fk = kinematics.position_batch(dataset.q_rad)
    reports = []
    for session in sorted(set(dataset.session_id)):
        mask = dataset.session_id == session
        q = dataset.q_rad[mask]
        nominal = dataset.x_nominal_m[mask]
        error_mm = dataset.error_m[mask] * 1000.0
        error_norm = np.linalg.norm(error_mm, axis=1)
        fk_delta = np.linalg.norm(fk[mask] - nominal, axis=1) * 1000.0
        unique_rows = np.unique(np.round(np.column_stack((q, nominal)), decimals=10), axis=0)
        reports.append(
            {
                "file": session.split("ur5::", 1)[-1],
                "session_id": session,
                "date_id": str(dataset.date_id[mask][0]),
                "rows": int(mask.sum()),
                "duplicate_q_tcp_rows": int(mask.sum() - len(unique_rows)),
                "q_range_deg": {
                    "min": np.rad2deg(q.min(axis=0)).tolist(),
                    "max": np.rad2deg(q.max(axis=0)).tolist(),
                },
                "nominal_tcp_range_mm": {
                    "min": (nominal.min(axis=0) * 1000.0).tolist(),
                    "max": (nominal.max(axis=0) * 1000.0).tolist(),
                },
                "position_error_norm_mm": metric_summary(error_norm),
                "position_error_components_mm": {
                    axis: {
                        "mean": float(error_mm[:, index].mean()),
                        "std": float(error_mm[:, index].std()),
                        "rmse": float(np.sqrt(np.mean(error_mm[:, index] ** 2))),
                        "min": float(error_mm[:, index].min()),
                        "max": float(error_mm[:, index].max()),
                    }
                    for index, axis in enumerate("xyz")
                },
                "fk_nominal_delta_mm": metric_summary(fk_delta),
                "fk_compatibility": (
                    "candidate_compatible_unverified" if np.median(fk_delta) < 3.0 else "frame_or_tcp_mismatch_candidate"
                ),
            }
        )
    return reports


def match_cross_date_static_case(
    dataset: MeasurementDataset,
    first_suffix: str = "20250806/10.csv",
    second_suffix: str = "20250807/10Pos.csv",
    max_joint_difference_deg: float = 0.02,
    max_nominal_tcp_difference_mm: float = 0.2,
) -> dict:
    first_mask = np.array([str(value).endswith(first_suffix) for value in dataset.session_id])
    second_mask = np.array([str(value).endswith(second_suffix) for value in dataset.session_id])
    if not first_mask.any() or not second_mask.any():
        raise ValueError("cross-date case-study files are missing")
    q_a, x_a, e_a = dataset.q_rad[first_mask], dataset.x_nominal_m[first_mask], dataset.error_m[first_mask]
    q_b, x_b, e_b = dataset.q_rad[second_mask], dataset.x_nominal_m[second_mask], dataset.error_m[second_mask]
    max_joint = np.max(np.abs(q_a[:, None, :] - q_b[None, :, :]), axis=2)
    candidates = []
    for index_a in range(len(q_a)):
        index_b = int(np.argmin(max_joint[index_a]))
        joint_difference_deg = float(np.rad2deg(max_joint[index_a, index_b]))
        tcp_difference_mm = float(np.linalg.norm(x_a[index_a] - x_b[index_b]) * 1000.0)
        if joint_difference_deg <= max_joint_difference_deg and tcp_difference_mm <= max_nominal_tcp_difference_mm:
            candidates.append((index_a, index_b, joint_difference_deg, tcp_difference_mm))
    # Enforce one-to-one matching; the present files have an unambiguous mapping.
    if len({item[1] for item in candidates}) != len(candidates):
        raise ValueError("nearest matching is not one-to-one")
    pairs = []
    changes = []
    for index_a, index_b, joint_difference_deg, tcp_difference_mm in candidates:
        first_error = e_a[index_a] * 1000.0
        second_error = e_b[index_b] * 1000.0
        change = second_error - first_error
        changes.append(change)
        pairs.append(
            {
                "first_index": index_a,
                "second_index": index_b,
                "max_joint_difference_deg": joint_difference_deg,
                "nominal_tcp_difference_mm": tcp_difference_mm,
                "first_error_mm": first_error.tolist(),
                "second_error_mm": second_error.tolist(),
                "paired_error_change_mm": change.tolist(),
                "paired_error_change_norm_mm": float(np.linalg.norm(change)),
            }
        )
    changes_array = np.asarray(changes)
    change_norm = np.linalg.norm(changes_array, axis=1)
    return {
        "evidence_level": "REAL_STATIC",
        "interpretation": "matched_static_repeat_case_study_not_trajectory_or_transition",
        "metadata_verification": "unverified",
        "first_file": first_suffix,
        "second_file": second_suffix,
        "first_rows": len(q_a),
        "second_rows": len(q_b),
        "matched_pairs": len(pairs),
        "unmatched_first_rows": len(q_a) - len(pairs),
        "thresholds": {
            "max_joint_difference_deg": max_joint_difference_deg,
            "max_nominal_tcp_difference_mm": max_nominal_tcp_difference_mm,
        },
        "paired_error_change_mean_mm": changes_array.mean(axis=0).tolist(),
        "paired_error_change_norm_mm": metric_summary(change_norm),
        "pairs": pairs,
    }


def candidate_split_document() -> dict:
    document = {
        "candidate_id": "ur5_real_static_candidate_v1",
        "status": "candidate_unverified_not_frozen",
        "evidence_level": "REAL_STATIC",
        "robot_id": "ur5",
        "runtime_assumptions": {
            "joint_unit": "degree_unverified",
            "length_unit": "mm_unverified",
            "coordinate_frame": "unverified",
            "tcp": "unverified",
            "date_semantics": "unverified",
        },
        "roles": {
            "train": [
                "data01.csv", "data02.csv", "data03.csv", "data04.csv", "data05.csv", "data06.csv", "data07.csv",
                "20250806/建模数据1.csv",
            ],
            "validation": ["20250806/验证数据.csv"],
            "test_cross_date": ["20250807/验证数据(1).csv"],
            "reserved_matched_case_study": ["20250806/10.csv", "20250807/10Pos.csv"],
            "reserved_additional": ["20250807/建模数据.csv"],
            "external_frame_shift_diagnostic": ["20250714/建模数据.csv"],
            "excluded": ["data_all.csv", "data08.csv"],
        },
        "rationale": {
            "grouping": "whole source files only; no row-level random split",
            "test": "filename-labeled validation file from a held-out date",
            "case_study": "matched static repeat points kept out of model selection",
            "external": "20250714 has approximately 50 mm FK/nominal mismatch and is not simulator-compatible",
        },
        "warning": "Do not promote this document to a final split manifest before metadata confirmation.",
    }
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    document["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    document["hash_scope"] = "candidate_definition_before_sha256_and_runtime_row_counts"
    return document


def masks_from_candidate_split(dataset: MeasurementDataset, document: dict) -> dict[str, np.ndarray]:
    file_names = np.array([str(value).split("ur5::", 1)[-1] for value in dataset.session_id])
    masks = {}
    for role, files in document["roles"].items():
        masks[role] = np.isin(file_names, files)
    primary = ("train", "validation", "test_cross_date")
    for first_index, first in enumerate(primary):
        for second in primary[first_index + 1 :]:
            if np.any(masks[first] & masks[second]):
                raise ValueError(f"candidate split leakage between {first} and {second}")
    return masks
