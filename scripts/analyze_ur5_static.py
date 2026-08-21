#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safe_residual_rl.data import (
    candidate_split_document,
    load_ur5_static_csvs,
    masks_from_candidate_split,
    match_cross_date_static_case,
    per_file_quality,
)
from safe_residual_rl.kinematics import UR5Kinematics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ur5_static_analysis")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_ur5_static_csvs(args.root, joint_unit="degree", length_unit="mm")
    quality = per_file_quality(loaded.dataset, UR5Kinematics())
    matched = match_cross_date_static_case(loaded.dataset)
    candidate = candidate_split_document()
    masks = masks_from_candidate_split(loaded.dataset, candidate)
    candidate["row_counts"] = {role: int(mask.sum()) for role, mask in masks.items()}
    result = {
        "evidence_level": "REAL_STATIC",
        "metadata_status": "units_frames_tcp_and_date_semantics_unverified",
        "git_commit": "unavailable_repository_not_initialized",
        "versions": {"python": sys.version.split()[0], "numpy": importlib.metadata.version("numpy")},
        "source_audit": loaded.audit,
        "per_file": quality,
        "matched_cross_date_case": matched,
        "candidate_split": candidate,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "per_file_quality.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "file", "date_id", "rows", "duplicate_q_tcp_rows", "error_mean_mm", "error_p95_mm", "error_max_mm",
            "fk_nominal_median_mm", "fk_nominal_p95_mm", "fk_compatibility",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in quality:
            writer.writerow(
                {
                    "file": item["file"], "date_id": item["date_id"], "rows": item["rows"],
                    "duplicate_q_tcp_rows": item["duplicate_q_tcp_rows"],
                    "error_mean_mm": item["position_error_norm_mm"]["mean"],
                    "error_p95_mm": item["position_error_norm_mm"]["p95"],
                    "error_max_mm": item["position_error_norm_mm"]["max"],
                    "fk_nominal_median_mm": item["fk_nominal_delta_mm"]["median"],
                    "fk_nominal_p95_mm": item["fk_nominal_delta_mm"]["p95"],
                    "fk_compatibility": item["fk_compatibility"],
                }
            )
    (args.output_dir / "candidate_split.json").write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    case = matched["paired_error_change_norm_mm"]
    report = f"""# UR5 REAL_STATIC detailed analysis

All units (`degree`, `mm`), coordinate frames, TCP definitions and date semantics remain **unverified**.

- Included rows/files: {loaded.audit['total_rows']} / {loaded.audit['session_count']}
- Cross-date matched case: {matched['matched_pairs']} pairs from {matched['first_rows']} and {matched['second_rows']} rows.
- Matched error-vector change norm: mean {case['mean']:.3f} mm, P95 {case['p95']:.3f} mm, max {case['max']:.3f} mm.
- Interpretation: matched static repeat case study; not a continuous trajectory and not RL transitions.
- `20250714/建模数据.csv` is reserved as a frame/TCP-shift diagnostic because FK/nominal mismatch is about 50 mm.
- Candidate split hash: `{candidate['sha256']}`; status: `{candidate['status']}`.

Candidate row counts: `{candidate['row_counts']}`.

`data_all.csv` and `data08.csv` remain excluded. See `per_file_quality.csv` and `analysis.json` for full workspace, component-error and FK diagnostics.
"""
    (args.output_dir / "analysis.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
