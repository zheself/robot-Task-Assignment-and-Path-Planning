#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safe_residual_rl.data import load_ur5_static_csvs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--joint-unit", choices=("degree", "radian"), required=True)
    parser.add_argument("--length-unit", choices=("mm", "m"), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ur5_legacy_audit")
    args = parser.parse_args()
    result = load_ur5_static_csvs(args.root, joint_unit=args.joint_unit, length_unit=args.length_unit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(result.audit, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = result.audit["position_error_norm_mm"]
    excluded = "\n".join(f"- `{item['file']}`: {item['reason']}" for item in result.audit["excluded_files"])
    report = f"""# UR5 legacy static-data audit

Evidence: **REAL_STATIC_UNVERIFIED_METADATA**. Runtime assumptions are `{args.joint_unit}` and `{args.length_unit}`; these are not yet confirmed metadata.

- Included rows: {result.audit['total_rows']}
- Included sessions/files: {result.audit['session_count']}
- Date IDs: {', '.join(result.audit['date_ids'])}
- Position-error norm: mean {metrics['mean']:.3f} mm, median {metrics['median']:.3f} mm, P95 {metrics['p95']:.3f} mm, max {metrics['max']:.3f} mm
- RL status: `{result.audit['transition_status']}`

## Automatic exclusions

{excluded}

Rows remain grouped by source file. Their row order is preserved but is not interpreted as state transitions. No train/test split has been frozen because file semantics, coordinates and dates still require confirmation.
"""
    (args.output_dir / "audit.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
