#!/usr/bin/env python3
"""Generate the preregistered A3.5 train/validation-only pilot corpus."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.pointer_pilot import (
    audit_manifest_overlap,
    load_pointer_pilot_config,
    materialize_pointer_pilot,
    write_pointer_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/allocation/a3_5_pointer_pilot_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1_allocation/a3_5_pointer_pilot_v1"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_pointer_pilot_config(_resolve(root, args.config))
    output = _resolve(root, args.output_dir)
    if output.exists():
        raise FileExistsError("A3.5 data output already exists; generation is non-overwriting")
    context = load_oracle_context(root / "configs/allocation/oracle_proxy_v1.json")
    relative = output.relative_to(root) / "data"
    manifest = materialize_pointer_pilot(config, context, root, relative)
    historical = [root / f"data/manifests/allocation/a2_paper_manifest_v{version}.json" for version in (2, 3, 4)]
    overlap = audit_manifest_overlap(manifest, historical)
    if any(overlap.values()):
        raise RuntimeError(f"A3.5 overlaps historical benchmark IDs: {overlap}")
    write_pointer_manifest(manifest, output / "manifest.json")
    counts = {}
    cells = {}
    fallback = 0
    methods = {}
    for item in manifest.records:
        counts[item.split] = counts.get(item.split, 0) + 1
        cells[f"{item.split}:{item.cell_id}"] = cells.get(f"{item.split}:{item.cell_id}", 0) + 1
        fallback += int(item.teacher_fallback)
        methods[item.teacher_method] = methods.get(item.teacher_method, 0) + 1
    summary = {
        "version": "a3-5-pointer-data-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "SIM_GEOMETRIC",
        "development_only": True,
        "config_sha256": config.sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "counts": counts,
        "cells": cells,
        "teacher_methods": dict(sorted(methods.items())),
        "constructive_fallback_count": fallback,
        "historical_id_overlap": overlap,
        "frozen_test_generated": False,
        "stress_generated": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
    }
    (output / "data_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


if __name__ == "__main__":
    main()
