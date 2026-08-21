#!/usr/bin/env python3
"""Audit constructive witness readiness on v3 train/validation only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from safe_residual_rl.allocation import (
    construct_feasible_witness,
    generate_paper_benchmark,
    load_oracle_context,
    load_paper_config,
    load_paper_manifest,
    verify_constructive_witness,
)
from safe_residual_rl.allocation.generation import canonical_instance_bytes


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/allocation/a2_witness_readiness_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["allowed_splits"] != ["train", "validation"] or set(config["forbidden_splits"]) != {"frozen_test", "stress"}:
        raise RuntimeError("witness readiness access policy is invalid")
    paper = load_paper_config(root / config["source_benchmark"])
    manifest = load_paper_manifest(root / config["source_manifest"])
    if manifest.manifest_sha256 != config["source_manifest_sha256"]:
        raise RuntimeError("source manifest mismatch")
    records = {item.instance_id: item for item in manifest.records}
    context = load_oracle_context(root / "configs/allocation/oracle_proxy_v1.json")
    rows = []
    selected = tuple(
        item
        for item in generate_paper_benchmark(paper)
        if item.split in {"train", "validation"}
    )
    for index, generated in enumerate(selected, start=1):
        record = records[generated.instance.instance_id]
        source_hash_ok = hashlib.sha256(canonical_instance_bytes(generated.instance)).hexdigest() == record.sha256
        try:
            first = construct_feasible_witness(
                generated.instance,
                context,
                tight_pre_margin_duration=float(config["tight_pre_margin_duration"]),
                tight_post_margin_duration=float(config["tight_post_margin_duration"]),
                loose_pre_margin_s=float(config["loose_pre_margin_s"]),
            )
            second = construct_feasible_witness(
                generated.instance,
                context,
                tight_pre_margin_duration=float(config["tight_pre_margin_duration"]),
                tight_post_margin_duration=float(config["tight_post_margin_duration"]),
                loose_pre_margin_s=float(config["loose_pre_margin_s"]),
            )
            constructed = True
            verified = not verify_constructive_witness(first, context)
            deterministic = first.witness_sha256 == second.witness_sha256
            semantics_preserved = _non_window_signature(first.instance) == _non_window_signature(generated.instance)
            witness_hash = first.witness_sha256
            error = ""
        except ValueError as exc:
            constructed = verified = deterministic = semantics_preserved = False
            witness_hash = ""
            error = str(exc)
        rows.append({
            "instance_id": generated.instance.instance_id,
            "split": generated.split,
            "cell_id": generated.cell_id,
            "source_hash_ok": source_hash_ok,
            "constructed": constructed,
            "verified": verified,
            "deterministic": deterministic,
            "semantics_preserved": semantics_preserved,
            "witness_sha256": witness_hash,
            "error": error,
        })
        if index % 20 == 0:
            print(json.dumps({"progress": index, "total": 240}), flush=True)
    count = len(rows)
    observed = {
        "construction_rate": sum(item["constructed"] for item in rows) / count,
        "verification_rate": sum(item["verified"] for item in rows) / count,
        "determinism_rate": sum(item["deterministic"] for item in rows) / count,
        "non_window_semantics_preservation_rate": sum(item["semantics_preserved"] for item in rows) / count,
        "zero_hash_failures": all(item["source_hash_ok"] for item in rows),
    }
    checks = {
        key: (value if isinstance(config["acceptance"][key], bool) else value >= config["acceptance"][key])
        for key, value in observed.items()
    }
    result = {
        "schema_version": "a2-witness-readiness-results-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": manifest.manifest_sha256,
        "accessed_splits": ["train", "validation"],
        "instance_count": count,
        "acceptance": {"passed": all(checks.values()), "checks": checks, "observed": observed},
        "boundaries": config["boundaries"],
        "rows": rows,
    }
    output = root / "outputs/phase1_allocation/a2_witness_readiness_v1"
    report = root / "reports/phase1_allocation"
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    compact = dict(result); compact.pop("rows")
    (report / "a2_witness_readiness_v1_summary.json").write_text(json.dumps(compact, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# A2 constructive witness readiness", "", f"Gate: **{'PASSED' if all(checks.values()) else 'FAILED'}**.", "", f"Instances: {count}; accessed splits: train/validation only.", ""]
    lines.extend(f"- {'PASS' if checks[key] else 'FAIL'} `{key}`: {observed[key]}" for key in checks)
    lines.extend(["", "This certifies A1 proxy witness construction only; it is not solver superiority, motion feasibility, collision safety or real execution evidence.", ""])
    (report / "a2_witness_readiness_v1_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result["acceptance"], indent=2, sort_keys=True))


def _non_window_signature(instance):
    value = instance.to_dict()
    for segment in value["segments"]:
        segment.pop("time_window")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
