#!/usr/bin/env python3
"""Run the bounded joint proxy reference on fixtures and v3-train only."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from safe_residual_rl.allocation import (
    generate_paper_benchmark,
    load_oracle_context,
    load_paper_config,
    load_paper_manifest,
    solve_joint_assignment_sequence_reference,
    verify_plan,
)
from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.generation import canonical_instance_bytes
from safe_residual_rl.allocation.schema import allocation_instance_from_dict


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "configs/allocation/a2_joint_reference_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["allowed_benchmark_split"] != "train" or set(protocol["forbidden_splits"]) != {
        "validation",
        "frozen_test",
        "stress",
    }:
        raise RuntimeError("joint reference access policy is not train-only")
    context = load_oracle_context(root / "configs/allocation/oracle_proxy_v1.json")
    rows = []

    fixture_dir = root / "data/fixtures/allocation"
    for path in sorted(fixture_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not raw.get("expected", {}).get("valid", False):
            continue
        loaded = load_auditable_fixture(path)
        instance = allocation_instance_from_dict(loaded["instance"])
        rows.append(_run("fixture", path.name, instance, context, protocol))

    paper_config = load_paper_config(root / protocol["benchmark_source"])
    manifest = load_paper_manifest(root / protocol["manifest_source"])
    if manifest.manifest_sha256 != protocol["manifest_sha256"]:
        raise RuntimeError("joint reference manifest hash mismatch")
    record_by_id = {item.instance_id: item for item in manifest.records}
    candidates = sorted(
        (
            item
            for item in generate_paper_benchmark(paper_config)
            if item.split == "train"
            and len(item.instance.segments) <= int(protocol["max_segments"])
        ),
        key=lambda item: (len(item.instance.segments), item.instance.instance_id),
    )[: int(protocol["train_sample_count"])]
    if len(candidates) != int(protocol["train_sample_count"]):
        raise RuntimeError("insufficient registered v3-train small instances")
    for item in candidates:
        record = record_by_id[item.instance.instance_id]
        digest = hashlib.sha256(canonical_instance_bytes(item.instance)).hexdigest()
        if record.split != "train" or record.sha256 != digest:
            raise RuntimeError("generated train instance differs from frozen manifest")
        rows.append(_run("v3_train", item.instance.instance_id, item.instance, context, protocol))

    fixture_rows = [item for item in rows if item["source"] == "fixture"]
    train_rows = [item for item in rows if item["source"] == "v3_train"]
    acceptance = {
        "all_valid_fixtures_complete_and_verified": all(
            item["status"] == "optimal" and item["verified"] for item in fixture_rows
        ),
        "minimum_train_complete_rate": (
            sum(item["status"] == "optimal" and item["verified"] for item in train_rows)
            / len(train_rows)
            >= float(protocol["acceptance"]["minimum_train_complete_rate"])
        ),
        "zero_verification_failures": not any(
            item["status"] == "verification_failed" for item in rows
        ),
    }
    result = {
        "schema_version": "a2-joint-reference-results-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": protocol["evidence_label"],
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "manifest_sha256": manifest.manifest_sha256,
        "accessed_benchmark_splits": ["train"],
        "acceptance": {"passed": all(acceptance.values()), "checks": acceptance},
        "boundaries": protocol["boundaries"],
        "rows": rows,
    }
    output_dir = root / "outputs/phase1_allocation/a2_joint_reference_v1"
    report_dir = root / "reports/phase1_allocation"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_dir / "results.csv", rows)
    compact = dict(result)
    compact.pop("rows")
    compact["rows"] = rows
    (report_dir / "a2_joint_reference_v1_summary.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True), encoding="utf-8"
    )
    (report_dir / "a2_joint_reference_v1_results.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    print(json.dumps(result["acceptance"], indent=2, sort_keys=True))


def _run(source, case_id, instance, context, protocol):
    result = solve_joint_assignment_sequence_reference(
        instance,
        context,
        max_segments=int(protocol["max_segments"]),
        max_assignment_combinations=int(protocol["max_assignment_combinations"]),
        node_limit=int(protocol["node_limit"]),
        time_limit_s=float(protocol["time_limit_s"]),
    )
    verification = verify_plan(instance, result.plan, context) if result.plan else None
    return {
        "source": source,
        "case_id": case_id,
        "segment_count": len(instance.segments),
        "robot_count": len(instance.robots),
        "status": result.status,
        "verified": bool(verification and verification.feasible),
        "runtime_s": result.runtime_s,
        "objective_value": result.objective_value,
        "diagnostics": "|".join(result.diagnostics),
    }


def _markdown(result):
    lines = [
        "# A2 bounded joint reference results",
        "",
        "Evidence: `SIM_GEOMETRIC`; benchmark access: v3 `train` only.",
        f"Acceptance: **{'PASSED' if result['acceptance']['passed'] else 'FAILED'}**.",
        "",
        "| source | case | segments | robots | status | verified | runtime s |",
        "|---|---|---:|---:|---|---|---:|",
    ]
    for item in result["rows"]:
        lines.append(
            f"| {item['source']} | {item['case_id']} | {item['segment_count']} | "
            f"{item['robot_count']} | {item['status']} | {item['verified']} | {item['runtime_s']:.6f} |"
        )
    lines.extend(
        [
            "",
            "`optimal` means complete enumeration only inside the A1 assignment/timing proxy. "
            "It is not motion-planning, collision, process-physics or factory optimality.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
