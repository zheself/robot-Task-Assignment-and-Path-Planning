#!/usr/bin/env python3
"""Create a dependency-free preliminary inventory of legacy CSV measurements.

This script does not modify source files and does not resolve semantic issues.
Its output is an audit queue, not a finalized data card.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ANGLE_PREFIXES = ("θ", "q", "a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path, help="Legacy DATA/DATA directory")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/data_audit"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[list[str]], str]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as stream:
                rows = list(csv.reader(stream))
            if not rows:
                return [], [], encoding
            return [item.strip() for item in rows[0]], rows[1:], encoding
        except UnicodeError as exc:
            last_error = exc
    raise RuntimeError(f"Cannot decode {path}: {last_error}")


def as_float(value: str) -> float | None:
    try:
        result = float(value.strip())
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metrics(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "median": median(values) if values else None,
        "p95": quantile(values, 0.95),
        "max": max(values) if values else None,
    }


def infer_robot(relative: Path) -> str:
    text = str(relative).lower()
    if "ur5" in text:
        return "ur5"
    if "kr210" in text and "r2700" in text:
        return "kuka_kr210_r2700_2"
    if "kr210" in text and "r3100" in text:
        return "kuka_kr210_r3100_2"
    if "kr240" in text or "r3330" in text:
        return "kuka_kr240_r3330"
    return "unverified"


def column_index(header: list[str], name: str) -> int | None:
    try:
        return header.index(name)
    except ValueError:
        return None


def error_norms(
    header: list[str], rows: Iterable[list[str]], *, actual_indices: tuple[int, int, int]
) -> list[float]:
    nominal = tuple(column_index(header, axis) for axis in ("x", "y", "z"))
    if any(index is None for index in nominal):
        return []
    norms: list[float] = []
    for row in rows:
        required = (*nominal, *actual_indices)
        if max(int(index) for index in required if index is not None) >= len(row):
            continue
        nominal_values = [as_float(row[int(index)]) for index in nominal]
        actual_values = [as_float(row[index]) for index in actual_indices]
        if any(value is None for value in (*nominal_values, *actual_values)):
            continue
        norm = math.sqrt(
            sum((float(real) - float(nom)) ** 2 for nom, real in zip(nominal_values, actual_values))
        )
        norms.append(norm)
    return norms


def joint_columns(header: list[str]) -> list[str]:
    result = []
    for index in range(1, 7):
        candidates = {f"{prefix}{index}" for prefix in ANGLE_PREFIXES}
        result.extend(item for item in header if item in candidates)
    return result


def inspect_file(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root)
    header, rows, encoding = read_csv(path)
    warnings: list[str] = []
    lower_name = path.name.lower()
    if "data_all" in lower_name:
        warnings.append("merged_file_candidate_do_not_combine_with_components")
    if relative.parent == Path(".") and path.name in {"train00.csv", "test00.csv"}:
        warnings.append("unknown_robot_frame_and_abnormal_scale_requires_confirmation")

    declared_indices = tuple(column_index(header, f"{axis}-real") for axis in ("x", "y", "z"))
    declared = []
    if all(index is not None for index in declared_indices):
        declared = error_norms(header, rows, actual_indices=tuple(int(i) for i in declared_indices))
    else:
        warnings.append("missing_or_nonstandard_real_xyz_headers")

    positional = []
    if len(header) >= 3:
        positional = error_norms(
            header,
            rows,
            actual_indices=(len(header) - 3, len(header) - 2, len(header) - 1),
        )
    declared_stats = metrics(declared)
    positional_stats = metrics(positional)

    if header[-3:] != ["x-real", "y-real", "z-real"] and any("-real" in item for item in header[-3:]):
        warnings.append(f"real_column_order_is_{header[-3:]}")
    if declared and positional:
        declared_median = float(declared_stats["median"])
        positional_median = float(positional_stats["median"])
        if declared_median > max(10.0, positional_median * 5.0):
            warnings.append("declared_header_mapping_much_worse_than_positional_xyz_mapping")

    nonempty_rows = [row for row in rows if any(cell.strip() for cell in row)]
    first_time = nonempty_rows[0][0] if nonempty_rows and nonempty_rows[0] else None
    last_time = nonempty_rows[-1][0] if nonempty_rows and nonempty_rows[-1] else None
    return {
        "relative_path": str(relative),
        "robot_id": infer_robot(relative),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "encoding": encoding,
        "row_count": len(nonempty_rows),
        "columns": header,
        "joint_columns": joint_columns(header),
        "first_field_first_row": first_time,
        "first_field_last_row": last_time,
        "declared_xyz_error_source_unit": declared_stats,
        "positional_last3_xyz_error_source_unit": positional_stats,
        "warnings": warnings,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Preliminary Legacy Data Inventory",
        "",
        "> Generated automatically. Units, frames, and semantics remain unverified.",
        "",
        f"- Source root: `{report['source_root']}`",
        f"- CSV files: {report['csv_file_count']}",
        f"- Rows: {report['total_rows']}",
        f"- Exact duplicate hash groups: {len(report['duplicate_hash_groups'])}",
        "",
        "## Robot summary",
        "",
        "| Robot | Files | Rows |",
        "|---|---:|---:|",
    ]
    for robot, summary in sorted(report["robot_summary"].items()):
        lines.append(f"| {robot} | {summary['files']} | {summary['rows']} |")
    lines.extend(["", "## Files requiring review", ""])
    for item in report["files"]:
        if item["warnings"]:
            lines.append(f"- `{item['relative_path']}`: " + "; ".join(item["warnings"]))
    lines.extend(["", "## File metrics", ""])
    lines.append("| File | Robot | Rows | Declared median | Positional median |")
    lines.append("|---|---|---:|---:|---:|")
    for item in report["files"]:
        declared = item["declared_xyz_error_source_unit"]["median"]
        positional = item["positional_last3_xyz_error_source_unit"]["median"]
        declared_text = "" if declared is None else f"{declared:.4f}"
        positional_text = "" if positional is None else f"{positional:.4f}"
        lines.append(
            f"| `{item['relative_path']}` | {item['robot_id']} | {item['row_count']} | "
            f"{declared_text} | {positional_text} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.source_root.resolve()
    files = [inspect_file(path, root) for path in sorted(root.rglob("*.csv"))]
    by_hash: dict[str, list[str]] = {}
    robot_summary: dict[str, dict[str, int]] = {}
    for item in files:
        by_hash.setdefault(item["sha256"], []).append(item["relative_path"])
        summary = robot_summary.setdefault(item["robot_id"], {"files": 0, "rows": 0})
        summary["files"] += 1
        summary["rows"] += item["row_count"]
    report = {
        "schema_version": 1,
        "source_root": str(root),
        "csv_file_count": len(files),
        "total_rows": sum(item["row_count"] for item in files),
        "robot_summary": robot_summary,
        "duplicate_hash_groups": [paths for paths in by_hash.values() if len(paths) > 1],
        "files": files,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preliminary_inventory.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "preliminary_inventory.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in report if key != "files"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

