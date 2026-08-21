#!/usr/bin/env python3
"""Build compact descriptive A4a reports from the sealed development rows."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/phase1_allocation/a4_warm_start_pilot_v1"
REPORT = ROOT / "reports/phase1_allocation"


def main():
    rows = list(csv.DictReader((OUTPUT / "validation_rows.csv").open()))
    summary = json.loads((OUTPUT / "summary.json").read_text())
    audit = json.loads((REPORT / "a4_warm_start_pilot_v1_integrity_audit.json").read_text())
    REPORT.mkdir(parents=True, exist_ok=True)
    methods = _method_table(rows)
    cells = _cell_table(rows)
    budgets = _budget_table(rows)
    failures = _failure_table(rows)
    edits = _edit_table(rows)
    _csv(REPORT / "a4_warm_start_pilot_v1_methods.csv", methods)
    _csv(REPORT / "a4_warm_start_pilot_v1_cells.csv", cells)
    _csv(REPORT / "a4_warm_start_pilot_v1_budgets.csv", budgets)
    _csv(REPORT / "a4_warm_start_pilot_v1_failures.csv", failures)
    _csv(REPORT / "a4_warm_start_pilot_v1_edits_runtime.csv", edits)
    (REPORT / "a4_warm_start_pilot_v1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    figures = REPORT / "figures/a4_warm_start_pilot_v1"; figures.mkdir(parents=True, exist_ok=True)
    _plot_budget(budgets, figures)
    _plot_pareto(methods, figures)
    (REPORT / "a4_warm_start_pilot_v1_results.md").write_text(_markdown(summary, audit, methods, cells, failures, edits))


def _select(rows, view="end_to_end_time", budget=1.0):
    return [x for x in rows if x["view"] == view and float(x["budget"]) == budget]


def _family(name):
    if name.startswith("pair_pointer_seed_"): return "pair_pointer"
    if name.startswith("static_seed_"): return "matched_static"
    return name


def _method_table(rows):
    chosen = _select(rows, "fixed_iterations", 50.0)
    result = []
    for family in sorted({_family(x["initializer"]) for x in chosen}):
        items = [x for x in chosen if _family(x["initializer"]) == family]
        verified = [x for x in items if _bool(x["verified"])]
        details = [json.loads(x["detail"]) for x in items]
        result.append({
            "method": family, "rows": len(items), "coverage": _mean([_bool(x["verified"]) for x in items]),
            "conditional_objective": _mean([float(x["final_objective"]) for x in verified if x["final_objective"]]),
            "median_initializer_runtime_s": float(np.median([float(x["initializer_runtime_s"]) for x in items])),
            "median_repair_runtime_s": float(np.median([float(x["repair_runtime_s"]) for x in items])),
            "median_end_to_end_runtime_s": float(np.median([float(x["end_to_end_runtime_s"]) for x in items])),
            "mean_assignment_retention": _mean([float(x["initializer_assignment_retention"]) for x in details if x.get("initializer_assignment_retention") is not None]),
            "mean_assignment_modifications": _mean([float(x["assignment_modifications"]) for x in details if x.get("assignment_modifications") is not None]),
            "mean_modified_atomic_units": _mean([float(x["modified_atomic_units"]) for x in details if x.get("modified_atomic_units") is not None]),
        })
    return result


def _cell_table(rows):
    chosen = _select(rows); result = []
    for cell in sorted({x["cell_id"] for x in chosen}):
        for family in ("pair_pointer", "matched_static", "hybrid_load_balanced", "hybrid_assignment_milp", "cold_start"):
            items = [x for x in chosen if x["cell_id"] == cell and _family(x["initializer"]) == family]
            result.append({"cell_id": cell, "method": family, "coverage": _mean([_bool(x["verified"]) for x in items]), "rows": len(items)})
    return result


def _budget_table(rows):
    result = []
    for view in ("fixed_iterations", "end_to_end_time"):
        for budget in sorted({float(x["budget"]) for x in rows if x["view"] == view}):
            for family in ("pair_pointer", "matched_static", "hybrid_load_balanced", "hybrid_assignment_milp", "cold_start"):
                items = [x for x in rows if x["view"] == view and float(x["budget"]) == budget and _family(x["initializer"]) == family]
                result.append({"view": view, "budget": budget, "method": family, "coverage": _mean([_bool(x["verified"]) for x in items]), "rows": len(items)})
    return result


def _failure_table(rows):
    chosen = _select(rows); counts = Counter((_family(x["initializer"]), x["failure_reason"] or "verified") for x in chosen)
    return [{"method": method, "failure_reason": reason, "count": count} for (method, reason), count in sorted(counts.items())]


def _edit_table(rows):
    chosen = _select(rows); result = []
    for family in sorted({_family(x["initializer"]) for x in chosen}):
        items = [json.loads(x["detail"]) for x in chosen if _family(x["initializer"]) == family]
        result.append({"method": family, "mean_assignment_modifications": _mean([x.get("assignment_modifications") for x in items]), "mean_order_modifications": _mean([x.get("order_modifications") for x in items]), "mean_modified_atomic_units": _mean([x.get("modified_atomic_units") for x in items]), "mean_assignment_retention": _mean([x.get("initializer_assignment_retention") for x in items])})
    return result


def _plot_budget(rows, root):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, view in zip(axes, ("fixed_iterations", "end_to_end_time")):
        for method in ("pair_pointer", "matched_static", "hybrid_load_balanced", "cold_start"):
            items = [x for x in rows if x["view"] == view and x["method"] == method]
            ax.plot([x["budget"] for x in items], [x["coverage"] for x in items], marker="o", label=method)
        ax.set_xlabel("iterations" if view == "fixed_iterations" else "end-to-end budget (s)"); ax.set_ylabel("verified coverage"); ax.set_ylim(0, 1.03); ax.grid(alpha=.25)
    axes[1].legend(fontsize=7); fig.tight_layout()
    for suffix in ("png", "pdf"): fig.savefig(root / f"coverage_by_budget.{suffix}", dpi=180)
    plt.close(fig)


def _plot_pareto(rows, root):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for x in rows:
        ax.scatter(x["median_end_to_end_runtime_s"], x["coverage"]); ax.annotate(x["method"], (x["median_end_to_end_runtime_s"], x["coverage"]), fontsize=7)
    ax.set_xlabel("median end-to-end runtime (s)"); ax.set_ylabel("verified coverage"); ax.set_ylim(0, 1.03); ax.grid(alpha=.25); fig.tight_layout()
    for suffix in ("png", "pdf"): fig.savefig(root / f"coverage_runtime_pareto.{suffix}", dpi=180)
    plt.close(fig)


def _markdown(summary, audit, methods, cells, failures, edits):
    lines = ["# A4a Pair-Pointer warm-start development-pilot result", "", f"Decision: **`{summary['decision']}`**", "", f"Integrity override: **`{audit['classification']}`**", "", "Evidence: **SIM_GEOMETRIC development-only**", "", "The preregistered 1.0 s primary view is invalid because of two implementation defects documented in the integrity audit. The following table is the unaffected, descriptive 50-iteration view; it is not a replacement confirmatory endpoint.", "", "## Descriptive fixed-50-iteration view", "", "| Initializer + identical repair | Coverage | Init median (s) | Repair median (s) | E2E median (s) | Assignment retention |", "|---|---:|---:|---:|---:|---:|"]
    for x in methods: lines.append(f"| {x['method']} | {x['coverage']:.3f} | {x['median_initializer_runtime_s']:.4f} | {x['median_repair_runtime_s']:.4f} | {x['median_end_to_end_runtime_s']:.4f} | {_fmt(x['mean_assignment_retention'])} |")
    lines += ["", "## Gate", ""] + [f"- `{key}`: {value}" for key, value in summary["gates"].items()]
    lines += ["", "## Mechanical integrity", ""] + [f"- `{key}`: {value}" for key, value in summary["integrity"].items()]
    lines += ["", "## Semantic integrity override", ""] + [f"- {item}" for item in audit["failures"]]
    lines += ["", "## Boundary", "", "This is a development-only geometric proxy experiment. It does not alter A3/A3.5, establish real-robot or collision safety, use RL/path planning/physical modelling, or authorise a new frozen benchmark.", ""]
    return "\n".join(lines)


def _csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _mean(values):
    values = [float(x) for x in values if x is not None]
    return None if not values else float(np.mean(values))


def _bool(value): return str(value).lower() == "true"
def _fmt(value): return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__": main()
