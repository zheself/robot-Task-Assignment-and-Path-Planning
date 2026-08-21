#!/usr/bin/env python3
"""Build descriptive paper evidence from immutable A3.5 outputs only.

This script performs no training, checkpoint selection, confirmatory retesting,
repair or frozen evaluation. It verifies the registered sealed summary and then
renders tables/figures from already persisted rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/a35-paper-mpl")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

EXPECTED_RESULT = "A3_5_DECODER_HYPOTHESIS_SUPPORTED"
EXPECTED_MANIFEST = "13b5d093d6edd863128f5dd683f1ff98ce21185c9deac1165e591bc0ae4dd400"
METHOD_ORDER = ["matched_static", "pair_pointer", "hybrid_load_balanced", "hybrid_assignment_milp", "order_aware_lns"]
LABELS = {"matched_static":"Matched static", "pair_pointer":"Pair-Pointer", "hybrid_load_balanced":"Hybrid load-balanced", "hybrid_assignment_milp":"Hybrid MILP", "order_aware_lns":"Order-aware LNS"}
COLORS = {"matched_static":"#7f7f7f", "pair_pointer":"#0072B2", "hybrid_load_balanced":"#009E73", "hybrid_assignment_milp":"#E69F00", "order_aware_lns":"#D55E00"}


def main():
    root = Path(__file__).resolve().parents[1]
    pilot = _read(root / "reports/phase1_allocation/a3_5_pointer_pilot_v1_summary.json")
    final = _read(root / "outputs/phase1_allocation/a3_5_sealed_final_v1_evaluation/summary.json")
    rows = list(csv.DictReader((root / "outputs/phase1_allocation/a3_5_sealed_final_v1_evaluation/rows.csv").open()))
    _guard(final, rows)
    report = root / "reports/phase1_allocation"
    figures = report / "figures/a3_5_sealed_final_v1"
    figures.mkdir(parents=True, exist_ok=True)

    main_rows = _main_rows(final)
    seed_rows = _seed_rows(rows)
    cell_rows = _cell_rows(rows, final)
    failure_rows = _failure_rows(rows)
    development_rows = _development_rows(pilot, final)
    claim_rows = _claim_rows()
    _write_csv(report / "a3_5_paper_main_results.csv", main_rows)
    _write_csv(report / "a3_5_paper_seed_stability.csv", seed_rows)
    _write_csv(report / "a3_5_paper_difficulty_cells.csv", cell_rows)
    _write_csv(report / "a3_5_paper_failure_counts.csv", failure_rows)
    _write_csv(report / "a3_5_paper_development_frozen.csv", development_rows)
    _write_csv(report / "a3_5_paper_claim_evidence_boundary.csv", claim_rows)

    _plot_cells(figures, cell_rows)
    _plot_seeds(figures, seed_rows)
    _plot_pareto(figures, main_rows)
    _plot_failures(figures, failure_rows)
    _plot_development(figures, development_rows)
    _plot_architecture(figures)

    sources = {
        "evidence_script": _sha(root / "scripts/build_a3_5_paper_evidence.py"),
        "pilot_summary": _sha(root / "reports/phase1_allocation/a3_5_pointer_pilot_v1_summary.json"),
        "sealed_summary": _sha(root / "outputs/phase1_allocation/a3_5_sealed_final_v1_evaluation/summary.json"),
        "sealed_rows": _sha(root / "outputs/phase1_allocation/a3_5_sealed_final_v1_evaluation/rows.csv"),
    }
    manifest_path = report / "a3_5_paper_evidence_manifest.json"
    generated = {p.relative_to(root).as_posix(): _sha(p) for p in sorted(report.glob("a3_5_paper_*")) if p.is_file() and p != manifest_path}
    generated.update({p.relative_to(root).as_posix(): _sha(p) for p in sorted(figures.iterdir()) if p.is_file()})
    manifest = {"version":"a3-5-paper-evidence-v1", "descriptive_only":True, "no_model_or_evaluation_run":True, "result_class":final["result_class"], "source_hashes":sources, "generated_hashes":generated}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"tables":6,"figures":12,"result_class":final["result_class"],"descriptive_only":True}), flush=True)


def _guard(final, rows):
    if final["result_class"] != EXPECTED_RESULT or final["manifest_file_sha256"] != EXPECTED_MANIFEST:
        raise RuntimeError("sealed A3.5 evidence differs from the immutable closure")
    if len(rows) != 1296 or {r["split"] for r in rows} != {"frozen_test"}:
        raise RuntimeError("sealed row matrix changed")
    if not all(final["checks"].values()):
        raise RuntimeError("cannot render invalid sealed result")


def _main_rows(final):
    values = {"matched_static":final["methods"]["static"], "pair_pointer":final["methods"]["pair_pointer"], **final["strong_baselines"]}
    result=[]
    for method in METHOD_ORDER:
        item=values[method]
        dominated = method == "pair_pointer"  # registered overall dominance by hybrid_load_balanced
        result.append({"method":method,"label":LABELS[method],"coverage":item["coverage"],"median_runtime_s":item["median_runtime_s"],"runtime_q1_s":item["runtime_iqr_s"][0],"runtime_q3_s":item["runtime_iqr_s"][1],"conditional_weighted_proxy_score":item["conditional_weighted_proxy_score"],"pair_pointer_dominated_by_hybrid_load_balanced":dominated})
    return result


def _seed_rows(rows):
    result=[]
    for seed in (101,211,307):
        for family,prefix in (("pair_pointer","pair_pointer_seed_"),("matched_static","static_seed_")):
            selected=[r for r in rows if r["method"]==f"{prefix}{seed}"]
            result.append({"seed":seed,"method":family,"coverage":np.mean([r["verified"]=="True" for r in selected]),"verified":sum(r["verified"]=="True" for r in selected),"instances":len(selected)})
    return result


def _cell_rows(rows, final):
    result=[]
    for cell,diff in sorted(final["primary"]["cell_differences"].items()):
        item={"cell":cell,"paired_difference":diff}
        for family,prefix in (("pair_pointer","pair_pointer_seed_"),("matched_static","static_seed_")):
            selected=[r for r in rows if r["cell_id"]==cell and r["method"].startswith(prefix)]
            item[f"{family}_coverage"]=np.mean([r["verified"]=="True" for r in selected])
        result.append(item)
    return result


def _failure_rows(rows):
    counter=Counter()
    totals=Counter()
    for r in rows:
        method=_family(r["method"]); totals[method]+=1
        if r["verified"]!="True": counter[(method,r["failure_class"] or "unclassified")]+=1
    return [{"method":m,"failure_class":c,"count":n,"total_rows":totals[m],"failure_rate":n/totals[m]} for (m,c),n in sorted(counter.items())]


def _development_rows(pilot, final):
    decision=pilot["decision"]
    return [
        {"stage":"development","pair_pointer_coverage":decision["pointer_mean_coverage"],"matched_static_coverage":decision["static_mean_coverage"],"paired_improvement":decision["mean_coverage_difference"],"independent_groups":24},
        {"stage":"untouched_final","pair_pointer_coverage":final["methods"]["pair_pointer"]["coverage"],"matched_static_coverage":final["methods"]["static"]["coverage"],"paired_improvement":final["primary"]["mean_paired_coverage_difference"],"independent_groups":72},
    ]


def _claim_rows():
    return [
        {"claim":"Dynamic autoregressive Pair-Pointer improves matched static decoding","evidence":"65.05% vs 40.28%; paired +24.77 points; CI [18.52,31.48]; p≈1e-5","status":"SUPPORTED","boundary":"Same hetero-GNN, training data, supervision and hard-mask framework; SIM_GEOMETRIC only"},
        {"claim":"Improvement is robust to seed and registered difficulty","evidence":"All three seed differences positive; all six cell differences non-negative","status":"SUPPORTED","boundary":"Six registered synthetic geometric cells, not arbitrary industrial distributions"},
        {"claim":"Pair-Pointer is better than strong optimisation/heuristic methods","evidence":"65.05% below 68.75% hybrid load-balanced, 72.22% MILP and 79.86% LNS","status":"NOT_SUPPORTED","boundary":"Hybrid load-balanced dominates Pair-Pointer in overall coverage-runtime"},
        {"claim":"Hard mask alone caused the improvement","evidence":"Both matched learned decoders use hard feasibility masking","status":"NOT_IDENTIFIED","boundary":"The comparison isolates the autoregressive pair decoder plus dynamic state, not a mask-only ablation"},
        {"claim":"Teacher labels are expert/global-optimal solutions","evidence":"MILP/LNS/constructive heterogeneous verified incumbents","status":"NOT_SUPPORTED","boundary":"Do not call them LNS expert demonstrations"},
        {"claim":"Real deployment, collision safety or physical quality improvement","evidence":"No real CAD/log execution, collision certification or physical model in this experiment","status":"NOT_SUPPORTED","boundary":"All evidence is SIM_GEOMETRIC"},
    ]


def _plot_cells(out, rows):
    labels=[r["cell"].replace("_","\n") for r in rows]; x=np.arange(len(rows)); width=.36
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.bar(x-width/2,[r["matched_static_coverage"] for r in rows],width,label="Matched static",color=COLORS["matched_static"]); ax.bar(x+width/2,[r["pair_pointer_coverage"] for r in rows],width,label="Pair-Pointer",color=COLORS["pair_pointer"])
    ax.set(ylabel="Verified candidate coverage",ylim=(0,1.08),xticks=x,xticklabels=labels,title="Untouched final coverage by difficulty cell"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.25); _save(fig,out/"difficulty_cell_coverage")


def _plot_seeds(out, rows):
    fig,ax=plt.subplots(figsize=(7,4.5))
    for method in ("matched_static","pair_pointer"):
        selected=[r for r in rows if r["method"]==method]; ax.plot([r["seed"] for r in selected],[r["coverage"] for r in selected],marker="o",linewidth=2,label=LABELS[method],color=COLORS[method])
    ax.set(ylabel="Verified candidate coverage",xlabel="Fixed training seed",ylim=(0,1),title="Seed stability on untouched final groups"); ax.set_xticks([101,211,307]); ax.legend(frameon=False); ax.grid(alpha=.25); _save(fig,out/"seed_stability")


def _plot_pareto(out, rows):
    fig,ax=plt.subplots(figsize=(8,5.2))
    for r in rows:
        ax.scatter(r["median_runtime_s"],r["coverage"],s=90,color=COLORS[r["method"]],zorder=3); ax.annotate(r["label"],(r["median_runtime_s"],r["coverage"]),xytext=(6,5),textcoords="offset points",fontsize=9)
    ax.set_xscale("log"); ax.set(xlabel="Median runtime per instance (s, log scale)",ylabel="Verified candidate coverage",ylim=(.35,.85),title="Coverage–runtime trade-off (descriptive)"); ax.grid(alpha=.25); ax.annotate("Hybrid load-balanced dominates\nPair-Pointer overall",xy=(.446,.6505),xytext=(.055,.58),arrowprops={"arrowstyle":"->","color":"#333333"},fontsize=9); _save(fig,out/"coverage_runtime_pareto")


def _plot_failures(out, rows):
    methods=METHOD_ORDER; counts={r["method"]:r["count"] for r in rows}; totals={r["method"]:r["total_rows"] for r in rows}
    fig,ax=plt.subplots(figsize=(8.5,4.6)); bars=ax.bar(range(len(methods)),[counts.get(m,0) for m in methods],color=[COLORS[m] for m in methods]); ax.bar_label(bars); ax.set(ylabel="Failed method–instance rows",xticks=range(len(methods)),xticklabels=[LABELS[m].replace(" ","\n") for m in methods],title="Retained schedule-infeasible failures"); ax.grid(axis="y",alpha=.25); _save(fig,out/"failure_counts")


def _plot_development(out, rows):
    x=np.arange(2); width=.34; fig,ax=plt.subplots(figsize=(7,4.6)); ax.bar(x-width/2,[r["matched_static_coverage"] for r in rows],width,label="Matched static",color=COLORS["matched_static"]); ax.bar(x+width/2,[r["pair_pointer_coverage"] for r in rows],width,label="Pair-Pointer",color=COLORS["pair_pointer"])
    for i,r in enumerate(rows): ax.text(i,r["pair_pointer_coverage"]+.025,f"Δ {100*r['paired_improvement']:.2f} pt",ha="center",fontsize=10)
    ax.set(ylabel="Verified candidate coverage",ylim=(0,1),xticks=x,xticklabels=["Development\n24 groups","Untouched final\n72 groups"],title="Development and untouched-final effect direction"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.25); _save(fig,out/"development_vs_frozen")


def _plot_architecture(out):
    fig,ax=plt.subplots(figsize=(12,6.6)); ax.set_xlim(0,12); ax.set_ylim(0,7); ax.axis("off")
    boxes=[(0.2,5.0,2.0,1.0,"Continuous segments\nrobots · resources\nconstraints","#DDEBF7"),(2.7,5.0,1.9,1.0,"Heterogeneous GNN\nshared encoder","#BDD7EE"),(5.1,5.0,2.1,1.0,"Unit · robot · edge\nembeddings","#9DC3E6"),(8.1,5.0,2.5,1.0,"Pair scores\n(unit, robot)","#5B9BD5"),(8.1,3.2,2.5,1.1,"Hard pair mask before softmax\nreachability · assigned\npredecessor readiness","#F4B183"),(4.6,2.8,2.7,1.7,"Dynamic decoder state\nload & completion time\nlast unit / position / action\npredecessor satisfaction\nresource usage","#C6E0B4"),(8.1,1.0,2.5,1.0,"Deterministic greedy\npair selection","#A9D18E"),(4.6,0.7,2.7,1.0,"State update loop\n(no repair / no beam)","#A9D18E"),(0.5,0.7,3.0,1.0,"After all units: unchanged A1 scheduler\n+ independent verifier","#FFD966")]
    for x,y,w,h,text,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.08",facecolor=color,edgecolor="#333")); ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
    arrows=[((2.2,5.5),(2.7,5.5)),((4.6,5.5),(5.1,5.5)),((7.2,5.5),(8.1,5.5)),((9.35,5.0),(9.35,4.3)),((7.3,3.65),(8.1,3.65)),((6.15,5.0),(6.15,4.5)),((9.35,3.2),(9.35,2.0)),((8.1,1.5),(7.3,1.3)),((4.6,1.2),(3.5,1.2))]
    for a,b in arrows: ax.add_patch(FancyArrowPatch(a,b,arrowstyle="->",mutation_scale=14,color="#333",linewidth=1.3,connectionstyle="arc3,rad=0.0"))
    ax.add_patch(FancyArrowPatch((5.7,1.7),(5.35,2.8),arrowstyle="->",mutation_scale=14,color="#267326",linewidth=1.6,connectionstyle="arc3,rad=-.35")); ax.text(0.3,6.55,"Feasibility-aware autoregressive decoding for learned continuous-process allocation",fontsize=15,weight="bold"); ax.text(0.8,.25,"verified coverage or retained failure",fontsize=9,color="#555"); _save(fig,out/"method_architecture_dynamic_state")


def _save(fig, base):
    fig.tight_layout(); fig.savefig(base.with_suffix(".png"),dpi=220,bbox_inches="tight"); fig.savefig(base.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)
def _family(method):
    if method.startswith("pair_pointer_seed_"): return "pair_pointer"
    if method.startswith("static_seed_"): return "matched_static"
    return method
def _write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
def _read(path): return json.loads(path.read_text(encoding="utf-8"))
def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=="__main__": main()
