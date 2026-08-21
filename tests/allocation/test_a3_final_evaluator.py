from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_residual_rl.allocation.final_evaluation import (
    classify_final_result,
    load_final_items,
    strong_pairwise_statistics,
    verify_protocol_locks,
)
from safe_residual_rl.allocation.paper_benchmark import PaperManifest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads(
    (ROOT / "configs/allocation/a3_final_evaluation_v1.json").read_text(
        encoding="utf-8"
    )
)


def _rows(*, learned_failed_cell: str | None = None):
    rows = []
    cells = PROTOCOL["data_access"]["primary_cells"]
    methods = [
        *PROTOCOL["baselines"]["context_methods"],
        *PROTOCOL["baselines"]["strong_methods"],
        "edge_mlp_seed_17",
        "edge_mlp_seed_29",
        "edge_mlp_seed_43",
    ]
    for cell in cells:
        for group_index in range(12):
            instance_id = f"fixture-{cell}-{group_index:02d}"
            for method in methods:
                learned = method.startswith("edge_mlp_seed_")
                verified = not (learned and cell == learned_failed_cell)
                # Context methods deliberately miss one group; strong methods
                # and the learned method are otherwise perfect.
                if method in PROTOCOL["baselines"]["context_methods"] and group_index == 0:
                    verified = False
                rows.append(
                    {
                        "instance_id": instance_id,
                        "split": "frozen_test",
                        "cell_id": cell,
                        "task_group_id": f"group-{cell}-{group_index:02d}",
                        "variant_index": 0,
                        "method": method,
                        "verified": verified,
                        "status": "feasible" if verified else "schedule_infeasible",
                        "runtime_s": 0.01,
                        "weighted_proxy_score": 1.0 if verified else None,
                    }
                )
    return rows


def _integrity(value: bool = True):
    return {
        "all_source_and_checkpoint_hashes_match": value,
        "complete_matrix": value,
        "negative_controls": value,
    }


def test_protocol_source_locks_match_before_final_evaluator_changes() -> None:
    assert not verify_protocol_locks(ROOT, PROTOCOL)


def test_sealed_loader_rejects_development_split_before_file_access(tmp_path) -> None:
    manifest = PaperManifest("a2-paper-split-manifest-v4", "v", "c", (), "m")
    with pytest.raises(PermissionError):
        load_final_items(tmp_path, manifest, ("validation",))


def test_fixed_decision_hierarchy_competitive_and_floor_failure() -> None:
    rows = _rows()
    pairwise = strong_pairwise_statistics(rows, PROTOCOL)
    decision = classify_final_result(rows, PROTOCOL, pairwise, _integrity())
    assert decision["result_class"] == "A3_FINAL_COMPETITIVE_NOT_SUPERIOR"
    assert decision["a3_final_passed"] is True

    failed_rows = _rows(learned_failed_cell="ood_scale")
    failed = classify_final_result(
        failed_rows,
        PROTOCOL,
        strong_pairwise_statistics(failed_rows, PROTOCOL),
        _integrity(),
    )
    assert failed["result_class"] == "A3_FINAL_FAILED_BASELINE_FLOOR"
    assert failed["checks"]["absolute_ood_scale"] is False


def test_integrity_failure_preempts_performance() -> None:
    rows = _rows()
    decision = classify_final_result(
        rows, PROTOCOL, strong_pairwise_statistics(rows, PROTOCOL), _integrity(False)
    )
    assert decision == {
        "result_class": "A3_FINAL_INVALID",
        "a3_final_passed": False,
        "checks": _integrity(False),
    }
