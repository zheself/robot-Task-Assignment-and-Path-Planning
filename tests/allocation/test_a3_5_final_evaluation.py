from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.pointer_final import (
    EXPECTED_CELLS,
    PROTOCOL_SHA256,
    _audit_records,
    aggregate_final,
    evaluate_candidates,
    load_final_protocol,
    load_fixed_models,
    validation_items,
    verify_registered_locks,
)
from safe_residual_rl.allocation.pointer_pilot import load_pointer_pilot_config
from safe_residual_rl.allocation.pointer_training import prepare_pointer_pilot

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/allocation/a3_5_sealed_final_v1.json"


@pytest.fixture(scope="module")
def protocol():
    return load_final_protocol(PROTOCOL_PATH)


@pytest.fixture(scope="module")
def prepared():
    context = load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")
    cfg = load_pointer_pilot_config(ROOT / "configs/allocation/a3_5_pointer_pilot_v1.json")
    data = ROOT / "outputs/phase1_allocation/a3_5_pointer_pilot_v1"
    return prepare_pointer_pilot(data / "data", data / "manifest.json", context, cfg)


def test_protocol_hash_and_primary_question_are_frozen(protocol):
    assert PROTOCOL_SHA256 == "8c4d3cb7cc6e61ee589e98786ab78e77958d09694afb102d7f806eda9c208368"
    assert protocol["primary_analysis"]["comparison"] == "hetero_gnn_pair_pointer_minus_matched_hetero_gnn_static"
    assert protocol["primary_analysis"]["independent_unit"] == "task_group_id"


def test_registered_source_checkpoint_and_provenance_locks(protocol):
    assert verify_registered_locks(ROOT, protocol) == ()


def test_final_namespace_and_counts_are_new(protocol):
    benchmark = protocol["benchmark"]
    assert benchmark["id_prefix"] == "a35f1"
    assert benchmark["groups_total"] == 72
    assert benchmark["instances_total"] == 144
    assert set(benchmark["must_be_disjoint_from"]) == {"a2_paper_v2", "a2_paper_v3", "a2_paper_v4", "a3_5_pointer_pilot_v1"}


def test_frozen_and_evaluation_outputs_do_not_exist_before_seal():
    assert not (ROOT / "outputs/phase1_allocation/a3_5_sealed_final_v1_benchmark").exists()
    assert not (ROOT / "outputs/phase1_allocation/a3_5_sealed_final_v1_evaluation").exists()


def test_retraining_repair_beam_and_rl_are_forbidden(protocol):
    assert protocol["training_lock"]["retraining_or_checkpoint_selection"] == "forbidden"
    assert protocol["failure_policy"]["repair"] == "forbidden"
    assert protocol["failure_policy"]["beam_search"] == "forbidden"
    assert protocol["failure_policy"]["rl"] == "forbidden"


def test_six_fixed_models_load_with_registered_state(prepared, protocol):
    models = load_fixed_models(ROOT, protocol, prepared)
    assert [seed for seed, _ in models.pointer] == [101, 211, 307]
    assert [seed for seed, _ in models.static] == [101, 211, 307]


def test_validation_only_matrix_integrates_decoder_scheduler_and_verifier(prepared, protocol):
    context = load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")
    models = load_fixed_models(ROOT, protocol, prepared)
    items = validation_items(prepared, 1)[:1]
    rows, raw = evaluate_candidates(items, models, prepared, context, protocol)
    assert len(rows) == len(raw) == 9
    assert {row["method"] for row in rows} == {
        "pair_pointer_seed_101", "pair_pointer_seed_211", "pair_pointer_seed_307",
        "static_seed_101", "static_seed_211", "static_seed_307",
        "hybrid_assignment_milp", "order_aware_lns", "hybrid_load_balanced",
    }
    assert all(row["split"] == "validation" for row in rows)
    assert sum(row["hard_mask_violations"] + row["atomicity_violations"] for row in rows) == 0


def test_validation_greedy_rollout_is_deterministic(prepared, protocol):
    context = load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")
    models = load_fixed_models(ROOT, protocol, prepared)
    item = validation_items(prepared, 1)[:1]
    left = evaluate_candidates(item, models, prepared, context, protocol)[1]
    right = evaluate_candidates(item, models, prepared, context, protocol)[1]
    for row in left + right:
        row.pop("runtime_s", None)
    assert left == right


def test_manifest_record_audit_enforces_groups_cells_and_namespace(protocol):
    records = []
    for cell in EXPECTED_CELLS:
        for group in range(12):
            gid = f"a35f1-frozen_test-{cell}-group-{group:03d}"
            for variant in range(2):
                records.append({"instance_id": f"{gid}-v{variant:02d}", "task_group_id": gid, "cell_id": cell})
    _audit_records(records, protocol)
    records[0] = dict(records[0], instance_id="a35p1-leak")
    with pytest.raises(RuntimeError, match="namespace"):
        _audit_records(records, protocol)


def test_primary_statistics_are_group_paired_and_deterministic(protocol):
    rows = _synthetic_rows(pointer_better=True)
    left = aggregate_final(rows, protocol, ())
    right = aggregate_final(rows, protocol, ())
    assert left == right
    assert left["primary"]["supported"]
    assert left["result_class"].startswith("A3_5_DECODER_HYPOTHESIS_SUPPORTED")


def test_unreproduced_development_gain_is_valid_negative_result(protocol):
    result = aggregate_final(_synthetic_rows(pointer_better=False), protocol, ())
    assert result["result_class"] == "A3_5_DECODER_HYPOTHESIS_NOT_SUPPORTED"
    assert not result["primary"]["supported"]


def test_integrity_failure_has_precedence_over_performance(protocol):
    result = aggregate_final(_synthetic_rows(pointer_better=True), protocol, ("MANIFEST_HASH_MISMATCH",))
    assert result["result_class"] == "A3_5_FINAL_INVALID"


def test_teacher_claim_is_heterogeneous_verified_incumbents(protocol):
    joined = " ".join(protocol["claim_boundaries"])
    assert "heterogeneous solver-generated verified incumbents" in joined
    assert "not LNS expert solutions" in joined


def _synthetic_rows(pointer_better: bool):
    rows = []
    for cell_index, cell in enumerate(EXPECTED_CELLS):
        for group in range(12):
            gid = f"a35f1-frozen_test-{cell}-group-{group:03d}"
            for variant in range(2):
                common = {"instance_id": f"{gid}-v{variant:02d}", "split": "frozen_test", "cell_id": cell, "task_group_id": gid, "variant_index": variant, "runtime_s": .1, "hard_mask_violations": 0, "atomicity_violations": 0, "weighted_proxy_score": 10.0}
                for seed in (101, 211, 307):
                    rows.append(dict(common, method=f"pair_pointer_seed_{seed}", verified=True))
                    static_ok = True if not pointer_better else (group % 3 == 0)
                    rows.append(dict(common, method=f"static_seed_{seed}", verified=static_ok))
                for method in ("hybrid_assignment_milp", "order_aware_lns", "hybrid_load_balanced"):
                    rows.append(dict(common, method=method, verified=True))
    return rows
