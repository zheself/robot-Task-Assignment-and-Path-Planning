from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.repair import InitializerState
from safe_residual_rl.allocation.schema import HandoffPolicy, TimeWindow, allocation_instance_from_dict
from safe_residual_rl.allocation.scheduling import build_schedule
from safe_residual_rl.allocation.search.anytime import InitializerOutcome, InitializerProvenance
from safe_residual_rl.allocation.search.data_protocol_r import (
    audit_protocol_r_records,
    build_regular_variant,
    generate_protocol_r_data,
    guard_protocol_r_path,
    load_protocol_r_config,
    make_initializer_failure_challenge,
    require_execution_ready,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/allocation/a4b_protocol_r_freeze_candidate_v1.json"


@pytest.fixture(scope="module")
def context():
    return load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")


def test_draft_config_is_complete_but_execution_fail_closed(context, tmp_path):
    config = load_protocol_r_config(CONFIG, allow_draft=True)
    assert config["data"]["train_groups_total"] == 56
    assert config["data"]["development_groups_total"] == 24
    assert set(config["data"]["generator"]["cell_specs"]) == set(
        config["data"]["cells"]
    )
    with pytest.raises(PermissionError, match="not frozen"):
        require_execution_ready(config)
    with pytest.raises(PermissionError, match="not frozen"):
        generate_protocol_r_data(config, tmp_path / "a4blnsr3", context)


def test_fresh_ids_and_group_sibling_geometry_are_deterministic():
    config = load_protocol_r_config(CONFIG, allow_draft=True)
    left = build_regular_variant(config, "iid_small", "train", 0, 0)
    replay = build_regular_variant(config, "iid_small", "train", 0, 0)
    sibling = build_regular_variant(config, "iid_small", "train", 0, 1)
    assert left == replay
    assert left.instance_id.startswith("a4blnsr3-train-iid_small-regular-group-000")
    assert left.workpiece_id.endswith("-workpiece")
    assert left.layout_id.endswith("-layout")
    assert [item.sampled_curve_m for item in left.segments] == [
        item.sampled_curve_m for item in sibling.segments
    ]


@pytest.mark.parametrize(
    "token",
    [
        "validation",
        "frozen_test",
        "stress",
        "a35f1",
        "a4_warm_start_pilot_v1",
        "a4b_ordinary_lns_dev_v2",
    ],
)
def test_forbidden_old_or_frozen_path_is_rejected(tmp_path, token):
    with pytest.raises(PermissionError):
        guard_protocol_r_path(tmp_path / token)


def _record(split, group, instance, workpiece, layout, parent):
    return {
        "split": split,
        "cell_id": "iid_small",
        "task_group_id": group,
        "instance_id": instance,
        "workpiece_id": workpiece,
        "layout_id": layout,
        "parent_curve_ids": [parent],
        "evidence_label": "SIM_GEOMETRIC",
    }


def test_manifest_audit_rejects_split_leakage():
    config = load_protocol_r_config(CONFIG, allow_draft=True)
    # Reduce only expected counts in a private test copy; scientific config is untouched.
    config = json.loads(json.dumps(config))
    config["data"].update(
        train_instances_total=1,
        development_instances_total=1,
        train_groups_total=1,
        development_groups_total=1,
    )
    records = [
        _record(
            "train",
            "a4blnsr3-train-g",
            "a4blnsr3-train-i",
            "a4blnsr3-shared-workpiece",
            "a4blnsr3-train-layout",
            "a4blnsr3-train-curve",
        ),
        _record(
            "development",
            "a4blnsr3-development-g",
            "a4blnsr3-development-i",
            "a4blnsr3-shared-workpiece",
            "a4blnsr3-development-layout",
            "a4blnsr3-development-curve",
        ),
    ]
    with pytest.raises(RuntimeError, match="workpiece_id"):
        audit_protocol_r_records(records, config)


def test_generator_source_hash_drift_is_rejected(tmp_path):
    raw = json.loads(CONFIG.read_text())
    raw["data"]["generator"]["parameter_source_sha256"] = "0" * 64
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(raw))
    # Preserve the repo-relative depth expected by the loader.
    with pytest.raises((RuntimeError, FileNotFoundError)):
        load_protocol_r_config(candidate, allow_draft=True)


def test_challenge_tightens_only_end_windows_and_preserves_witness(
    context, monkeypatch
):
    import safe_residual_rl.allocation.search.data_protocol_r as module

    base = allocation_instance_from_dict(
        load_auditable_fixture(
            ROOT / "data/fixtures/allocation/01_valid_minimal.json"
        )["instance"]
    )
    template = base.segments[0]
    segments = []
    for index, offset in enumerate((0.0, 0.35)):
        start = (offset, 0.0, 0.0)
        end = (offset + 0.1, 0.0, 0.0)
        segments.append(
            replace(
                template,
                id=f"segment-{index}",
                parent_curve_id=f"curve-{index}",
                segment_index=0,
                sampled_curve_m=(start, end),
                start_pose=replace(template.start_pose, position_m=start),
                end_pose=replace(template.end_pose, position_m=end),
                predecessor_ids=(),
                handoff_policy=HandoffPolicy.FREE,
                time_window=TimeWindow(0.0, 100.0),
            )
        )
    robot0 = base.robots[0]
    robot1 = replace(
        robot0,
        id="robot-1",
        base_pose=replace(robot0.base_pose, position_m=(0.55, 0.0, 0.0)),
    )
    instance = replace(
        base,
        instance_id="challenge-fixture",
        segments=tuple(segments),
        robots=(robot0, robot1),
    )
    witness_built = build_schedule(
        instance,
        {"robot-0": ("segment-0",), "robot-1": ("segment-1",)},
        context,
        "protocol-r-challenge-witness",
    )
    assert witness_built.plan is not None
    bad_state = InitializerState(
        ("robot-0", "robot-0"),
        (("robot-0", (0, 1)), ("robot-1", ())),
    )

    def provenance(feasible, reason):
        return InitializerProvenance(
            "hybrid_load_balanced",
            "hybrid_load_balanced",
            "fixture",
            True,
            False,
            None,
            "fixture-hash",
            feasible,
            reason,
            0,
            1,
            1e-9,
        )

    outcomes = iter(
        (
            InitializerOutcome(bad_state, provenance(True, None)),
            InitializerOutcome(bad_state, provenance(False, "time_window_failure")),
        )
    )
    monkeypatch.setattr(
        module, "build_hybrid_load_balanced_initializer", lambda *args, **kwargs: next(outcomes)
    )
    weights = {
        "makespan": 1.0,
        "load_variance": 0.05,
        "travel_setup_time": 0.1,
        "priority_tardiness": 1.0,
    }
    result = make_initializer_failure_challenge(
        instance, witness_built.plan, context, weights
    )
    assert result.eligible and result.failure_reason == "time_window_failure"
    assert result.tightened_unit_indices
    before = {item.id: item for item in instance.segments}
    after = {item.id: item for item in result.instance.segments}
    assert all(after[key].time_window.start_s == before[key].time_window.start_s for key in before)
    assert any(after[key].time_window.end_s < before[key].time_window.end_s for key in before)
