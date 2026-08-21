from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from safe_residual_rl.allocation.benchmark import (
    SplitAccessError,
    audit_split_leakage,
    load_manifest,
    materialize_benchmark,
    select_records,
    split_counts,
    verify_materialized_instances,
    write_manifest,
)
from safe_residual_rl.allocation.generation import (
    canonical_instance_bytes,
    generate_benchmark,
    load_benchmark_config,
    stable_seed,
)
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.schema import EvidenceLabel, validate_instance
from safe_residual_rl.allocation.solvers.lns import solve_deterministic_lns
from safe_residual_rl.allocation.verifier import verify_plan

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config():
    return load_benchmark_config(ROOT / "configs/allocation/benchmark_v1.json")


@pytest.fixture(scope="module")
def generated(config):
    return generate_benchmark(config)


@pytest.fixture(scope="module")
def materialized(config, tmp_path_factory):
    root = tmp_path_factory.mktemp("a2-materialized")
    manifest, generated = materialize_benchmark(config, root)
    path = root / "manifest.json"
    write_manifest(manifest, path)
    return root, path, manifest, generated


def test_config_freezes_required_splits_and_evidence(config) -> None:
    assert config.evidence_label is EvidenceLabel.SIM_GEOMETRIC
    assert set(dict(config.splits)) == {"train", "validation", "frozen_test", "stress"}
    assert config.baseline_protocol["methods"][-1] == "deterministic_lns"


def test_benchmark_count_and_scale_are_predeclared(generated) -> None:
    counts = {}
    for item in generated:
        counts[item.split] = counts.get(item.split, 0) + 1
    assert counts == {"train": 8, "validation": 4, "frozen_test": 6, "stress": 1}
    stress = next(item for item in generated if item.split == "stress")
    assert len(stress.instance.robots) == 8
    assert 64 <= len(stress.instance.segments) <= 80


def test_every_generated_instance_passes_a0_schema(generated) -> None:
    assert all(not validate_instance(item.instance) for item in generated)
    assert {item.instance.evidence_label for item in generated} == {EvidenceLabel.SIM_GEOMETRIC}


def test_generation_is_byte_deterministic(config, generated) -> None:
    second = generate_benchmark(config)
    assert [canonical_instance_bytes(item.instance) for item in generated] == [canonical_instance_bytes(item.instance) for item in second]
    assert [item.seed for item in generated] == [item.seed for item in second]


def test_group_variants_share_geometry_but_change_constraints(generated) -> None:
    left, right = generated[0], generated[1]
    assert left.task_group_id == right.task_group_id
    assert left.instance.robots == right.instance.robots
    assert [item.sampled_curve_m for item in left.instance.segments] == [item.sampled_curve_m for item in right.instance.segments]
    left_constraints = [(x.priority, x.time_window, x.required_tool_id, x.shared_resource_ids) for x in left.instance.segments]
    right_constraints = [(x.priority, x.time_window, x.required_tool_id, x.shared_resource_ids) for x in right.instance.segments]
    assert left_constraints != right_constraints


def test_different_groups_have_controlled_geometry_change(generated) -> None:
    assert generated[0].instance.robots != generated[2].instance.robots
    assert stable_seed(1, "x") == stable_seed(1, "x")
    assert stable_seed(1, "x") != stable_seed(2, "x")


def test_line_bspline_and_closed_loop_semantics_exist(generated) -> None:
    instance = generated[0].instance
    groups = {}
    for segment in instance.segments:
        groups.setdefault(segment.parent_curve_id, []).append(segment)
    closed = sorted(groups[next(key for key in groups if key.endswith("curve-003"))], key=lambda x: x.segment_index)
    assert closed[0].start_pose.position_m == pytest.approx(closed[-1].end_pose.position_m)
    spline = sorted(groups[next(key for key in groups if key.endswith("curve-002"))], key=lambda x: x.segment_index)
    points = [point for segment in spline for point in segment.sampled_curve_m]
    vectors = [(b[0]-a[0], b[1]-a[1], b[2]-a[2]) for a, b in zip(points, points[1:])]
    assert len({tuple(round(value, 6) for value in vector) for vector in vectors}) > 2


def test_manifest_has_no_group_or_parent_leakage(materialized) -> None:
    _, _, manifest, _ = materialized
    assert not audit_split_leakage(manifest.records)
    assert split_counts(manifest.records) == {"frozen_test": 6, "stress": 1, "train": 8, "validation": 4}


def test_leakage_auditor_detects_layout_and_task_group(materialized) -> None:
    _, _, manifest, _ = materialized
    source = next(item for item in manifest.records if item.split == "train")
    target = next(item for item in manifest.records if item.split == "validation")
    bad = replace(target, layout_id=source.layout_id, task_group_id=source.task_group_id)
    codes = {item.code for item in audit_split_leakage(manifest.records + (bad,))}
    assert {"LAYOUT_LEAKAGE", "TASK_GROUP_LEAKAGE"}.issubset(codes)


def test_manifest_roundtrip_hash_and_instance_hashes(materialized) -> None:
    root, path, manifest, _ = materialized
    loaded = load_manifest(path)
    assert loaded == manifest
    assert not verify_materialized_instances(loaded, root)


def test_manifest_tampering_is_rejected(materialized, tmp_path) -> None:
    _, path, _, _ = materialized
    data = json.loads(path.read_text())
    data["records"][0]["seed"] += 1
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_manifest(tampered)


def test_split_access_policy_blocks_frozen_selection(materialized) -> None:
    _, _, manifest, _ = materialized
    assert len(select_records(manifest, "train", "train_fit")) == 8
    with pytest.raises(SplitAccessError):
        select_records(manifest, "frozen_test", "validation_select")


def test_deterministic_lns_returns_reverified_plan(config, generated) -> None:
    context = load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")
    instance = next(item.instance for item in generated if item.instance.instance_id == "train-group-003-v00")
    kwargs = dict(iterations=20, seed=0, objective_weights=dict(config.objective_weights))
    first = solve_deterministic_lns(instance, context, **kwargs)
    second = solve_deterministic_lns(instance, context, **kwargs)
    assert first.status == "feasible"
    assert first.plan.schedule == second.plan.schedule
    assert verify_plan(instance, first.plan, context).feasible
    assert "NOT_A4_CONSTRAINT_REPAIR" in first.diagnostics
