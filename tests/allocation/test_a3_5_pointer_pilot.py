from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.graphs import build_a3_graph, fit_feature_vocabulary
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.pointer_decoder import (
    FeasiblePairPointer,
    _initial_state,
    _prepare,
    _update_state,
)
from safe_residual_rl.allocation.pointer_pilot import (
    audit_manifest_overlap,
    canonical_teacher_actions,
    load_pointer_manifest,
    load_pointer_pilot_config,
    replay_pointer_actions,
    validate_teacher_prefixes,
)
from safe_residual_rl.allocation.pointer_training import prepare_pointer_pilot
from safe_residual_rl.allocation.schema import allocation_instance_from_dict
from safe_residual_rl.allocation.solvers import solve_hybrid_load_balanced
from safe_residual_rl.allocation.solvers.common import allocation_units
from safe_residual_rl.allocation.verifier import verify_plan

ROOT = Path(__file__).resolve().parents[2]
TARGET_FIXTURES = Path("/public/home/v-chengwy/cjz/RL_credit-assign/Data-Calibrated-Safe-Residual-RL/data/fixtures/allocation")
PILOT_ROOT = ROOT / "outputs/phase1_allocation/a3_5_pointer_pilot_v1"


@pytest.fixture(scope="module")
def context():
    return load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")


def _fixture(name: str):
    return allocation_instance_from_dict(load_auditable_fixture(TARGET_FIXTURES / name)["instance"])


def _graph(instance, context):
    vocabulary = fit_feature_vocabulary([instance], split="train")
    return build_a3_graph(instance, context, vocabulary, split="train")


def test_preregistered_protocol_is_train_validation_only() -> None:
    config = load_pointer_pilot_config(ROOT / "configs/allocation/a3_5_pointer_pilot_v1.json")
    assert config.sha256 == "876933101dc5ed2e56984d8666dae36509481e93d6d6422ec4eed5380a7bea17"
    assert sum(item.train_groups * item.variants for item in config.cells) == 96
    assert sum(item.validation_groups * item.variants for item in config.cells) == 48
    assert config.raw["training"]["seeds"] == [101, 211, 307]
    assert config.raw["selection"]["repair"] == "forbidden"


def test_teacher_serialization_prefix_and_replay_are_deterministic(context) -> None:
    instance = _fixture("03_valid_explicit_boundary.json")
    result = solve_hybrid_load_balanced(instance, context)
    assert result.plan is not None
    first = canonical_teacher_actions(instance, result.plan)
    second = canonical_teacher_actions(instance, result.plan)
    assert first == second
    validate_teacher_prefixes(instance, first)
    replay = replay_pointer_actions(instance, first, context)
    assert replay is not None and verify_plan(instance, replay, context).feasible
    expected = sorted((item.segment_id, item.robot_id, item.order_index) for item in result.plan.schedule)
    actual = sorted((item.segment_id, item.robot_id, item.order_index) for item in replay.schedule)
    assert actual == expected


def test_pair_mask_blocks_predecessor_until_satisfied(context) -> None:
    instance = _fixture("03_valid_explicit_boundary.json")
    graph = _graph(instance, context)
    model = FeasiblePairPointer(graph, encoder_family="hetero_gnn", hidden_dim=16, layers=1, heads=4, dropout=0.0)
    state = _initial_state(instance, graph, torch.device("cpu"))
    mask = model.feasible_pair_mask(graph, instance, state)
    assert mask[0].any()
    assert not mask[1].any()
    state.assigned[0] = True
    state.step = 1
    assert model.feasible_pair_mask(graph, instance, state)[1].any()


def test_atomic_unit_cannot_split_or_repeat(context) -> None:
    instance = _fixture("02_valid_same_robot_segments.json")
    graph = _graph(instance, context)
    assert len(allocation_units(instance)) == 1
    model = FeasiblePairPointer(graph, encoder_family="hetero_gnn", hidden_dim=16, layers=1, heads=4, dropout=0.0)
    rollout = model.greedy_rollout(graph, instance, context)
    assert len(rollout.actions) == 1
    assert set(rollout.actions[0].unit) == {"seg-0", "seg-1"}
    assert rollout.atomicity_violations == 0


def test_unreachable_pair_is_never_available(context) -> None:
    instance = _fixture("01_valid_minimal.json")
    bad = replace(instance.robots[0], id="robot-z-bad", capabilities=())
    expanded = replace(instance, robots=(instance.robots[0], bad))
    graph = _graph(expanded, context)
    model = FeasiblePairPointer(graph, encoder_family="hetero_gnn", hidden_dim=16, layers=1, heads=4, dropout=0.0)
    state = _initial_state(expanded, graph, torch.device("cpu"))
    mask = model.feasible_pair_mask(graph, expanded, state)
    assert mask[0, graph.robot_ids.index("robot-0")]
    assert not mask[0, graph.robot_ids.index("robot-z-bad")]
    rollout = model.greedy_rollout(graph, expanded, context)
    assert all(item.robot_id != "robot-z-bad" for item in rollout.actions)


def test_state_updates_load_position_resource_and_prevents_reselection(context) -> None:
    instance = _fixture("04_valid_shared_zone.json")
    graph = _graph(instance, context)
    model = FeasiblePairPointer(graph, encoder_family="hetero_gnn", hidden_dim=16, layers=1, heads=4, dropout=0.0)
    _, segment, _ = model.encode(graph)
    prepared = _prepare(instance, graph, segment)
    state = _initial_state(instance, graph, torch.device("cpu"))
    _update_state(state, instance, graph, prepared, 0, 0)
    assert state.step == 1 and state.assigned[0]
    assert state.robot_load[0] > 0 and state.resource_usage[0] > 0
    assert not model.feasible_pair_mask(graph, instance, state).any()
    with pytest.raises(ValueError, match="twice"):
        _update_state(state, instance, graph, prepared, 0, 0)


def test_greedy_rollout_is_deterministic_and_input_tuple_equivalent(context) -> None:
    instance = _fixture("03_valid_explicit_boundary.json")
    permuted = replace(instance, segments=tuple(reversed(instance.segments)), robots=tuple(reversed(instance.robots)), resources=tuple(reversed(instance.resources)))
    left_graph = _graph(instance, context)
    right_graph = _graph(permuted, context)
    assert left_graph.canonical_sha256() == right_graph.canonical_sha256()
    torch.manual_seed(77)
    left = FeasiblePairPointer(left_graph, encoder_family="graph_transformer", hidden_dim=16, layers=1, heads=4, dropout=0.0)
    right = FeasiblePairPointer(right_graph, encoder_family="graph_transformer", hidden_dim=16, layers=1, heads=4, dropout=0.0)
    right.load_state_dict(left.state_dict())
    left_actions = left.greedy_rollout(left_graph, instance, context).actions
    assert left_actions == left.greedy_rollout(left_graph, instance, context).actions
    assert left_actions == right.greedy_rollout(right_graph, permuted, context).actions


def test_loader_rejects_forbidden_split_and_v4_paths(tmp_path, context) -> None:
    config = load_pointer_pilot_config(ROOT / "configs/allocation/a3_5_pointer_pilot_v1.json")
    bad = tmp_path / "a2_paper_v4" / "data"
    bad.mkdir(parents=True)
    with pytest.raises(PermissionError, match="forbidden"):
        prepare_pointer_pilot(bad, tmp_path / "manifest.json", context, config)
    clean = tmp_path / "pilot"; (clean / "frozen_test").mkdir(parents=True)
    with pytest.raises(PermissionError, match="forbidden"):
        prepare_pointer_pilot(clean, tmp_path / "manifest.json", context, config)


@pytest.mark.skipif(not (PILOT_ROOT / "manifest.json").is_file(), reason="A3.5 data not materialized yet")
def test_materialized_manifest_is_deterministic_disjoint_and_source_locked(context) -> None:
    config = load_pointer_pilot_config(ROOT / "configs/allocation/a3_5_pointer_pilot_v1.json")
    manifest = load_pointer_manifest(PILOT_ROOT / "manifest.json")
    assert len(manifest.records) == 144
    assert {item.split for item in manifest.records} == {"train", "validation"}
    historical = [ROOT / f"data/manifests/allocation/a2_paper_manifest_v{version}.json" for version in (2, 3, 4)]
    assert audit_manifest_overlap(manifest, historical) == {"task_group_ids": [], "instance_ids": []}
    prepared = prepare_pointer_pilot(PILOT_ROOT / "data", PILOT_ROOT / "manifest.json", context, config)
    assert len(prepared.train_examples) == 96 and len(prepared.validation_examples) == 48
    action_count = 0
    for example in prepared.train_examples + prepared.validation_examples:
        segment_index = {value: index for index, value in enumerate(example.graph.segment_ids)}
        robot_index = {value: index for index, value in enumerate(example.graph.robot_ids)}
        validate_teacher_prefixes(example.instance, example.teacher_actions)
        replay = replay_pointer_actions(example.instance, example.teacher_actions, context)
        assert replay is not None and verify_plan(example.instance, replay, context).feasible
        for action in example.teacher_actions:
            action_count += 1
            assert all(
                example.graph.allowed_mask[segment_index[item], robot_index[action.robot_id]]
                for item in action.unit
            )
    assert action_count == 4474


@pytest.mark.skipif(not (PILOT_ROOT / "aggregate_summary.json").is_file(), reason="A3.5 experiment not aggregated yet")
def test_three_seed_matrix_and_access_provenance_are_complete() -> None:
    summary = json.loads((PILOT_ROOT / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert summary["seeds"] == [101, 211, 307]
    assert len(summary["variant_results"]) == 5
    assert summary["integrity_checks"]["all_fifteen_shards_complete"]
    assert summary["integrity_checks"]["train_validation_only"]
    assert summary["frozen_test_generated_or_accessed"] is False
    assert summary["v4_instance_or_witness_accessed"] is False
