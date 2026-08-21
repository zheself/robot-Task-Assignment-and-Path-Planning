from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from safe_residual_rl.allocation import load_oracle_context, verify_plan
from safe_residual_rl.allocation.decoding import decode_masked_candidate
from safe_residual_rl.allocation.graphs import (
    build_a3_graph,
    build_teacher_target,
    discover_a3_records,
    fit_feature_normalizer,
    fit_feature_vocabulary,
    load_a3_record,
)
from safe_residual_rl.allocation.models import (
    A3AllocationModel,
    assignment_order_loss,
    atomic_unit_assignment_accuracy,
)
from safe_residual_rl.allocation.solvers.common import allocation_units

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ROOT = ROOT / "outputs/phase1_allocation/a3_development_v1/data"


@pytest.fixture(scope="module")
def context():
    return load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")


@pytest.fixture(scope="module")
def records(context):
    train = discover_a3_records(INSTANCE_ROOT, "train", context)
    validation = discover_a3_records(INSTANCE_ROOT, "validation", context)
    return train, validation


@pytest.fixture(scope="module")
def vocabulary(records):
    train, _ = records
    return fit_feature_vocabulary([item.instance for item in train], split="train")


@pytest.fixture(scope="module")
def raw_graphs(records, vocabulary, context):
    train, validation = records
    train_graphs = tuple(
        build_a3_graph(item.instance, context, vocabulary, split="train") for item in train
    )
    validation_graphs = tuple(
        build_a3_graph(item.instance, context, vocabulary, split="validation")
        for item in validation
    )
    return train_graphs, validation_graphs


def test_access_guard_and_expected_counts(records, context) -> None:
    train, validation = records
    assert (len(train), len(validation)) == (192, 48)
    with pytest.raises(PermissionError, match="forbids split"):
        discover_a3_records(INSTANCE_ROOT, "frozen_test", context)
    with pytest.raises(PermissionError, match="forbids split"):
        discover_a3_records(INSTANCE_ROOT, "stress", context)


def test_train_validation_groups_do_not_leak(records) -> None:
    train, validation = records
    for accessor in (
        lambda item: {item.instance.workpiece_id},
        lambda item: {item.instance.layout_id},
        lambda item: {segment.parent_curve_id for segment in item.instance.segments},
    ):
        train_keys = set().union(*(accessor(item) for item in train))
        validation_keys = set().union(*(accessor(item) for item in validation))
        assert not train_keys & validation_keys


def test_teacher_hash_tampering_is_rejected(records, context, tmp_path) -> None:
    train, _ = records
    record = train[0]
    instance_dir = tmp_path / "train" / record.cell_id
    witness_dir = tmp_path / "witnesses" / "train" / record.cell_id
    instance_dir.mkdir(parents=True)
    witness_dir.mkdir(parents=True)
    instance_path = instance_dir / record.instance_path.name
    witness_path = witness_dir / record.witness_path.name
    instance_path.write_text(record.instance_path.read_text(encoding="utf-8"), encoding="utf-8")
    raw = json.loads(record.witness_path.read_text(encoding="utf-8"))
    raw["witness_sha256"] = "0" * 64
    witness_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="verification failed"):
        load_a3_record(instance_path, witness_path, "train", record.cell_id, context)


def test_graph_is_canonical_under_input_tuple_permutations(records, vocabulary, context) -> None:
    train, _ = records
    instance = train[0].instance
    permuted = replace(
        instance,
        segments=tuple(reversed(instance.segments)),
        robots=tuple(reversed(instance.robots)),
        resources=tuple(reversed(instance.resources)),
    )
    left = build_a3_graph(instance, context, vocabulary, split="train")
    right = build_a3_graph(permuted, context, vocabulary, split="train")
    assert left.canonical_sha256() == right.canonical_sha256()


def test_vocabulary_and_normalizer_are_train_only(records, raw_graphs) -> None:
    train, validation = records
    train_graphs, validation_graphs = raw_graphs
    vocabulary = fit_feature_vocabulary([item.instance for item in train], split="train")
    assert vocabulary.fit_split == "train"
    assert vocabulary.fit_instance_count == 192
    with pytest.raises(ValueError, match="train only"):
        fit_feature_vocabulary([item.instance for item in validation], split="validation")
    normalizer = fit_feature_normalizer(train_graphs, split="train")
    assert normalizer.fit_graph_count == 192
    assert normalizer.fit_split == "train"
    assert normalizer.transform(validation_graphs[0]).normalizer_sha256 == normalizer.sha256
    with pytest.raises(ValueError, match="exclusively"):
        fit_feature_normalizer(validation_graphs, split="validation")


@pytest.mark.parametrize("family", ["edge_mlp", "hetero_gnn", "graph_transformer"])
def test_model_families_apply_hard_mask_and_finite_loss(
    family, records, raw_graphs
) -> None:
    train, _ = records
    train_graphs, _ = raw_graphs
    normalizer = fit_feature_normalizer(train_graphs, split="train")
    graph = normalizer.transform(train_graphs[0])
    target = build_teacher_target(graph, train[0].teacher_plan, train[0].teacher_sha256)
    torch.manual_seed(17)
    model = A3AllocationModel(
        graph, family=family, hidden_dim=32, layers=1, heads=4, dropout=0.0
    )
    output = model(graph)
    assert torch.all(torch.isneginf(output.assignment_logits[~torch.as_tensor(graph.allowed_mask)]))
    loss, parts = assignment_order_loss(output, graph, train[0].instance, target)
    assert torch.isfinite(loss)
    assert parts["order_pairs"] > 0
    correct, total = atomic_unit_assignment_accuracy(
        output.assignment_logits, graph, train[0].instance, target
    )
    assert 0 <= correct <= total


def test_same_seed_is_deterministic(records, raw_graphs) -> None:
    train, _ = records
    train_graphs, _ = raw_graphs
    graph = fit_feature_normalizer(train_graphs, split="train").transform(train_graphs[0])
    outputs = []
    for _ in range(2):
        torch.manual_seed(29)
        model = A3AllocationModel(
            graph, family="hetero_gnn", hidden_dim=32, layers=1, heads=4, dropout=0.0
        )
        model.eval()
        with torch.no_grad():
            outputs.append(model(graph))
    assert torch.equal(outputs[0].assignment_logits, outputs[1].assignment_logits)
    assert torch.equal(outputs[0].order_scores, outputs[1].order_scores)


def test_decoder_preserves_atomic_units_mask_and_coverage(records, raw_graphs, context) -> None:
    train, _ = records
    graph = raw_graphs[0][0]
    target = build_teacher_target(graph, train[0].teacher_plan, train[0].teacher_sha256)
    logits = np.full(graph.allowed_mask.shape, -np.inf, dtype=np.float32)
    logits[graph.allowed_mask] = 0.0
    for segment_index, robot_index in enumerate(target.target_robot_index):
        logits[segment_index, robot_index] = 10.0
    candidate = decode_masked_candidate(
        graph,
        train[0].instance,
        context,
        logits,
        -target.target_order_index.astype(np.float32),
    )
    assert candidate.status == "feasible"
    assert candidate.plan is not None
    assert verify_plan(train[0].instance, candidate.plan, context).feasible
    assignment = dict(candidate.assignment)
    assert set(assignment) == set(graph.segment_ids)
    assert all(
        len({assignment[segment_id] for segment_id in unit}) == 1
        for unit in allocation_units(train[0].instance)
    )
    for segment_id, robot_id in assignment.items():
        assert graph.allowed_mask[
            graph.segment_ids.index(segment_id), graph.robot_ids.index(robot_id)
        ]


def test_decoder_rejects_finite_invalid_logits(records, raw_graphs, context) -> None:
    train, _ = records
    selected = next(
        (record, graph)
        for record, graph in zip(train, raw_graphs[0])
        if np.any(~graph.allowed_mask)
    )
    record, graph = selected
    logits = np.zeros(graph.allowed_mask.shape, dtype=np.float32)
    logits[graph.allowed_mask] = 0.0
    with pytest.raises(ValueError, match="negative infinity"):
        decode_masked_candidate(
            graph, record.instance, context, logits, np.zeros(len(graph.segment_ids))
        )
