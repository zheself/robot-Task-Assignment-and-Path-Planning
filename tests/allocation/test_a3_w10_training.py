from __future__ import annotations

import json
from pathlib import Path

from safe_residual_rl.allocation import load_oracle_context
from safe_residual_rl.allocation.a3_protocol import (
    load_a3_development_config,
    prepare_a3_development,
)
from safe_residual_rl.allocation.models import A3AllocationModel
from safe_residual_rl.allocation.training import (
    evaluate_a3_by_cell,
    train_a3_with_validation_selection,
)

ROOT = Path(__file__).resolve().parents[2]


def test_w10_protocol_and_preprocessing_match_w9() -> None:
    config, digest = load_a3_development_config(
        ROOT / "configs/allocation/a3_development_v1.json"
    )
    assert len(digest) == 64
    assert config["models"]["families"] == [
        "edge_mlp",
        "hetero_gnn",
        "graph_transformer",
    ]
    assert config["training"]["seeds"] == [17, 29, 43]
    assert config["training"]["device"] == "cpu"
    context = load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")
    prepared = prepare_a3_development(
        ROOT / "outputs/phase1_allocation/a3_development_v1/data", context
    )
    w9 = json.loads(
        (ROOT / "reports/phase1_allocation/a3_w9_foundation_v1_summary.json").read_text()
    )
    assert len(prepared.train_examples) == 192
    assert len(prepared.validation_examples) == 48
    assert prepared.access_record_count == 240
    assert prepared.vocabulary.sha256 == w9["vocabulary_sha256"]
    assert prepared.normalizer.sha256 == w9["normalizer_sha256"]


def test_validation_selection_is_same_seed_deterministic() -> None:
    context = load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")
    prepared = prepare_a3_development(
        ROOT / "outputs/phase1_allocation/a3_development_v1/data", context
    )
    weights = {
        "makespan": 1.0,
        "load_variance": 0.05,
        "travel_setup_time": 0.1,
        "priority_tardiness": 1.0,
    }
    keyword = {
        "family": "edge_mlp",
        "seed": 17,
        "max_epochs": 2,
        "patience": 1,
        "hidden_dim": 16,
        "layers": 1,
        "heads": 4,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "assignment_weight": 1.0,
        "order_weight": 0.25,
    }
    train = prepared.train_examples[:2]
    validation = prepared.validation_examples[:2]
    first = train_a3_with_validation_selection(
        train, validation, context, weights, **keyword
    )
    second = train_a3_with_validation_selection(
        train, validation, context, weights, **keyword
    )
    assert first.state_sha256 == second.state_sha256
    assert first.history == second.history
    assert 1 <= first.best_epoch <= first.epochs_completed <= 2
    assert first.validation_evaluation.instance_count == 2

    model = A3AllocationModel(
        train[0].graph,
        family="edge_mlp",
        hidden_dim=16,
        layers=1,
        heads=4,
        dropout=0.0,
    )
    model.load_state_dict(first.state_dict)
    by_cell = evaluate_a3_by_cell(
        model,
        validation,
        context,
        weights,
        assignment_weight=1.0,
        order_weight=0.25,
    )
    assert set(by_cell) == {validation[0].cell_id}
    assert next(iter(by_cell.values())).instance_count == 2
