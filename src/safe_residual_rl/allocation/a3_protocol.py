"""Shared, split-guarded preparation for A3 W10 runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .graphs import (
    FeatureNormalizer,
    FeatureVocabulary,
    build_a3_graph,
    build_teacher_target,
    discover_a3_records,
    fit_feature_normalizer,
    fit_feature_vocabulary,
)
from .oracle import OracleContext
from .training import A3TrainingExample


@dataclass(frozen=True)
class PreparedA3Development:
    train_examples: tuple[A3TrainingExample, ...]
    validation_examples: tuple[A3TrainingExample, ...]
    vocabulary: FeatureVocabulary
    normalizer: FeatureNormalizer
    access_sha256: str
    access_record_count: int


def prepare_a3_development(
    data_root: str | Path,
    context: OracleContext,
    *,
    expected_train: int = 192,
    expected_validation: int = 48,
) -> PreparedA3Development:
    root = Path(data_root).resolve()
    forbidden = [
        split
        for split in ("frozen_test", "stress")
        if (root / split).exists() or (root / "witnesses" / split).exists()
    ]
    if forbidden:
        raise PermissionError(f"forbidden split directories in A3 development root: {forbidden}")
    train_records = discover_a3_records(root, "train", context)
    validation_records = discover_a3_records(root, "validation", context)
    if (len(train_records), len(validation_records)) != (
        expected_train,
        expected_validation,
    ):
        raise ValueError("A3 development record count mismatch")
    leakage = _leakage(train_records, validation_records)
    if leakage:
        raise ValueError(f"A3 train/validation leakage: {leakage}")
    vocabulary = fit_feature_vocabulary(
        [item.instance for item in train_records], split="train"
    )
    train_raw = tuple(
        build_a3_graph(item.instance, context, vocabulary, split="train")
        for item in train_records
    )
    validation_raw = tuple(
        build_a3_graph(item.instance, context, vocabulary, split="validation")
        for item in validation_records
    )
    normalizer = fit_feature_normalizer(train_raw, split="train")
    train_graphs = tuple(normalizer.transform(item) for item in train_raw)
    validation_graphs = tuple(normalizer.transform(item) for item in validation_raw)
    train_examples = tuple(
        A3TrainingExample(
            graph,
            record.instance,
            build_teacher_target(graph, record.teacher_plan, record.teacher_sha256),
            record.cell_id,
        )
        for graph, record in zip(train_graphs, train_records)
    )
    validation_examples = tuple(
        A3TrainingExample(
            graph,
            record.instance,
            build_teacher_target(graph, record.teacher_plan, record.teacher_sha256),
            record.cell_id,
        )
        for graph, record in zip(validation_graphs, validation_records)
    )
    access_records = [
        {
            "split": item.split,
            "cell_id": item.cell_id,
            "instance_id": item.instance.instance_id,
            "record_sha256": item.record_sha256,
        }
        for item in train_records + validation_records
    ]
    access_sha256 = _digest(
        {
            "version": "a3-development-access-manifest-v1",
            "records": access_records,
        }
    )
    return PreparedA3Development(
        train_examples,
        validation_examples,
        vocabulary,
        normalizer,
        access_sha256,
        len(access_records),
    )


def load_a3_development_config(path: str | Path) -> tuple[dict[str, object], str]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") != "a3-train-validation-development-v1":
        raise ValueError("unexpected A3 development config version")
    guard = raw["access_guard"]
    if guard["allowed_splits"] != ["train", "validation"]:
        raise ValueError("A3 allowed split contract changed")
    if guard["forbidden_splits"] != ["frozen_test", "stress"]:
        raise ValueError("A3 forbidden split contract changed")
    return raw, hashlib.sha256(path.read_bytes()).hexdigest()


def _leakage(train_records, validation_records) -> dict[str, list[str]]:
    result = {}
    for name, accessor in (
        ("workpiece", lambda item: {item.instance.workpiece_id}),
        ("layout", lambda item: {item.instance.layout_id}),
        (
            "parent_curve",
            lambda item: {
                segment.parent_curve_id for segment in item.instance.segments
            },
        ),
    ):
        train_values = set().union(*(accessor(item) for item in train_records))
        validation_values = set().union(
            *(accessor(item) for item in validation_records)
        )
        overlap = sorted(train_values & validation_values)
        if overlap:
            result[name] = overlap
    return result


def _digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
