"""Split-guarded A3.5 preparation, training and validation evaluation."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

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
from .pointer_decoder import FeasiblePairPointer
from .pointer_pilot import (
    PilotManifest,
    PointerAction,
    canonical_teacher_actions,
    load_pointer_manifest,
    source_hashes_for_config,
    PointerPilotConfig,
)
from .training import A3TrainingExample, set_a3_determinism, state_dict_sha256
from .verifier import verify_plan


@dataclass(frozen=True)
class PointerTrainingExample:
    static: A3TrainingExample
    teacher_actions: tuple[PointerAction, ...]
    task_group_id: str

    @property
    def graph(self):
        return self.static.graph

    @property
    def instance(self):
        return self.static.instance

    @property
    def cell_id(self):
        return self.static.cell_id


@dataclass(frozen=True)
class PreparedPointerPilot:
    train_examples: tuple[PointerTrainingExample, ...]
    validation_examples: tuple[PointerTrainingExample, ...]
    vocabulary: FeatureVocabulary
    normalizer: FeatureNormalizer
    manifest: PilotManifest
    access_sha256: str


@dataclass(frozen=True)
class PointerEvaluation:
    instances: int
    mean_loss: float
    pair_correct: int
    pair_total: int
    teacher_forced_pair_accuracy: float
    complete_rollouts: int
    greedy_rollout_completion_rate: float
    verified_candidates: int
    verified_candidate_coverage: float
    hard_mask_violations: int
    atomicity_violations: int
    decoder_dead_ends: int
    conditional_weighted_proxy_score: float | None
    conditional_makespan_s: float | None
    conditional_load_imbalance_s2: float | None
    median_inference_runtime_s: float
    failures: tuple[Mapping[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PointerTrainingRun:
    variant: str
    encoder_family: str
    seed: int
    max_epochs: int
    epochs_completed: int
    best_epoch: int
    stopped_early: bool
    state_sha256: str
    history: tuple[Mapping[str, float], ...]
    train_evaluation: PointerEvaluation
    validation_evaluation: PointerEvaluation
    state_dict: Mapping[str, torch.Tensor]


def prepare_pointer_pilot(
    root: str | Path,
    manifest_path: str | Path,
    context: OracleContext,
    config: PointerPilotConfig,
) -> PreparedPointerPilot:
    data_root = Path(root).resolve()
    forbidden_tokens = set(config.raw["access_guard"]["forbidden_path_tokens"])
    if set(data_root.parts) & forbidden_tokens:
        raise PermissionError("A3.5 data root contains a forbidden corpus/split token")
    forbidden_dirs = [
        name for name in ("frozen_test", "stress")
        if (data_root / name).exists() or (data_root / "witnesses" / name).exists()
    ]
    if forbidden_dirs:
        raise PermissionError(f"A3.5 data root contains forbidden splits: {forbidden_dirs}")
    manifest = load_pointer_manifest(manifest_path)
    if manifest.config_sha256 != config.sha256:
        raise ValueError("A3.5 manifest/config hash mismatch")
    current_sources = source_hashes_for_config(config, Path(__file__).resolve().parents[3])
    if dict(manifest.source_hashes) != current_sources:
        raise ValueError("A3.5 registered source changed after data generation")
    train_records = discover_a3_records(data_root, "train", context)
    validation_records = discover_a3_records(data_root, "validation", context)
    expected_train = sum(item.train_groups * item.variants for item in config.cells)
    expected_validation = sum(item.validation_groups * item.variants for item in config.cells)
    if (len(train_records), len(validation_records)) != (expected_train, expected_validation):
        raise ValueError("A3.5 record count mismatch")
    record_by_id = {item.instance_id: item for item in manifest.records}
    loaded_ids = {item.instance.instance_id for item in train_records + validation_records}
    if loaded_ids != set(record_by_id):
        raise ValueError("A3.5 loaded records differ from manifest")
    vocabulary = fit_feature_vocabulary([item.instance for item in train_records], split="train")
    train_raw = tuple(build_a3_graph(item.instance, context, vocabulary, split="train") for item in train_records)
    validation_raw = tuple(build_a3_graph(item.instance, context, vocabulary, split="validation") for item in validation_records)
    normalizer = fit_feature_normalizer(train_raw, split="train")
    train_graphs = tuple(normalizer.transform(item) for item in train_raw)
    validation_graphs = tuple(normalizer.transform(item) for item in validation_raw)

    def make(graphs, records):
        return tuple(
            PointerTrainingExample(
                A3TrainingExample(
                    graph,
                    record.instance,
                    build_teacher_target(graph, record.teacher_plan, record.teacher_sha256),
                    record.cell_id,
                ),
                canonical_teacher_actions(record.instance, record.teacher_plan),
                record_by_id[record.instance.instance_id].task_group_id,
            )
            for graph, record in zip(graphs, records)
        )

    train = make(train_graphs, train_records)
    validation = make(validation_graphs, validation_records)
    access = [
        {
            "split": item.static.graph.split,
            "instance_id": item.instance.instance_id,
            "task_group_id": item.task_group_id,
            "graph_sha256": item.graph.canonical_sha256(),
            "teacher_sha256": item.static.target.teacher_sha256,
        }
        for item in train + validation
    ]
    access_sha = hashlib.sha256(json.dumps(access, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return PreparedPointerPilot(train, validation, vocabulary, normalizer, manifest, access_sha)


def train_pointer_with_validation_selection(
    train_examples: Sequence[PointerTrainingExample],
    validation_examples: Sequence[PointerTrainingExample],
    context: OracleContext,
    weights: Mapping[str, float],
    *,
    variant: str,
    encoder_family: str,
    seed: int,
    max_epochs: int,
    patience: int,
    hidden_dim: int,
    layers: int,
    heads: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
) -> PointerTrainingRun:
    if not train_examples or not validation_examples or max_epochs < 1 or patience < 1:
        raise ValueError("A3.5 training requires non-empty data and positive budgets")
    set_a3_determinism(seed)
    model = FeasiblePairPointer(
        train_examples[0].graph,
        encoder_family=encoder_family,
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_key = None
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        indices = list(range(len(train_examples)))
        random.Random(seed + epoch).shuffle(indices)
        losses = []
        for index in indices:
            example = train_examples[index]
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model.teacher_forced_loss(example.graph, example.instance, example.teacher_actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_pointer(model, validation_examples, context, weights)
        key = _selection_key(validation)
        improved = best_key is None or key > best_key
        if improved:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        history.append({
            "epoch": float(epoch + 1),
            "train_mean_loss": float(np.mean(losses)),
            "validation_mean_loss": validation.mean_loss,
            "validation_pair_accuracy": validation.teacher_forced_pair_accuracy,
            "validation_rollout_completion": validation.greedy_rollout_completion_rate,
            "validation_verified_coverage": validation.verified_candidate_coverage,
            "selected_checkpoint": float(improved),
        })
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("A3.5 did not produce a checkpoint")
    model.load_state_dict(best_state)
    return PointerTrainingRun(
        variant,
        encoder_family,
        seed,
        max_epochs,
        len(history),
        best_epoch,
        len(history) < max_epochs,
        state_dict_sha256(best_state),
        tuple(history),
        evaluate_pointer(model, train_examples, context, weights),
        evaluate_pointer(model, validation_examples, context, weights),
        best_state,
    )


def evaluate_pointer(
    model: FeasiblePairPointer,
    examples: Sequence[PointerTrainingExample],
    context: OracleContext,
    weights: Mapping[str, float],
) -> PointerEvaluation:
    model.eval()
    losses: list[float] = []
    pair_correct = pair_total = complete = verified = 0
    hard_mask = atomic = dead_ends = 0
    scores: list[float] = []
    makespans: list[float] = []
    imbalance: list[float] = []
    runtimes: list[float] = []
    failures: list[Mapping[str, object]] = []
    with torch.no_grad():
        for example in examples:
            loss, correct, total = model.teacher_forced_loss(example.graph, example.instance, example.teacher_actions)
            losses.append(float(loss))
            pair_correct += correct
            pair_total += total
            started = time.perf_counter()
            rollout = model.greedy_rollout(example.graph, example.instance, context)
            runtimes.append(time.perf_counter() - started)
            hard_mask += rollout.hard_mask_violations
            atomic += rollout.atomicity_violations
            dead_ends += rollout.status == "decoder_dead_end"
            complete += len(rollout.actions) == len(example.teacher_actions)
            checked = verify_plan(example.instance, rollout.plan, context) if rollout.plan is not None else None
            if checked is not None and checked.feasible:
                verified += 1
                terms = dict(checked.objective_terms)
                scores.append(sum(float(weights.get(key, 0.0)) * float(value) for key, value in terms.items()))
                makespans.append(float(terms.get("makespan", 0.0)))
                imbalance.append(float(terms.get("load_variance", 0.0)))
            else:
                violations = [] if checked is None else sorted({item.code for item in checked.violations})
                failures.append({
                    "instance_id": example.instance.instance_id,
                    "task_group_id": example.task_group_id,
                    "cell_id": example.cell_id,
                    "failure_class": _failure_class(rollout.status, violations, rollout.diagnostics),
                    "status": rollout.status,
                    "dead_end_step": rollout.dead_end_step,
                    "actions_selected": len(rollout.actions),
                    "actions_required": len(example.teacher_actions),
                    "diagnostics": list(rollout.diagnostics),
                    "violation_codes": violations,
                })
    count = len(examples)
    return PointerEvaluation(
        count,
        float(np.mean(losses)) if losses else 0.0,
        pair_correct,
        pair_total,
        pair_correct / pair_total if pair_total else 0.0,
        complete,
        complete / count if count else 0.0,
        verified,
        verified / count if count else 0.0,
        hard_mask,
        atomic,
        dead_ends,
        float(np.mean(scores)) if scores else None,
        float(np.mean(makespans)) if makespans else None,
        float(np.mean(imbalance)) if imbalance else None,
        float(statistics.median(runtimes)) if runtimes else 0.0,
        tuple(failures),
    )


def evaluate_pointer_by_cell(model, examples, context, weights) -> dict[str, PointerEvaluation]:
    return {
        cell: evaluate_pointer(model, [item for item in examples if item.cell_id == cell], context, weights)
        for cell in sorted({item.cell_id for item in examples})
    }


def _selection_key(evaluation: PointerEvaluation) -> tuple[float, float, float]:
    score = evaluation.conditional_weighted_proxy_score
    return (
        evaluation.verified_candidate_coverage,
        -float("inf") if score is None else -score,
        evaluation.teacher_forced_pair_accuracy,
    )


def _failure_class(status: str, violations: Sequence[str], diagnostics: Sequence[str]) -> str:
    values = set(violations) | set(diagnostics)
    if status == "decoder_dead_end":
        return "decoder_dead_end"
    if status == "mask_integrity_failure" or "EDGE_INFEASIBLE" in values:
        return "mask_integrity_failure"
    if status == "atomicity_failure" or "SAME_ROBOT_HANDOFF" in values:
        return "atomic_unit_violation"
    if "SEGMENT_COVERAGE" in values:
        return "incomplete_assignment"
    if "PRECEDENCE" in values or "PARENT_SEGMENT_ORDER" in values:
        return "precedence_failure"
    if "SEGMENT_TIME_WINDOW" in values or any("TIME" in item for item in values):
        return "time_window_failure"
    if "RESOURCE_CAPACITY" in values or any("RESOURCE" in item for item in values):
        return "shared_resource_failure"
    return "schedule_infeasible"
