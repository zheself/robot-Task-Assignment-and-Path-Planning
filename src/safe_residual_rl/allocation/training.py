"""Deterministic A3 imitation training and validation utilities."""

from __future__ import annotations

import hashlib
import random
import statistics
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from .decoding import decode_masked_candidate
from .graphs import A3Graph, GraphTeacherTarget
from .models import (
    A3AllocationModel,
    assignment_order_loss,
    atomic_unit_assignment_accuracy,
)
from .oracle import OracleContext
from .schema import AllocationInstance
from .verifier import verify_plan


@dataclass(frozen=True)
class A3TrainingExample:
    graph: A3Graph
    instance: AllocationInstance
    target: GraphTeacherTarget
    cell_id: str


@dataclass(frozen=True)
class A3Evaluation:
    instance_count: int
    mean_loss: float
    atomic_unit_correct: int
    atomic_unit_total: int
    atomic_unit_accuracy: float
    verified_candidates: int
    verified_candidate_coverage: float
    mean_verified_weighted_proxy_score: float | None
    median_inference_runtime_s: float
    failures: tuple[Mapping[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class A3TrainingRun:
    seed: int
    family: str
    epochs: int
    state_sha256: str
    history: tuple[Mapping[str, float], ...]
    train_evaluation: A3Evaluation
    validation_evaluation: A3Evaluation
    state_dict: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class A3SelectedTrainingRun:
    seed: int
    family: str
    max_epochs: int
    epochs_completed: int
    best_epoch: int
    stopped_early: bool
    state_sha256: str
    history: tuple[Mapping[str, float], ...]
    train_evaluation: A3Evaluation
    validation_evaluation: A3Evaluation
    state_dict: Mapping[str, torch.Tensor]


def train_a3_imitation(
    train_examples: Sequence[A3TrainingExample],
    validation_examples: Sequence[A3TrainingExample],
    context: OracleContext,
    objective_weights: Mapping[str, float],
    *,
    family: str,
    seed: int,
    epochs: int,
    hidden_dim: int,
    layers: int,
    heads: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    assignment_weight: float,
    order_weight: float,
) -> A3TrainingRun:
    if not train_examples or not validation_examples or epochs < 1:
        raise ValueError("A3 training requires non-empty train/validation and epochs")
    set_a3_determinism(seed)
    model = A3AllocationModel(
        train_examples[0].graph,
        family=family,
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[Mapping[str, float]] = []
    for epoch in range(epochs):
        model.train()
        indices = list(range(len(train_examples)))
        random.Random(seed + epoch).shuffle(indices)
        losses = []
        for index in indices:
            example = train_examples[index]
            optimizer.zero_grad(set_to_none=True)
            output = model(example.graph)
            loss, _ = assignment_order_loss(
                output,
                example.graph,
                example.instance,
                example.target,
                assignment_weight=assignment_weight,
                order_weight=order_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_a3_model(
            model,
            validation_examples,
            context,
            objective_weights,
            assignment_weight=assignment_weight,
            order_weight=order_weight,
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_mean_loss": float(np.mean(losses)),
                "validation_mean_loss": validation.mean_loss,
                "validation_assignment_accuracy": validation.atomic_unit_accuracy,
                "validation_verified_coverage": validation.verified_candidate_coverage,
            }
        )
    train_evaluation = evaluate_a3_model(
        model,
        train_examples,
        context,
        objective_weights,
        assignment_weight=assignment_weight,
        order_weight=order_weight,
    )
    validation_evaluation = evaluate_a3_model(
        model,
        validation_examples,
        context,
        objective_weights,
        assignment_weight=assignment_weight,
        order_weight=order_weight,
    )
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return A3TrainingRun(
        seed,
        family,
        epochs,
        state_dict_sha256(state),
        tuple(history),
        train_evaluation,
        validation_evaluation,
        state,
    )


def train_a3_with_validation_selection(
    train_examples: Sequence[A3TrainingExample],
    validation_examples: Sequence[A3TrainingExample],
    context: OracleContext,
    objective_weights: Mapping[str, float],
    *,
    family: str,
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
    assignment_weight: float,
    order_weight: float,
) -> A3SelectedTrainingRun:
    """Train with validation-only checkpoint selection and early stopping.

    Checkpoints are ordered by verified candidate coverage, conditional proxy
    score, then atomic-unit assignment accuracy. Runtime is deliberately not an
    epoch tie-breaker because architecture is fixed within a run; it remains a
    registered cross-family tie-breaker in the W10 aggregate.
    """
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    if not train_examples or not validation_examples:
        raise ValueError("A3 W10 needs non-empty train and validation examples")
    set_a3_determinism(seed)
    model = A3AllocationModel(
        train_examples[0].graph,
        family=family,
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[Mapping[str, float]] = []
    best_key: tuple[float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(max_epochs):
        losses = _train_one_epoch(
            model,
            optimizer,
            train_examples,
            seed + epoch,
            gradient_clip_norm,
            assignment_weight,
            order_weight,
        )
        validation = evaluate_a3_model(
            model,
            validation_examples,
            context,
            objective_weights,
            assignment_weight=assignment_weight,
            order_weight=order_weight,
        )
        selection_key = _checkpoint_key(validation)
        improved = best_key is None or selection_key > best_key
        if improved:
            best_key = selection_key
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_mean_loss": float(np.mean(losses)),
                "validation_mean_loss": validation.mean_loss,
                "validation_assignment_accuracy": validation.atomic_unit_accuracy,
                "validation_verified_coverage": validation.verified_candidate_coverage,
                "validation_mean_verified_weighted_proxy_score": validation.mean_verified_weighted_proxy_score,
                "selected_checkpoint": float(improved),
                "stale_epochs": float(stale_epochs),
            }
        )
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("A3 W10 did not produce a checkpoint")
    model.load_state_dict(best_state)
    train_evaluation = evaluate_a3_model(
        model,
        train_examples,
        context,
        objective_weights,
        assignment_weight=assignment_weight,
        order_weight=order_weight,
    )
    validation_evaluation = evaluate_a3_model(
        model,
        validation_examples,
        context,
        objective_weights,
        assignment_weight=assignment_weight,
        order_weight=order_weight,
    )
    return A3SelectedTrainingRun(
        seed,
        family,
        max_epochs,
        len(history),
        best_epoch,
        len(history) < max_epochs,
        state_dict_sha256(best_state),
        tuple(history),
        train_evaluation,
        validation_evaluation,
        best_state,
    )


def evaluate_a3_model(
    model: A3AllocationModel,
    examples: Sequence[A3TrainingExample],
    context: OracleContext,
    objective_weights: Mapping[str, float],
    *,
    assignment_weight: float,
    order_weight: float,
) -> A3Evaluation:
    model.eval()
    losses: list[float] = []
    correct = total = verified = 0
    scores: list[float] = []
    runtimes: list[float] = []
    failures: list[Mapping[str, object]] = []
    with torch.no_grad():
        for example in examples:
            started = time.perf_counter()
            output = model(example.graph)
            loss, _ = assignment_order_loss(
                output,
                example.graph,
                example.instance,
                example.target,
                assignment_weight=assignment_weight,
                order_weight=order_weight,
            )
            losses.append(float(loss))
            item_correct, item_total = atomic_unit_assignment_accuracy(
                output.assignment_logits,
                example.graph,
                example.instance,
                example.target,
            )
            correct += item_correct
            total += item_total
            candidate = decode_masked_candidate(
                example.graph,
                example.instance,
                context,
                output.assignment_logits,
                output.order_scores,
                method_id=f"a3-{model.family}-candidate-v1",
            )
            check = (
                verify_plan(example.instance, candidate.plan, context)
                if candidate.plan is not None
                else None
            )
            runtimes.append(time.perf_counter() - started)
            if check is not None and check.feasible:
                verified += 1
                scores.append(
                    sum(
                        float(objective_weights.get(key, 0.0)) * float(value)
                        for key, value in check.objective_terms
                    )
                )
            else:
                failures.append(
                    {
                        "instance_id": example.instance.instance_id,
                        "cell_id": example.cell_id,
                        "status": candidate.status,
                        "diagnostics": list(candidate.diagnostics),
                        "violation_codes": []
                        if check is None
                        else sorted({item.code for item in check.violations}),
                    }
                )
    count = len(examples)
    return A3Evaluation(
        count,
        float(np.mean(losses)),
        correct,
        total,
        correct / total if total else 0.0,
        verified,
        verified / count if count else 0.0,
        float(np.mean(scores)) if scores else None,
        float(statistics.median(runtimes)) if runtimes else 0.0,
        tuple(failures),
    )


def evaluate_a3_by_cell(
    model: A3AllocationModel,
    examples: Sequence[A3TrainingExample],
    context: OracleContext,
    objective_weights: Mapping[str, float],
    *,
    assignment_weight: float,
    order_weight: float,
) -> dict[str, A3Evaluation]:
    result = {}
    for cell_id in sorted({item.cell_id for item in examples}):
        selected = [item for item in examples if item.cell_id == cell_id]
        result[cell_id] = evaluate_a3_model(
            model,
            selected,
            context,
            objective_weights,
            assignment_weight=assignment_weight,
            order_weight=order_weight,
        )
    return result


def set_a3_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _train_one_epoch(
    model: A3AllocationModel,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[A3TrainingExample],
    shuffle_seed: int,
    gradient_clip_norm: float,
    assignment_weight: float,
    order_weight: float,
) -> list[float]:
    model.train()
    indices = list(range(len(examples)))
    random.Random(shuffle_seed).shuffle(indices)
    losses = []
    for index in indices:
        example = examples[index]
        optimizer.zero_grad(set_to_none=True)
        output = model(example.graph)
        loss, _ = assignment_order_loss(
            output,
            example.graph,
            example.instance,
            example.target,
            assignment_weight=assignment_weight,
            order_weight=order_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def _checkpoint_key(evaluation: A3Evaluation) -> tuple[float, float, float]:
    score = evaluation.mean_verified_weighted_proxy_score
    return (
        evaluation.verified_candidate_coverage,
        -float("inf") if score is None else -score,
        evaluation.atomic_unit_accuracy,
    )
