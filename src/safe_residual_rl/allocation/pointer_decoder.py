"""Atomic-unit–robot Feasible-Pair Autoregressive Decoder for A3.5."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .graphs import A3Graph
from .models import A3AllocationModel
from .oracle import OracleContext
from .pointer_pilot import PointerAction, replay_pointer_actions, unit_dependencies
from .schema import AllocationInstance, AllocationPlan
from .solvers.common import allocation_units
from .verifier import verify_plan


@dataclass(frozen=True)
class PairPointerOutput:
    status: str
    actions: tuple[PointerAction, ...]
    plan: AllocationPlan | None
    diagnostics: tuple[str, ...]
    dead_end_step: int | None
    hard_mask_violations: int
    atomicity_violations: int


@dataclass
class PointerState:
    assigned: torch.Tensor
    robot_load: torch.Tensor
    robot_completion: torch.Tensor
    robot_last_position: torch.Tensor
    robot_last_unit: torch.Tensor
    resource_usage: torch.Tensor
    last_unit_index: int | None
    last_robot_index: int | None
    step: int


class FeasiblePairPointer(nn.Module):
    """Autoregressively score feasible ``(atomic unit, robot)`` pairs.

    The encoder is the unchanged A3 encoder implementation.  The pointer head
    conditions each decision on accumulated proxy load/completion, robot last
    position/unit, resource use, predecessor satisfaction and the previous
    action.  No scheduler repair or fallback is contained in this module.
    """

    def __init__(
        self,
        graph: A3Graph,
        *,
        encoder_family: str,
        hidden_dim: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if encoder_family not in {"hetero_gnn", "graph_transformer"}:
            raise ValueError("Pair-Pointer requires a graph encoder")
        self.encoder_family = encoder_family
        self.hidden_dim = hidden_dim
        self.encoder = A3AllocationModel(
            graph,
            family=encoder_family,
            hidden_dim=hidden_dim,
            layers=layers,
            heads=heads,
            dropout=dropout,
        )
        pair_dim = graph.pair_features.shape[-1]
        self.global_dynamic = nn.Sequential(nn.Linear(7, hidden_dim), nn.GELU())
        self.query = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.score = nn.Sequential(
            nn.Linear(3 * hidden_dim + pair_dim + 8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, graph: A3Graph):
        output = self.encoder(graph)
        count_segment = len(graph.segment_ids)
        count_robot = len(graph.robot_ids)
        segment = output.node_embeddings[:count_segment]
        robot = output.node_embeddings[count_segment : count_segment + count_robot]
        return output.node_embeddings, segment, robot

    def teacher_forced_loss(
        self,
        graph: A3Graph,
        instance: AllocationInstance,
        teacher_actions: Sequence[PointerAction],
    ) -> tuple[torch.Tensor, int, int]:
        nodes, segment_embeddings, robot_embeddings = self.encode(graph)
        prepared = _prepare(instance, graph, segment_embeddings)
        state = _initial_state(instance, graph, nodes.device)
        unit_lookup = {item: index for index, item in enumerate(prepared.unit_ids)}
        robot_lookup = {item: index for index, item in enumerate(graph.robot_ids)}
        losses = []
        correct = 0
        for expected in teacher_actions:
            scores, mask = self._scores(
                graph, instance, nodes, robot_embeddings, prepared, state
            )
            unit_index = unit_lookup[expected.unit_id]
            robot_index = robot_lookup[expected.robot_id]
            flat_target = unit_index * len(graph.robot_ids) + robot_index
            flat_scores = scores.reshape(-1)
            flat_mask = mask.reshape(-1)
            if not bool(flat_mask[flat_target]):
                raise ValueError("teacher prefix selects a masked pair")
            losses.append(F.cross_entropy(flat_scores[None, :], torch.tensor([flat_target], device=flat_scores.device)))
            correct += int(torch.argmax(flat_scores).item() == flat_target)
            _update_state(state, instance, graph, prepared, unit_index, robot_index)
        if state.step != len(prepared.units) or not bool(torch.all(state.assigned)):
            raise ValueError("teacher sequence did not assign every unit")
        return torch.stack(losses).mean(), correct, len(losses)

    def greedy_rollout(
        self,
        graph: A3Graph,
        instance: AllocationInstance,
        context: OracleContext,
    ) -> PairPointerOutput:
        self.eval()
        with torch.no_grad():
            nodes, segment_embeddings, robot_embeddings = self.encode(graph)
            prepared = _prepare(instance, graph, segment_embeddings)
            state = _initial_state(instance, graph, nodes.device)
            actions: list[PointerAction] = []
            while state.step < len(prepared.units):
                scores, mask = self._scores(
                    graph, instance, nodes, robot_embeddings, prepared, state
                )
                if not bool(torch.any(mask)):
                    return PairPointerOutput(
                        "decoder_dead_end", tuple(actions), None,
                        ("NO_CURRENT_FEASIBLE_PAIR", "NO_REPAIR_APPLIED"),
                        state.step, 0, 0,
                    )
                selected = int(torch.argmax(scores.reshape(-1)).item())
                unit_index, robot_index = divmod(selected, len(graph.robot_ids))
                if not bool(mask[unit_index, robot_index]):
                    return PairPointerOutput(
                        "mask_integrity_failure", tuple(actions), None,
                        ("MASKED_PAIR_SELECTED",), state.step, 1, 0,
                    )
                unit = prepared.units[unit_index]
                actions.append(PointerAction(prepared.unit_ids[unit_index], unit, graph.robot_ids[robot_index]))
                _update_state(state, instance, graph, prepared, unit_index, robot_index)
            if len({item.unit_id for item in actions}) != len(prepared.units):
                return PairPointerOutput(
                    "atomicity_failure", tuple(actions), None,
                    ("DUPLICATE_OR_MISSING_UNIT",), None, 0, 1,
                )
            plan = replay_pointer_actions(instance, actions, context, method_id=f"a3-5-{self.encoder_family}-pair-pointer-v1")
            if plan is None:
                return PairPointerOutput(
                    "schedule_infeasible", tuple(actions), None,
                    ("A1_SCHEDULER_REJECTED_RAW_POINTER_CANDIDATE", "NO_REPAIR_APPLIED"),
                    None, 0, 0,
                )
            checked = verify_plan(instance, plan, context)
            if not checked.feasible:
                return PairPointerOutput(
                    "verification_failed", tuple(actions), plan,
                    tuple(sorted({item.code for item in checked.violations})) + ("NO_REPAIR_APPLIED",),
                    None, 0, 0,
                )
            return PairPointerOutput(
                "feasible", tuple(actions), plan,
                ("HARD_PAIR_MASK_APPLIED", "ATOMIC_UNITS_PRESERVED", "NO_REPAIR_APPLIED"),
                None, 0, 0,
            )

    def feasible_pair_mask(
        self, graph: A3Graph, instance: AllocationInstance, state: PointerState
    ) -> torch.Tensor:
        units = allocation_units(instance)
        dependencies = unit_dependencies(instance, units)
        unit_ids = tuple("+".join(item) for item in units)
        segment_lookup = {item: index for index, item in enumerate(graph.segment_ids)}
        completed = {unit_ids[index] for index in range(len(units)) if bool(state.assigned[index])}
        rows = []
        for index, unit in enumerate(units):
            allowed = torch.as_tensor(
                np.all(graph.allowed_mask[[segment_lookup[item] for item in unit]], axis=0),
                dtype=torch.bool,
                device=state.assigned.device,
            )
            ready = (not bool(state.assigned[index])) and dependencies[unit_ids[index]] <= completed
            rows.append(allowed & ready)
        return torch.stack(rows)

    def _scores(
        self,
        graph: A3Graph,
        instance: AllocationInstance,
        nodes: torch.Tensor,
        robot_embeddings: torch.Tensor,
        prepared: "_PreparedUnits",
        state: PointerState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.feasible_pair_mask(graph, instance, state)
        device = nodes.device
        unit_embeddings = prepared.unit_embeddings
        graph_embedding = nodes.mean(dim=0)
        zero = torch.zeros(self.hidden_dim, device=device)
        last_unit = zero if state.last_unit_index is None else unit_embeddings[state.last_unit_index]
        last_robot = zero if state.last_robot_index is None else robot_embeddings[state.last_robot_index]
        resource_mean = float(state.resource_usage.mean()) if state.resource_usage.numel() else 0.0
        resource_max = float(state.resource_usage.max()) if state.resource_usage.numel() else 0.0
        global_values = torch.tensor(
            [
                state.step / max(1, len(prepared.units)),
                float(state.robot_load.mean()),
                float(state.robot_load.max()),
                float(state.robot_completion.mean()),
                float(state.robot_completion.max()),
                resource_mean,
                resource_max,
            ],
            dtype=torch.float32,
            device=device,
        )
        query = self.query(torch.cat((graph_embedding, last_unit, last_robot, self.global_dynamic(global_values))))
        query_pairs = query[None, None, :].expand(len(prepared.units), len(graph.robot_ids), -1)
        unit_pairs = unit_embeddings[:, None, :].expand(-1, len(graph.robot_ids), -1)
        robot_pairs = robot_embeddings[None, :, :].expand(len(prepared.units), -1, -1)
        pair_features = _unit_pair_features(graph, prepared.units, device)
        dynamic = _dynamic_pair_features(instance, graph, prepared, state, device)
        scores = self.score(torch.cat((query_pairs, unit_pairs, robot_pairs, pair_features, dynamic), dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        return scores, mask


@dataclass(frozen=True)
class _PreparedUnits:
    units: tuple[tuple[str, ...], ...]
    unit_ids: tuple[str, ...]
    unit_embeddings: torch.Tensor
    unit_positions: torch.Tensor
    unit_durations: torch.Tensor
    unit_resource_indices: tuple[tuple[int, ...], ...]


def _prepare(instance: AllocationInstance, graph: A3Graph, segment_embeddings: torch.Tensor) -> _PreparedUnits:
    units = allocation_units(instance)
    segment_lookup = {item: index for index, item in enumerate(graph.segment_ids)}
    segment_by_id = {item.id: item for item in instance.segments}
    resource_lookup = {item: index for index, item in enumerate(graph.resource_ids)}
    embeddings = []
    positions = []
    durations = []
    resources = []
    for unit in units:
        indices = [segment_lookup[item] for item in unit]
        embeddings.append(segment_embeddings[indices].mean(dim=0))
        first = min((segment_by_id[item] for item in unit), key=lambda item: (item.segment_index, item.id))
        last = max((segment_by_id[item] for item in unit), key=lambda item: (item.segment_index, item.id))
        positions.append((*first.start_pose.position_m, *last.end_pose.position_m))
        durations.append(sum(segment_by_id[item].process_duration_s for item in unit))
        resources.append(tuple(sorted({resource_lookup[value] for item in unit for value in segment_by_id[item].shared_resource_ids})))
    return _PreparedUnits(
        tuple(tuple(item) for item in units),
        tuple("+".join(item) for item in units),
        torch.stack(embeddings),
        torch.tensor(positions, dtype=torch.float32, device=segment_embeddings.device),
        torch.tensor(durations, dtype=torch.float32, device=segment_embeddings.device),
        tuple(resources),
    )


def _initial_state(instance: AllocationInstance, graph: A3Graph, device: torch.device) -> PointerState:
    bases = torch.tensor([item.base_pose.position_m for item in sorted(instance.robots, key=lambda item: item.id)], dtype=torch.float32, device=device)
    return PointerState(
        assigned=torch.zeros(len(allocation_units(instance)), dtype=torch.bool, device=device),
        robot_load=torch.zeros(len(graph.robot_ids), dtype=torch.float32, device=device),
        robot_completion=torch.zeros(len(graph.robot_ids), dtype=torch.float32, device=device),
        robot_last_position=bases,
        robot_last_unit=torch.full((len(graph.robot_ids),), -1, dtype=torch.long, device=device),
        resource_usage=torch.zeros(len(graph.resource_ids), dtype=torch.float32, device=device),
        last_unit_index=None,
        last_robot_index=None,
        step=0,
    )


def _update_state(
    state: PointerState,
    instance: AllocationInstance,
    graph: A3Graph,
    prepared: _PreparedUnits,
    unit_index: int,
    robot_index: int,
) -> None:
    if bool(state.assigned[unit_index]):
        raise ValueError("Pair-Pointer attempted to select a unit twice")
    robot = sorted(instance.robots, key=lambda item: item.id)[robot_index]
    entry = prepared.unit_positions[unit_index, :3]
    exit_position = prepared.unit_positions[unit_index, 3:]
    travel = torch.linalg.vector_norm(state.robot_last_position[robot_index] - entry) / robot.nominal_cartesian_speed_m_s
    duration = prepared.unit_durations[unit_index]
    increment = travel + duration
    state.assigned[unit_index] = True
    state.robot_load[robot_index] += increment
    state.robot_completion[robot_index] += increment
    state.robot_last_position[robot_index] = exit_position
    state.robot_last_unit[robot_index] = unit_index
    for resource_index in prepared.unit_resource_indices[unit_index]:
        state.resource_usage[resource_index] += duration
    state.last_unit_index = unit_index
    state.last_robot_index = robot_index
    state.step += 1


def _unit_pair_features(graph: A3Graph, units: Sequence[Sequence[str]], device: torch.device) -> torch.Tensor:
    lookup = {item: index for index, item in enumerate(graph.segment_ids)}
    values = [torch.as_tensor(graph.pair_features[[lookup[item] for item in unit]], dtype=torch.float32, device=device).mean(dim=0) for unit in units]
    return torch.stack(values)


def _dynamic_pair_features(
    instance: AllocationInstance,
    graph: A3Graph,
    prepared: _PreparedUnits,
    state: PointerState,
    device: torch.device,
) -> torch.Tensor:
    count_u, count_r = len(prepared.units), len(graph.robot_ids)
    result = torch.zeros((count_u, count_r, 8), dtype=torch.float32, device=device)
    max_horizon = max(item.availability.end_s - item.availability.start_s for item in instance.robots)
    max_horizon = max(max_horizon, 1.0)
    resource_denominator = max_horizon
    for unit_index in range(count_u):
        entry = prepared.unit_positions[unit_index, :3]
        resource_load = sum(float(state.resource_usage[item]) for item in prepared.unit_resource_indices[unit_index]) / resource_denominator
        for robot_index, robot in enumerate(sorted(instance.robots, key=lambda item: item.id)):
            distance = float(torch.linalg.vector_norm(state.robot_last_position[robot_index] - entry))
            result[unit_index, robot_index] = torch.tensor(
                [
                    float(state.robot_load[robot_index]) / max_horizon,
                    float(state.robot_completion[robot_index]) / max_horizon,
                    distance,
                    float(prepared.unit_durations[unit_index]) / max_horizon,
                    resource_load,
                    float(state.robot_last_unit[robot_index] == unit_index),
                    float(state.last_robot_index == robot_index),
                    state.step / max(1, count_u),
                ],
                device=device,
            )
    return result
