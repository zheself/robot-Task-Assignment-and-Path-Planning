"""Pure-PyTorch masked heterogeneous allocation models for A3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..graphs import A3Graph, GraphTeacherTarget
from ..graphs.core import RELATION_NAMES
from ..schema import AllocationInstance
from ..solvers.common import allocation_units

MODEL_FAMILIES = frozenset({"edge_mlp", "hetero_gnn", "graph_transformer"})


@dataclass(frozen=True)
class A3ModelOutput:
    assignment_logits: torch.Tensor
    order_scores: torch.Tensor
    node_embeddings: torch.Tensor


class A3AllocationModel(nn.Module):
    """Score segment–robot assignments and segment order priorities.

    The hard feasibility mask is applied inside ``forward`` and therefore also
    applies during the training loss.  This model does not schedule or repair.
    """

    def __init__(
        self,
        graph: A3Graph,
        *,
        family: str,
        hidden_dim: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if family not in MODEL_FAMILIES:
            raise ValueError(f"unknown A3 model family: {family}")
        if hidden_dim < 4 or layers < 0 or hidden_dim % heads:
            raise ValueError("hidden_dim/layers/heads are incompatible")
        self.family = family
        self.hidden_dim = hidden_dim
        self.segment_projection = _projection(graph.segment_features.shape[1], hidden_dim)
        self.robot_projection = _projection(graph.robot_features.shape[1], hidden_dim)
        self.resource_projection = _projection(graph.resource_features.shape[1], hidden_dim)
        edge_dim = graph.edge_features.shape[1]
        if family == "hetero_gnn":
            self.encoder_layers = nn.ModuleList(
                [_RelationalMessageLayer(hidden_dim, edge_dim, len(RELATION_NAMES), dropout) for _ in range(layers)]
            )
        elif family == "graph_transformer":
            self.encoder_layers = nn.ModuleList(
                [_RelationalAttentionLayer(hidden_dim, edge_dim, len(RELATION_NAMES), heads, dropout) for _ in range(layers)]
            )
        else:
            self.encoder_layers = nn.ModuleList()
        pair_dim = graph.pair_features.shape[-1]
        self.assignment_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.order_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph: A3Graph) -> A3ModelOutput:
        device = next(self.parameters()).device
        segment = self.segment_projection(_tensor(graph.segment_features, device))
        robot = self.robot_projection(_tensor(graph.robot_features, device))
        resource = self.resource_projection(_tensor(graph.resource_features, device))
        nodes = torch.cat((segment, robot, resource), dim=0)
        edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
        edge_type = torch.as_tensor(graph.edge_type, dtype=torch.long, device=device)
        edge_features = _tensor(graph.edge_features, device)
        for layer in self.encoder_layers:
            nodes = layer(nodes, edge_index, edge_type, edge_features)
        segment_embeddings = nodes[: len(graph.segment_ids)]
        robot_embeddings = nodes[
            len(graph.segment_ids) : len(graph.segment_ids) + len(graph.robot_ids)
        ]
        segment_pairs = segment_embeddings[:, None, :].expand(-1, len(graph.robot_ids), -1)
        robot_pairs = robot_embeddings[None, :, :].expand(len(graph.segment_ids), -1, -1)
        pair_features = _tensor(graph.pair_features, device)
        logits = self.assignment_head(
            torch.cat((segment_pairs, robot_pairs, pair_features), dim=-1)
        ).squeeze(-1)
        allowed = torch.as_tensor(graph.allowed_mask, dtype=torch.bool, device=device)
        logits = logits.masked_fill(~allowed, float("-inf"))
        order_scores = self.order_head(segment_embeddings).squeeze(-1)
        return A3ModelOutput(logits, order_scores, nodes)


class _RelationalMessageLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, relations: int, dropout: float) -> None:
        super().__init__()
        self.relation_linears = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(relations)]
        )
        self.edge_projection = nn.Linear(edge_dim, hidden, bias=False)
        self.self_projection = nn.Linear(hidden, hidden)
        self.normalization = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, edge_index, edge_type, edge_features):
        source, destination = edge_index
        messages = torch.zeros((len(source), nodes.shape[1]), device=nodes.device, dtype=nodes.dtype)
        for relation_id, projection in enumerate(self.relation_linears):
            selected = edge_type == relation_id
            if torch.any(selected):
                messages[selected] = projection(nodes[source[selected]])
        messages = messages + self.edge_projection(edge_features)
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, destination, messages)
        counts = torch.zeros((len(nodes), 1), device=nodes.device, dtype=nodes.dtype)
        counts.index_add_(0, destination, torch.ones((len(destination), 1), device=nodes.device, dtype=nodes.dtype))
        aggregate = aggregate / counts.clamp_min(1.0)
        return self.normalization(self.self_projection(nodes) + self.dropout(F.gelu(aggregate)))


class _RelationalAttentionLayer(nn.Module):
    def __init__(
        self, hidden: int, edge_dim: int, relations: int, heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.edge_bias = nn.Linear(edge_dim, heads, bias=False)
        self.relation_bias = nn.Parameter(torch.zeros(relations + 1, heads))
        self.relation_value = nn.Parameter(torch.empty(relations + 1, heads, self.head_dim))
        nn.init.xavier_uniform_(self.relation_value)
        self.output = nn.Linear(hidden, hidden)
        self.norm_attention = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * hidden, hidden)
        )
        self.norm_ffn = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, edge_index, edge_type, edge_features):
        count = len(nodes)
        self_nodes = torch.arange(count, device=nodes.device)
        source = torch.cat((edge_index[0], self_nodes))
        destination = torch.cat((edge_index[1], self_nodes))
        self_relation = torch.full((count,), len(self.relation_bias) - 1, device=nodes.device, dtype=torch.long)
        relation = torch.cat((edge_type, self_relation))
        zero_edges = torch.zeros((count, edge_features.shape[1]), device=nodes.device, dtype=nodes.dtype)
        all_edge_features = torch.cat((edge_features, zero_edges))
        q = self.q(nodes).reshape(count, self.heads, self.head_dim)
        k = self.k(nodes).reshape(count, self.heads, self.head_dim)
        v = self.v(nodes).reshape(count, self.heads, self.head_dim)
        score = (q[destination] * k[source]).sum(-1) / math.sqrt(self.head_dim)
        score = score + self.relation_bias[relation] + self.edge_bias(all_edge_features)
        index = destination[:, None].expand(-1, self.heads)
        maximum = torch.full((count, self.heads), float("-inf"), device=nodes.device, dtype=nodes.dtype)
        maximum.scatter_reduce_(0, index, score, reduce="amax", include_self=True)
        exponent = torch.exp(score - maximum[destination])
        denominator = torch.zeros((count, self.heads), device=nodes.device, dtype=nodes.dtype)
        denominator.index_add_(0, destination, exponent)
        weights = exponent / denominator[destination].clamp_min(1e-12)
        values = v[source] + self.relation_value[relation]
        weighted = weights[:, :, None] * values
        aggregate = torch.zeros((count, self.heads, self.head_dim), device=nodes.device, dtype=nodes.dtype)
        aggregate.index_add_(0, destination, weighted)
        attended = self.output(aggregate.reshape(count, -1))
        nodes = self.norm_attention(nodes + self.dropout(attended))
        return self.norm_ffn(nodes + self.dropout(self.ffn(nodes)))


def assignment_order_loss(
    output: A3ModelOutput,
    graph: A3Graph,
    instance: AllocationInstance,
    target: GraphTeacherTarget,
    *,
    assignment_weight: float = 1.0,
    order_weight: float = 0.25,
) -> tuple[torch.Tensor, Mapping[str, float]]:
    device = output.assignment_logits.device
    segment_lookup = {item: index for index, item in enumerate(graph.segment_ids)}
    target_robot = torch.as_tensor(target.target_robot_index, dtype=torch.long, device=device)
    unit_losses = []
    for unit in allocation_units(instance):
        indices = torch.as_tensor([segment_lookup[item] for item in unit], dtype=torch.long, device=device)
        robots = target_robot[indices]
        if not torch.all(robots == robots[0]):
            raise ValueError("teacher splits an atomic allocation unit")
        unit_logits = output.assignment_logits[indices].mean(dim=0, keepdim=True)
        if not torch.isfinite(unit_logits[0, robots[0]]):
            raise ValueError("teacher selects a hard-masked edge")
        unit_losses.append(F.cross_entropy(unit_logits, robots[:1]))
    assignment_loss = torch.stack(unit_losses).mean()
    order_pairs = []
    order_index = torch.as_tensor(target.target_order_index, dtype=torch.long, device=device)
    for robot_index in range(len(graph.robot_ids)):
        members = torch.nonzero(target_robot == robot_index, as_tuple=False).flatten()
        if len(members) < 2:
            continue
        for left_position in range(len(members)):
            for right_position in range(left_position + 1, len(members)):
                left = members[left_position]
                right = members[right_position]
                if order_index[left] < order_index[right]:
                    order_pairs.append(output.order_scores[left] - output.order_scores[right])
                elif order_index[right] < order_index[left]:
                    order_pairs.append(output.order_scores[right] - output.order_scores[left])
    order_loss = (
        F.softplus(-torch.stack(order_pairs)).mean()
        if order_pairs
        else torch.zeros((), dtype=assignment_loss.dtype, device=device)
    )
    total = assignment_weight * assignment_loss + order_weight * order_loss
    return total, {
        "total": float(total.detach()),
        "assignment": float(assignment_loss.detach()),
        "order": float(order_loss.detach()),
        "order_pairs": float(len(order_pairs)),
    }


def atomic_unit_assignment_accuracy(
    logits: torch.Tensor,
    graph: A3Graph,
    instance: AllocationInstance,
    target: GraphTeacherTarget,
) -> tuple[int, int]:
    segment_lookup = {item: index for index, item in enumerate(graph.segment_ids)}
    correct = 0
    units = allocation_units(instance)
    for unit in units:
        indices = [segment_lookup[item] for item in unit]
        prediction = int(torch.argmax(logits[indices].mean(dim=0)).item())
        targets = {int(target.target_robot_index[item]) for item in indices}
        if len(targets) != 1:
            raise ValueError("teacher splits an atomic allocation unit")
        correct += prediction == next(iter(targets))
    return correct, len(units)


def _projection(input_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())


def _tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)
