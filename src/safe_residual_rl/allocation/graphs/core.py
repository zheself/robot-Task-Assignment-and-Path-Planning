"""Canonical heterogeneous graph tensors for continuous-process allocation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from ..features import extract_curve_features
from ..masks import build_edge_mask
from ..oracle import EdgeReason, OracleContext
from ..schema import (
    AllocationInstance,
    AllocationPlan,
    HandoffPolicy,
    ProcessDirection,
    ResourceType,
)

GRAPH_VERSION = "continuous-heterogeneous-graph-v1"
UNKNOWN_TOKEN = "<UNK>"
RELATION_NAMES = (
    "robot_to_segment",
    "segment_to_robot",
    "segment_precedes_segment",
    "segment_follows_segment",
    "segment_uses_resource",
    "resource_used_by_segment",
)
PAIR_FEATURE_NAMES = (
    "feasible",
    "travel_time_s",
    "process_time_s",
    "path_length_m",
    "kinematic_risk",
    "conflict_proxy",
    "confidence",
) + tuple(f"reason_{item.value}" for item in EdgeReason)


@dataclass(frozen=True)
class FeatureVocabulary:
    capabilities: tuple[str, ...]
    tools: tuple[str, ...]
    kinematic_models: tuple[str, ...]
    fit_split: str
    fit_instance_count: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "capabilities": list(self.capabilities),
            "tools": list(self.tools),
            "kinematic_models": list(self.kinematic_models),
            "fit_split": self.fit_split,
            "fit_instance_count": self.fit_instance_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class A3Graph:
    version: str
    instance_id: str
    split: str
    segment_ids: tuple[str, ...]
    robot_ids: tuple[str, ...]
    resource_ids: tuple[str, ...]
    segment_features: np.ndarray
    robot_features: np.ndarray
    resource_features: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    edge_features: np.ndarray
    pair_features: np.ndarray
    allowed_mask: np.ndarray
    vocabulary_sha256: str
    normalizer_sha256: str | None = None

    @property
    def node_count(self) -> int:
        return len(self.segment_ids) + len(self.robot_ids) + len(self.resource_ids)

    def canonical_sha256(self) -> str:
        digest = hashlib.sha256()
        metadata = {
            "version": self.version,
            "instance_id": self.instance_id,
            "split": self.split,
            "segment_ids": self.segment_ids,
            "robot_ids": self.robot_ids,
            "resource_ids": self.resource_ids,
            "vocabulary_sha256": self.vocabulary_sha256,
            "normalizer_sha256": self.normalizer_sha256,
        }
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        for array in (
            self.segment_features,
            self.robot_features,
            self.resource_features,
            self.edge_index,
            self.edge_type,
            self.edge_features,
            self.pair_features,
            self.allowed_mask,
        ):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode())
            digest.update(str(contiguous.shape).encode())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class GraphTeacherTarget:
    instance_id: str
    target_robot_index: np.ndarray
    target_order_index: np.ndarray
    teacher_sha256: str


@dataclass(frozen=True)
class FeatureNormalizer:
    version: str
    fit_split: str
    fit_graph_count: int
    means: tuple[tuple[str, tuple[float, ...]], ...]
    scales: tuple[tuple[str, tuple[float, ...]], ...]
    epsilon: float
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "fit_split": self.fit_split,
            "fit_graph_count": self.fit_graph_count,
            "means": {key: list(value) for key, value in self.means},
            "scales": {key: list(value) for key, value in self.scales},
            "epsilon": self.epsilon,
            "sha256": self.sha256,
        }

    def transform(self, graph: A3Graph) -> A3Graph:
        means = {key: np.asarray(value, dtype=np.float32) for key, value in self.means}
        scales = {key: np.asarray(value, dtype=np.float32) for key, value in self.scales}

        def apply(name: str, values: np.ndarray) -> np.ndarray:
            if values.shape[0] == 0:
                return values.astype(np.float32, copy=True)
            return ((values - means[name]) / scales[name]).astype(np.float32)

        return replace(
            graph,
            segment_features=apply("segment", graph.segment_features),
            robot_features=apply("robot", graph.robot_features),
            resource_features=apply("resource", graph.resource_features),
            edge_features=apply("edge", graph.edge_features),
            pair_features=apply("pair", graph.pair_features),
            normalizer_sha256=self.sha256,
        )


def fit_feature_vocabulary(
    instances: Sequence[AllocationInstance], *, split: str
) -> FeatureVocabulary:
    if split != "train":
        raise ValueError("feature vocabulary must be fitted on train only")
    if not instances:
        raise ValueError("cannot fit vocabulary on an empty training set")
    capabilities = {
        item
        for instance in instances
        for segment in instance.segments
        for item in segment.required_capabilities
    } | {
        item for instance in instances for robot in instance.robots for item in robot.capabilities
    }
    tools = {
        item
        for instance in instances
        for segment in instance.segments
        for item in ([segment.required_tool_id] if segment.required_tool_id else [])
    } | {item for instance in instances for robot in instance.robots for item in robot.tool_ids}
    models = {robot.kinematic_model_id for instance in instances for robot in instance.robots}
    payload = {
        "capabilities": [UNKNOWN_TOKEN] + sorted(capabilities),
        "tools": [UNKNOWN_TOKEN] + sorted(tools),
        "kinematic_models": [UNKNOWN_TOKEN] + sorted(models),
        "fit_split": split,
        "fit_instance_count": len(instances),
    }
    digest = _json_digest(payload)
    return FeatureVocabulary(
        tuple(payload["capabilities"]),
        tuple(payload["tools"]),
        tuple(payload["kinematic_models"]),
        split,
        len(instances),
        digest,
    )


def build_a3_graph(
    instance: AllocationInstance,
    context: OracleContext,
    vocabulary: FeatureVocabulary,
    *,
    split: str,
) -> A3Graph:
    if split not in {"train", "validation"}:
        raise ValueError(f"A3 development forbids split: {split}")
    segments = tuple(sorted(instance.segments, key=lambda item: item.id))
    robots = tuple(sorted(instance.robots, key=lambda item: item.id))
    resources = tuple(sorted(instance.resources, key=lambda item: item.id))
    segment_ids = tuple(item.id for item in segments)
    robot_ids = tuple(item.id for item in robots)
    resource_ids = tuple(item.id for item in resources)
    segment_features = np.asarray(
        [_segment_features(item, vocabulary) for item in segments], dtype=np.float32
    )
    robot_features = np.asarray(
        [_robot_features(item, vocabulary) for item in robots], dtype=np.float32
    )
    resource_width = 7
    resource_features = np.asarray(
        [_resource_features(item) for item in resources], dtype=np.float32
    ).reshape(len(resources), resource_width)

    mask = build_edge_mask(instance, context)
    pair_features = np.zeros(
        (len(segments), len(robots), len(PAIR_FEATURE_NAMES)), dtype=np.float32
    )
    allowed = np.zeros((len(segments), len(robots)), dtype=np.bool_)
    edges: list[tuple[int, int]] = []
    edge_types: list[int] = []
    edge_values: list[list[float]] = []
    segment_offset = 0
    robot_offset = len(segments)
    resource_offset = robot_offset + len(robots)
    relation = {name: index for index, name in enumerate(RELATION_NAMES)}
    for segment_index, segment in enumerate(segments):
        original_segment_index = mask.segment_ids.index(segment.id)
        for robot_index, robot in enumerate(robots):
            original_robot_index = mask.robot_ids.index(robot.id)
            estimate = mask.estimates[original_segment_index][original_robot_index]
            values = _pair_features(estimate)
            pair_features[segment_index, robot_index] = values
            allowed[segment_index, robot_index] = estimate.feasible
            segment_node = segment_offset + segment_index
            robot_node = robot_offset + robot_index
            edges.extend(((robot_node, segment_node), (segment_node, robot_node)))
            edge_types.extend(
                (relation["robot_to_segment"], relation["segment_to_robot"])
            )
            edge_values.extend((values, values))

    segment_lookup = {item.id: index for index, item in enumerate(segments)}
    precedence: dict[tuple[str, str], tuple[float, float]] = {}
    for segment in segments:
        for predecessor in segment.predecessor_ids:
            precedence[(predecessor, segment.id)] = (1.0, 0.0)
    parent_groups: dict[str, list[object]] = {}
    for segment in segments:
        parent_groups.setdefault(segment.parent_curve_id, []).append(segment)
    for group in parent_groups.values():
        ordered = sorted(group, key=lambda item: item.segment_index)
        for left, right in zip(ordered, ordered[1:]):
            declared, _ = precedence.get((left.id, right.id), (0.0, 0.0))
            precedence[(left.id, right.id)] = (declared, 1.0)
    for (left_id, right_id), flags in sorted(precedence.items()):
        left = segment_lookup[left_id]
        right = segment_lookup[right_id]
        values = [flags[0], flags[1]] + [0.0] * (len(PAIR_FEATURE_NAMES) - 2)
        edges.extend(((left, right), (right, left)))
        edge_types.extend(
            (
                relation["segment_precedes_segment"],
                relation["segment_follows_segment"],
            )
        )
        edge_values.extend((values, values))

    resource_lookup = {item.id: index for index, item in enumerate(resources)}
    for segment_index, segment in enumerate(segments):
        for resource_id in segment.shared_resource_ids:
            resource_index = resource_lookup[resource_id]
            resource_node = resource_offset + resource_index
            values = [1.0] + [0.0] * (len(PAIR_FEATURE_NAMES) - 1)
            edges.extend(((segment_index, resource_node), (resource_node, segment_index)))
            edge_types.extend(
                (relation["segment_uses_resource"], relation["resource_used_by_segment"])
            )
            edge_values.extend((values, values))

    edge_index = np.asarray(edges, dtype=np.int64).T.reshape(2, len(edges))
    edge_type = np.asarray(edge_types, dtype=np.int64)
    edge_features = np.asarray(edge_values, dtype=np.float32).reshape(
        len(edges), len(PAIR_FEATURE_NAMES)
    )
    return A3Graph(
        GRAPH_VERSION,
        instance.instance_id,
        split,
        segment_ids,
        robot_ids,
        resource_ids,
        segment_features,
        robot_features,
        resource_features,
        edge_index,
        edge_type,
        edge_features,
        pair_features,
        allowed,
        vocabulary.sha256,
    )


def build_teacher_target(graph: A3Graph, plan: AllocationPlan, teacher_sha256: str) -> GraphTeacherTarget:
    if plan.instance_id != graph.instance_id:
        raise ValueError("teacher plan and graph instance differ")
    scheduled = {item.segment_id: item for item in plan.schedule}
    if set(scheduled) != set(graph.segment_ids):
        raise ValueError("teacher plan does not cover graph segments")
    robot_lookup = {item: index for index, item in enumerate(graph.robot_ids)}
    return GraphTeacherTarget(
        graph.instance_id,
        np.asarray([robot_lookup[scheduled[item].robot_id] for item in graph.segment_ids], dtype=np.int64),
        np.asarray([scheduled[item].order_index for item in graph.segment_ids], dtype=np.int64),
        teacher_sha256,
    )


def fit_feature_normalizer(
    graphs: Sequence[A3Graph], *, split: str, epsilon: float = 1e-8
) -> FeatureNormalizer:
    if split != "train" or any(graph.split != "train" for graph in graphs):
        raise ValueError("feature normalizer must be fitted exclusively on train graphs")
    if not graphs or epsilon <= 0:
        raise ValueError("normalizer needs train graphs and positive epsilon")
    arrays = {
        "segment": np.concatenate([item.segment_features for item in graphs]),
        "robot": np.concatenate([item.robot_features for item in graphs]),
        "resource": _concatenate_nonempty([item.resource_features for item in graphs]),
        "edge": np.concatenate([item.edge_features for item in graphs]),
        "pair": np.concatenate([item.pair_features.reshape(-1, item.pair_features.shape[-1]) for item in graphs]),
    }
    means: dict[str, tuple[float, ...]] = {}
    scales: dict[str, tuple[float, ...]] = {}
    for name, values in arrays.items():
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale < epsilon, 1.0, scale)
        means[name] = tuple(float(item) for item in mean)
        scales[name] = tuple(float(item) for item in scale)
    payload = {
        "version": "a3-feature-normalizer-v1",
        "fit_split": split,
        "fit_graph_count": len(graphs),
        "means": means,
        "scales": scales,
        "epsilon": epsilon,
    }
    digest = _json_digest(payload)
    return FeatureNormalizer(
        str(payload["version"]),
        split,
        len(graphs),
        tuple(sorted(means.items())),
        tuple(sorted(scales.items())),
        epsilon,
        digest,
    )


def _segment_features(segment, vocabulary: FeatureVocabulary) -> list[float]:
    curve = extract_curve_features(segment)
    tortuosity = math.log1p(min(curve.tortuosity, 1e6))
    numeric = [
        curve.polyline_length_m,
        curve.chord_length_m,
        tortuosity,
        *curve.centroid_m,
        *curve.aabb_min_m,
        *curve.aabb_max_m,
        *curve.direction_unit,
        curve.mean_turn_angle_rad,
        curve.max_turn_angle_rad,
        segment.length_m,
        segment.process_duration_s,
        float(segment.priority),
        segment.time_window.start_s,
        segment.time_window.end_s,
        segment.time_window.end_s - segment.time_window.start_s,
        float(len(segment.predecessor_ids)),
        float(len(segment.shared_resource_ids)),
        float(segment.segment_index),
    ]
    directions = [float(segment.process_direction is item) for item in ProcessDirection]
    handoffs = [float(segment.handoff_policy is item) for item in HandoffPolicy]
    capabilities = _multi_hot(segment.required_capabilities, vocabulary.capabilities)
    tools = _multi_hot(
        () if segment.required_tool_id is None else (segment.required_tool_id,), vocabulary.tools
    )
    return numeric + directions + handoffs + capabilities + tools


def _robot_features(robot, vocabulary: FeatureVocabulary) -> list[float]:
    joints = np.asarray(robot.initial_joint_state_rad, dtype=float)
    numeric = [
        *robot.base_pose.position_m,
        *robot.base_pose.quaternion_xyzw,
        robot.availability.start_s,
        robot.availability.end_s,
        robot.availability.end_s - robot.availability.start_s,
        robot.nominal_cartesian_speed_m_s,
        float(len(joints)),
        float(joints.mean()) if len(joints) else 0.0,
        float(joints.std()) if len(joints) else 0.0,
        float(np.max(np.abs(joints))) if len(joints) else 0.0,
    ]
    return (
        numeric
        + _multi_hot(robot.capabilities, vocabulary.capabilities)
        + _multi_hot(robot.tool_ids, vocabulary.tools)
        + _multi_hot((robot.kinematic_model_id,), vocabulary.kinematic_models)
    )


def _resource_features(resource) -> list[float]:
    return [
        *[float(resource.resource_type is item) for item in ResourceType],
        float(resource.capacity),
        resource.availability.start_s,
        resource.availability.end_s,
        resource.availability.end_s - resource.availability.start_s,
    ]


def _pair_features(estimate) -> list[float]:
    reasons = set(estimate.reason_codes)
    return [
        float(estimate.feasible),
        estimate.travel_time_s,
        estimate.process_time_s,
        estimate.path_length_m,
        estimate.kinematic_risk,
        estimate.conflict_proxy,
        estimate.confidence,
        *[float(item.value in reasons) for item in EdgeReason],
    ]


def _multi_hot(values: Sequence[str], vocabulary: Sequence[str]) -> list[float]:
    result = [0.0] * len(vocabulary)
    lookup = {item: index for index, item in enumerate(vocabulary)}
    for value in values:
        result[lookup.get(value, 0)] = 1.0
    return result


def _concatenate_nonempty(arrays: Sequence[np.ndarray]) -> np.ndarray:
    nonempty = [item for item in arrays if item.shape[0]]
    if not nonempty:
        width = arrays[0].shape[1]
        return np.zeros((1, width), dtype=np.float32)
    return np.concatenate(nonempty)


def _json_digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
