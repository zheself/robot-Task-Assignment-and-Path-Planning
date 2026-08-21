"""Deterministic A2 generation of continuous-process geometric instances."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import (
    SCHEMA_VERSION,
    AllocationInstance,
    EvidenceLabel,
    HandoffPolicy,
    Pose,
    ProcessDirection,
    ProcessSegment,
    ResourceSpec,
    ResourceType,
    RobotSpec,
    TimeWindow,
    validate_instance,
)

GENERATOR_VERSION = "a2-continuous-generator-v1"


@dataclass(frozen=True)
class SplitGenerationSpec:
    group_count: int
    robot_count: tuple[int, int]
    segment_count: tuple[int, int]
    max_segments_per_curve: int
    precedence_probability: float
    shared_resource_probability: float
    tight_window_probability: float


@dataclass(frozen=True)
class BenchmarkConfig:
    version: str
    manifest_version: str
    master_seed: int
    evidence_label: EvidenceLabel
    coordinate_frame: str
    variants_per_group: int
    geometry: Mapping[str, Any]
    splits: tuple[tuple[str, SplitGenerationSpec], ...]
    objective_weights: tuple[tuple[str, float], ...]
    baseline_protocol: Mapping[str, Any]
    boundaries: tuple[str, ...]

    def split(self, name: str) -> SplitGenerationSpec:
        return dict(self.splits)[name]


@dataclass(frozen=True)
class GeneratedInstance:
    split: str
    family: str
    task_group_id: str
    variant_index: int
    seed: int
    instance: AllocationInstance


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    splits = tuple(
        (
            str(name),
            SplitGenerationSpec(
                group_count=int(raw["group_count"]),
                robot_count=_pair(raw["robot_count"]),
                segment_count=_pair(raw["segment_count"]),
                max_segments_per_curve=int(raw["max_segments_per_curve"]),
                precedence_probability=float(raw["precedence_probability"]),
                shared_resource_probability=float(raw["shared_resource_probability"]),
                tight_window_probability=float(raw["tight_window_probability"]),
            ),
        )
        for name, raw in data["splits"].items()
    )
    config = BenchmarkConfig(
        version=str(data["version"]),
        manifest_version=str(data["manifest_version"]),
        master_seed=int(data["master_seed"]),
        evidence_label=EvidenceLabel(data["evidence_label"]),
        coordinate_frame=str(data["coordinate_frame"]),
        variants_per_group=int(data["variants_per_group"]),
        geometry=dict(data["geometry"]),
        splits=splits,
        objective_weights=tuple(sorted((str(key), float(value)) for key, value in data["objective_weights"].items())),
        baseline_protocol=dict(data["baseline_protocol"]),
        boundaries=tuple(str(item) for item in data["boundaries"]),
    )
    _validate_config(config)
    return config


def generate_benchmark(config: BenchmarkConfig) -> tuple[GeneratedInstance, ...]:
    generated: list[GeneratedInstance] = []
    for split, spec in config.splits:
        variants = 1 if split == "stress" else config.variants_per_group
        for group_index in range(spec.group_count):
            group_id = f"{split}-group-{group_index:03d}"
            for variant_index in range(variants):
                generated.append(generate_instance(config, split, group_id, group_index, variant_index))
    return tuple(generated)


def generate_instance(
    config: BenchmarkConfig,
    split: str,
    task_group_id: str,
    group_index: int,
    variant_index: int,
) -> GeneratedInstance:
    spec = config.split(split)
    geometry_seed = stable_seed(config.master_seed, split, group_index, "geometry")
    variant_seed = stable_seed(config.master_seed, split, group_index, variant_index, "constraints")
    geometry_rng = np.random.default_rng(geometry_seed)
    rng = np.random.default_rng(variant_seed)
    robot_count = int(geometry_rng.integers(spec.robot_count[0], spec.robot_count[1] + 1))
    target_segments = int(geometry_rng.integers(spec.segment_count[0], spec.segment_count[1] + 1))
    robot_radius = float(config.geometry["robot_base_radius_m"])
    horizon = float(config.geometry["planning_horizon_s"])
    robots = _robots(robot_count, robot_radius, horizon, geometry_rng)
    resources = (
        ResourceSpec("shared-zone-0", ResourceType.SHARED_ZONE, 1, TimeWindow(0.0, horizon)),
        ResourceSpec("fixture-0", ResourceType.FIXTURE, 2, TimeWindow(0.0, horizon)),
    )
    segments = _segments(
        target_segments,
        spec,
        task_group_id,
        config.geometry,
        geometry_rng,
        rng,
        horizon,
    )
    instance = AllocationInstance(
        schema_version=SCHEMA_VERSION,
        evidence_label=config.evidence_label,
        instance_id=f"{task_group_id}-v{variant_index:02d}",
        workpiece_id=f"workpiece-{task_group_id}",
        layout_id=f"layout-{task_group_id}",
        coordinate_frame=config.coordinate_frame,
        segments=segments,
        robots=robots,
        resources=resources,
    )
    issues = validate_instance(instance)
    if issues:
        raise ValueError("generated invalid instance: " + ";".join(item.code for item in issues))
    family = {
        "train": "in_domain_train",
        "validation": "in_domain_validation",
        "frozen_test": "unseen_layout_constraint",
        "stress": "scale_resource_stress",
    }[split]
    return GeneratedInstance(split, family, task_group_id, variant_index, variant_seed, instance)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(item) for item in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def canonical_instance_bytes(instance: AllocationInstance) -> bytes:
    return json.dumps(instance.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _robots(count: int, radius: float, horizon: float, rng: np.random.Generator) -> tuple[RobotSpec, ...]:
    phase = float(rng.uniform(-0.15, 0.15))
    robots = []
    for index in range(count):
        angle = phase + 2.0 * math.pi * index / count
        capabilities = ("continuous-process", "precision") if index % 2 == 0 else ("continuous-process",)
        tools = ("tool-a", "tool-b") if index == 0 or index % 3 == 0 else ("tool-a",)
        robots.append(
            RobotSpec(
                id=f"robot-{index:02d}",
                base_pose=_pose((radius * math.cos(angle), radius * math.sin(angle), 0.0)),
                capabilities=capabilities,
                tool_ids=tools,
                availability=TimeWindow(0.0, horizon),
                kinematic_model_id="abstract-arm-v1",
                initial_joint_state_rad=(0.0,) * 6,
                nominal_cartesian_speed_m_s=float(rng.uniform(0.08, 0.14)),
            )
        )
    return tuple(robots)


def _segments(
    target_count: int,
    spec: SplitGenerationSpec,
    group_id: str,
    geometry: Mapping[str, Any],
    geometry_rng: np.random.Generator,
    rng: np.random.Generator,
    horizon: float,
) -> tuple[ProcessSegment, ...]:
    counts: list[int] = []
    remaining = target_count
    while remaining:
        count = min(remaining, int(geometry_rng.integers(1, spec.max_segments_per_curve + 1)))
        counts.append(count)
        remaining -= count
    result: list[ProcessSegment] = []
    previous_curve_last: str | None = None
    curve_types = tuple(str(item) for item in geometry["curve_types"])
    samples = int(geometry["samples_per_curve"])
    process_speed = float(geometry["nominal_process_speed_m_s"])
    workpiece_radius = float(geometry["workpiece_radius_m"])
    for curve_index, part_count in enumerate(counts):
        curve_type = curve_types[curve_index % len(curve_types)]
        points = _curve_points(curve_type, curve_index, len(counts), samples, workpiece_radius, geometry_rng)
        slices = _split_curve(points, part_count)
        handoff = HandoffPolicy.NOT_SPLITTABLE if part_count == 1 and curve_type == "closed_loop" else HandoffPolicy(rng.choice(["free", "same_robot", "explicit_boundary"]))
        parent_id = f"{group_id}-curve-{curve_index:03d}"
        curve_segment_ids = [f"{parent_id}-seg-{index:02d}" for index in range(len(slices))]
        cross_predecessor = previous_curve_last if previous_curve_last and rng.random() < spec.precedence_probability else None
        release = float(rng.uniform(0.0, 10.0))
        tight = rng.random() < spec.tight_window_probability
        curve_direction = ProcessDirection(str(rng.choice(["forward", "reverse", "either"]))) if part_count == 1 else ProcessDirection.FORWARD
        for segment_index, (segment_id, sampled) in enumerate(zip(curve_segment_ids, slices)):
            length = _polyline_length(sampled)
            duration = max(0.25, length / process_speed)
            predecessors = (cross_predecessor,) if segment_index == 0 and cross_predecessor else ()
            required_precision = rng.random() < 0.25
            tool_id = "tool-b" if rng.random() < 0.15 else "tool-a"
            resource_ids: list[str] = []
            if rng.random() < spec.shared_resource_probability:
                resource_ids.append("shared-zone-0")
            if rng.random() < 0.15:
                resource_ids.append("fixture-0")
            due = min(horizon, release + (duration * len(slices) * 4.0 if tight else horizon - release))
            result.append(
                ProcessSegment(
                    id=segment_id,
                    parent_curve_id=parent_id,
                    segment_index=segment_index,
                    sampled_curve_m=tuple(sampled),
                    start_pose=_pose(sampled[0]),
                    end_pose=_pose(sampled[-1]),
                    process_direction=curve_direction,
                    length_m=length,
                    process_duration_s=duration,
                    required_capabilities=("continuous-process", "precision") if required_precision else ("continuous-process",),
                    required_tool_id=tool_id,
                    priority=int(rng.integers(1, 6)),
                    time_window=TimeWindow(release, due),
                    predecessor_ids=predecessors,
                    handoff_policy=handoff,
                    shared_resource_ids=tuple(resource_ids),
                )
            )
        previous_curve_last = curve_segment_ids[-1]
    return tuple(result)


def _curve_points(
    curve_type: str,
    index: int,
    count: int,
    samples: int,
    radius: float,
    rng: np.random.Generator,
) -> tuple[tuple[float, float, float], ...]:
    angle = 2.0 * math.pi * (index + 0.3) / max(count, 1)
    center = np.array([0.62 * radius * math.cos(angle), 0.62 * radius * math.sin(angle), 0.22 + 0.03 * math.sin(angle)])
    tangent = np.array([-math.sin(angle), math.cos(angle), 0.0])
    radial = np.array([math.cos(angle), math.sin(angle), 0.0])
    t = np.linspace(0.0, 1.0, samples)
    length = float(rng.uniform(0.10, 0.22))
    if curve_type == "line":
        values = center + (t[:, None] - 0.5) * length * tangent
    elif curve_type == "arc":
        theta = (t - 0.5) * float(rng.uniform(0.8, 1.7))
        arc_radius = float(rng.uniform(0.06, 0.11))
        values = center + arc_radius * ((np.cos(theta) - 1.0)[:, None] * radial + np.sin(theta)[:, None] * tangent)
    elif curve_type == "bspline":
        controls = np.stack((center - 0.7 * length * tangent, center - 0.2 * length * tangent + 0.04 * radial, center + 0.2 * length * tangent - 0.04 * radial, center + 0.7 * length * tangent))
        weights = np.stack(((1-t)**3, 3*t**3-6*t**2+4, -3*t**3+3*t**2+3*t+1, t**3), axis=1) / 6.0
        values = weights @ controls
    elif curve_type == "closed_loop":
        theta = 2.0 * math.pi * t
        loop_radius = float(rng.uniform(0.035, 0.065))
        values = center + loop_radius * (np.cos(theta)[:, None] * radial + np.sin(theta)[:, None] * tangent)
        values[-1] = values[0]
    else:
        raise ValueError(f"unsupported curve type: {curve_type}")
    return tuple(tuple(float(x) for x in row) for row in values)


def _split_curve(points: Sequence[tuple[float, float, float]], count: int) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    boundaries = np.linspace(0, len(points) - 1, count + 1).round().astype(int)
    return tuple(tuple(points[left : right + 1]) for left, right in zip(boundaries[:-1], boundaries[1:]))


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    return sum(math.dist(left, right) for left, right in zip(points, points[1:]))


def _pose(position: Sequence[float]) -> Pose:
    return Pose(tuple(float(item) for item in position), (0.0, 0.0, 0.0, 1.0))  # type: ignore[arg-type]


def _pair(value: Sequence[Any]) -> tuple[int, int]:
    pair = tuple(int(item) for item in value)
    if len(pair) != 2:
        raise ValueError("range must contain two values")
    return pair  # type: ignore[return-value]


def _validate_config(config: BenchmarkConfig) -> None:
    if config.version != "a2-geometric-benchmark-v1" or config.manifest_version != "a2-split-manifest-v1":
        raise ValueError("unsupported A2 config version")
    if config.evidence_label is not EvidenceLabel.SIM_GEOMETRIC or config.variants_per_group < 1:
        raise ValueError("A2 benchmark must be SIM_GEOMETRIC with variants")
    expected = {"train", "validation", "frozen_test", "stress"}
    if set(dict(config.splits)) != expected:
        raise ValueError("A2 requires train/validation/frozen_test/stress")
    for _, spec in config.splits:
        probabilities = (spec.precedence_probability, spec.shared_resource_probability, spec.tight_window_probability)
        if spec.group_count < 1 or spec.robot_count[0] < 2 or spec.segment_count[0] < 1 or not all(0 <= item <= 1 for item in probabilities):
            raise ValueError("invalid A2 split generation spec")
