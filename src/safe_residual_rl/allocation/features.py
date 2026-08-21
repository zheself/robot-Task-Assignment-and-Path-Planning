"""Deterministic analytical features for continuous process segments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .schema import ProcessSegment


@dataclass(frozen=True)
class CurveFeatures:
    polyline_length_m: float
    chord_length_m: float
    tortuosity: float
    centroid_m: tuple[float, float, float]
    aabb_min_m: tuple[float, float, float]
    aabb_max_m: tuple[float, float, float]
    direction_unit: tuple[float, float, float]
    mean_turn_angle_rad: float
    max_turn_angle_rad: float


def extract_curve_features(segment: ProcessSegment) -> CurveFeatures:
    points = segment.sampled_curve_m
    edge_vectors = [_subtract(b, a) for a, b in zip(points, points[1:])]
    edge_lengths = [_norm(vector) for vector in edge_vectors]
    polyline_length = sum(edge_lengths)
    chord = _subtract(points[-1], points[0])
    chord_length = _norm(chord)
    direction = _unit(chord)
    angles: list[float] = []
    for left, right in zip(edge_vectors, edge_vectors[1:]):
        left_norm = _norm(left)
        right_norm = _norm(right)
        if left_norm > 0.0 and right_norm > 0.0:
            cosine = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
            angles.append(math.acos(max(-1.0, min(1.0, cosine))))
    centroid = tuple(sum(point[axis] for point in points) / len(points) for axis in range(3))
    aabb_min = tuple(min(point[axis] for point in points) for axis in range(3))
    aabb_max = tuple(max(point[axis] for point in points) for axis in range(3))
    if chord_length > 1e-12:
        tortuosity = polyline_length / chord_length
    elif polyline_length <= 1e-12:
        tortuosity = 1.0
    else:
        tortuosity = math.inf
    return CurveFeatures(
        polyline_length_m=polyline_length,
        chord_length_m=chord_length,
        tortuosity=tortuosity,
        centroid_m=centroid,  # type: ignore[arg-type]
        aabb_min_m=aabb_min,  # type: ignore[arg-type]
        aabb_max_m=aabb_max,  # type: ignore[arg-type]
        direction_unit=direction,
        mean_turn_angle_rad=sum(angles) / len(angles) if angles else 0.0,
        max_turn_angle_rad=max(angles, default=0.0),
    )


def point_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return _norm(_subtract(a, b))


def point_to_segment_distance(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    edge = _subtract(end, start)
    edge_squared = sum(value * value for value in edge)
    if edge_squared <= 1e-24:
        return point_distance(point, start)
    offset = _subtract(point, start)
    fraction = sum(a * b for a, b in zip(offset, edge)) / edge_squared
    fraction = max(0.0, min(1.0, fraction))
    closest = tuple(start[i] + fraction * edge[i] for i in range(3))
    return point_distance(point, closest)


def _subtract(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (float(a[0] - b[0]), float(a[1] - b[1]), float(a[2] - b[2]))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _unit(vector: Sequence[float]) -> tuple[float, float, float]:
    norm = _norm(vector)
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0)
    return tuple(value / norm for value in vector)  # type: ignore[return-value]
