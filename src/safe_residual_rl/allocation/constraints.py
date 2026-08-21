"""A0 constraint-dictionary loader and structural validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CONSTRAINT_DICTIONARY_VERSION = "allocation-constraints-v1"
REQUIRED_HARD_CONSTRAINT_IDS = frozenset(
    {
        "unique_assignment",
        "tool_capability",
        "reachability_proxy",
        "parent_segment_order",
        "precedence",
        "handoff_policy",
        "process_direction",
        "time_window",
        "robot_non_overlap",
        "resource_capacity",
        "no_go_proxy",
    }
)
REQUIRED_OBJECTIVE_IDS = frozenset(
    {"makespan", "load_variance", "travel_setup_time", "priority_tardiness"}
)


@dataclass(frozen=True)
class ConstraintDefinition:
    id: str
    severity: str
    scope: str
    stage: str
    description: str
    evidence_boundary: str


@dataclass(frozen=True)
class ObjectiveDefinition:
    id: str
    sense: str
    unit: str
    description: str


@dataclass(frozen=True)
class ConstraintDictionary:
    version: str
    constraints: tuple[ConstraintDefinition, ...]
    objectives: tuple[ObjectiveDefinition, ...]


def constraint_dictionary_from_dict(data: Mapping[str, Any]) -> ConstraintDictionary:
    dictionary = ConstraintDictionary(
        version=str(data["version"]),
        constraints=tuple(ConstraintDefinition(**x) for x in data["constraints"]),
        objectives=tuple(ObjectiveDefinition(**x) for x in data["objectives"]),
    )
    validate_constraint_dictionary(dictionary)
    return dictionary


def validate_constraint_dictionary(dictionary: ConstraintDictionary) -> None:
    if dictionary.version != CONSTRAINT_DICTIONARY_VERSION:
        raise ValueError(f"expected constraint version {CONSTRAINT_DICTIONARY_VERSION}")
    constraint_ids = [x.id for x in dictionary.constraints]
    objective_ids = [x.id for x in dictionary.objectives]
    if len(constraint_ids) != len(set(constraint_ids)):
        raise ValueError("constraint IDs must be unique")
    if len(objective_ids) != len(set(objective_ids)):
        raise ValueError("objective IDs must be unique")
    missing_constraints = REQUIRED_HARD_CONSTRAINT_IDS - set(constraint_ids)
    missing_objectives = REQUIRED_OBJECTIVE_IDS - set(objective_ids)
    if missing_constraints:
        raise ValueError(f"missing required constraints: {sorted(missing_constraints)}")
    if missing_objectives:
        raise ValueError(f"missing required objectives: {sorted(missing_objectives)}")
    for item in dictionary.constraints:
        if item.severity not in {"hard", "soft"}:
            raise ValueError(f"invalid severity for {item.id}: {item.severity}")
        if not all((item.scope, item.stage, item.description, item.evidence_boundary)):
            raise ValueError(f"constraint {item.id} has empty semantics")
    for item in dictionary.objectives:
        if item.sense != "minimize":
            raise ValueError(f"objective {item.id} must explicitly minimize")
        if not item.unit or not item.description:
            raise ValueError(f"objective {item.id} has empty semantics")


def load_constraint_dictionary(path: str | Path) -> ConstraintDictionary:
    with Path(path).open("r", encoding="utf-8") as handle:
        return constraint_dictionary_from_dict(json.load(handle))
