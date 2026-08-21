from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from safe_residual_rl.allocation.constraints import (
    REQUIRED_HARD_CONSTRAINT_IDS,
    REQUIRED_OBJECTIVE_IDS,
    load_constraint_dictionary,
)
from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.schema import (
    SchemaValidationError,
    allocation_instance_from_dict,
    allocation_plan_from_dict,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "data" / "fixtures" / "allocation"


def fixture_paths() -> list[Path]:
    return sorted(FIXTURE_DIR.glob("*.json"))


def test_has_at_least_ten_auditable_fixtures() -> None:
    paths = fixture_paths()
    assert len(paths) >= 10
    fixture_ids = [load_auditable_fixture(path)["fixture_id"] for path in paths]
    assert len(fixture_ids) == len(set(fixture_ids))


@pytest.mark.parametrize("path", fixture_paths(), ids=lambda path: path.stem)
def test_fixture_matches_declared_validity_and_issue_codes(path: Path) -> None:
    payload = load_auditable_fixture(path)
    expected = payload["expected"]
    if expected["valid"]:
        instance = allocation_instance_from_dict(payload["instance"])
        assert instance.instance_id
    else:
        with pytest.raises(SchemaValidationError) as captured:
            allocation_instance_from_dict(payload["instance"])
        observed = {issue.code for issue in captured.value.issues}
        assert set(expected["issue_codes"]).issubset(observed)


def test_valid_instance_round_trip_is_stable() -> None:
    payload = load_auditable_fixture(FIXTURE_DIR / "02_valid_same_robot_segments.json")
    instance = allocation_instance_from_dict(payload["instance"])
    restored = allocation_instance_from_dict(instance.to_dict())
    assert restored == instance


def test_fixture_inheritance_does_not_mutate_base() -> None:
    base = load_auditable_fixture(FIXTURE_DIR / "01_valid_minimal.json")
    derived = load_auditable_fixture(FIXTURE_DIR / "05_valid_priority_window.json")
    assert base["instance"]["segments"][0]["priority"] == 1
    assert derived["instance"]["segments"][0]["priority"] == 10


def test_constraint_dictionary_is_complete_and_versioned() -> None:
    dictionary = load_constraint_dictionary(ROOT / "configs" / "allocation" / "constraints_v1.json")
    constraint_ids = {x.id for x in dictionary.constraints}
    objective_ids = {x.id for x in dictionary.objectives}
    assert REQUIRED_HARD_CONSTRAINT_IDS <= constraint_ids
    assert REQUIRED_OBJECTIVE_IDS <= objective_ids
    assert all(x.evidence_boundary for x in dictionary.constraints)


def test_machine_readable_schema_agrees_with_runtime_schema() -> None:
    schema = json.loads((ROOT / "configs" / "allocation" / "schema_v1.json").read_text())
    assert schema["schema_version"] == "allocation-instance-v1"
    assert schema["plan_schema_version"] == "allocation-plan-v1"
    assert schema["units"]["position"] == "m"
    assert schema["units"]["angle"] == "rad"
    assert set(schema["evidence_labels"]) == {"SYNTHETIC", "SIM_GEOMETRIC"}
    assert "not_splittable" in schema["enums"]["handoff_policy"]


def test_allocation_plan_contract_round_trip_and_coverage() -> None:
    payload = load_auditable_fixture(FIXTURE_DIR / "02_valid_same_robot_segments.json")
    instance = allocation_instance_from_dict(payload["instance"])
    plan_data = {
        "schema_version": "allocation-plan-v1",
        "instance_id": instance.instance_id,
        "method_id": "a0-fixture",
        "schedule": [
            {"segment_id": "seg-0", "robot_id": "robot-0", "order_index": 0, "planned_start_s": 0.0, "planned_end_s": 2.0},
            {"segment_id": "seg-1", "robot_id": "robot-0", "order_index": 1, "planned_start_s": 2.0, "planned_end_s": 4.0},
        ],
        "solver_status": "fixture",
        "objective_terms": {"makespan": 4.0},
        "diagnostics": [],
    }
    plan = allocation_plan_from_dict(plan_data, instance)
    assert allocation_plan_from_dict(plan.to_dict(), instance) == plan


def test_frozen_a0_manifest_hashes_match_files() -> None:
    manifest_path = ROOT / "data" / "manifests" / "allocation" / "a0_schema_fixture_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "frozen_a0"
    assert manifest["fixture_count"] == len(fixture_paths()) == 12
    for relative_path, expected_hash in manifest["files_sha256"].items():
        actual_hash = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash, relative_path
