from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from safe_residual_rl.allocation import (
    construct_feasible_witness,
    load_oracle_context,
    solve_assignment_beam_sequence,
    verify_constructive_witness,
    verify_plan,
)
from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.schema import TimeWindow, allocation_instance_from_dict

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def context():
    return load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")


@pytest.fixture
def impossible_window_instance():
    raw = load_auditable_fixture(
        ROOT / "data/fixtures/allocation/05_valid_priority_window.json"
    )
    instance = allocation_instance_from_dict(raw["instance"])
    segment = instance.segments[0]
    impossible = replace(
        segment,
        time_window=TimeWindow(0.0, 0.5 * segment.process_duration_s),
    )
    return replace(instance, instance_id="witness-recalibration", segments=(impossible,))


def test_constructive_witness_recalibrates_windows_and_verifies(
    impossible_window_instance, context
) -> None:
    witness = construct_feasible_witness(impossible_window_instance, context)
    assert not verify_constructive_witness(witness, context)
    assert verify_plan(witness.instance, witness.plan, context).feasible
    assert witness.instance.segments[0].sampled_curve_m == impossible_window_instance.segments[0].sampled_curve_m
    assert witness.instance.segments[0].required_capabilities == impossible_window_instance.segments[0].required_capabilities
    assert witness.instance.segments[0].time_window != impossible_window_instance.segments[0].time_window


def test_constructive_witness_is_deterministic(impossible_window_instance, context) -> None:
    first = construct_feasible_witness(impossible_window_instance, context)
    second = construct_feasible_witness(impossible_window_instance, context)
    assert first.instance == second.instance
    assert first.plan.schedule == second.plan.schedule
    assert first.witness_sha256 == second.witness_sha256


def test_constructive_witness_hash_detects_tampering(impossible_window_instance, context) -> None:
    witness = construct_feasible_witness(impossible_window_instance, context)
    segment = witness.instance.segments[0]
    tampered = replace(
        witness,
        instance=replace(
            witness.instance,
            segments=(
                replace(
                    segment,
                    time_window=TimeWindow(
                        segment.time_window.start_s,
                        segment.time_window.end_s + 1.0,
                    ),
                ),
            ),
        ),
    )
    assert "WITNESS_HASH_MISMATCH" in verify_constructive_witness(tampered, context)


def test_assignment_beam_returns_deterministic_verified_candidate(
    impossible_window_instance, context
) -> None:
    witness = construct_feasible_witness(impossible_window_instance, context)
    first = solve_assignment_beam_sequence(
        witness.instance,
        context,
        assignment_beam_width=4,
        sequence_beam_width=4,
        sequence_node_limit=100,
    )
    second = solve_assignment_beam_sequence(
        witness.instance,
        context,
        assignment_beam_width=4,
        sequence_beam_width=4,
        sequence_node_limit=100,
    )
    assert first.status == "feasible"
    assert first.plan is not None
    assert first.plan.schedule == second.plan.schedule
    assert verify_plan(witness.instance, first.plan, context).feasible
    assert "ASSIGNMENT_BEAM_THEN_SEQUENCE_BEAM" in first.diagnostics
