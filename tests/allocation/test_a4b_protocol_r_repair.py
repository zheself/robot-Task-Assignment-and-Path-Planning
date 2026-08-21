from __future__ import annotations

from pathlib import Path

import pytest

from safe_residual_rl.allocation.fixtures import load_auditable_fixture
from safe_residual_rl.allocation.oracle import load_oracle_context
from safe_residual_rl.allocation.repair import InitializerState
from safe_residual_rl.allocation.schema import allocation_instance_from_dict
from safe_residual_rl.allocation.search.alns_protocol_r import (
    run_search_protocol_r,
    transition_signature,
)
from safe_residual_rl.allocation.search.alns_v2 import AlnsV2Config, run_search_v2
from safe_residual_rl.allocation.search.anytime import (
    InitializerOutcome,
    InitializerProvenance,
    build_hybrid_load_balanced_initializer,
)
from safe_residual_rl.allocation.search.diagnostics import analyze_state
from safe_residual_rl.allocation.search.prepared_repair import (
    analyze_state_prepared,
    prepare_repair_problem,
)
from safe_residual_rl.allocation.search.repair_protocol_r import (
    RepairTraceCache,
    parity_report,
    repair_destroyed_state_protocol_r,
    repair_destroyed_state_reference_audited,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "data/fixtures/allocation"
WEIGHTS = {
    "makespan": 1.0,
    "load_variance": 0.05,
    "travel_setup_time": 0.1,
    "priority_tardiness": 1.0,
}


@pytest.fixture(scope="module")
def context():
    return load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")


def fixture(name):
    return allocation_instance_from_dict(load_auditable_fixture(FIXTURES / name)["instance"])


@pytest.mark.parametrize(
    "name",
    [
        "01_valid_minimal.json",
        "03_valid_explicit_boundary.json",
        "04_valid_shared_zone.json",
    ],
)
def test_prepared_diagnostic_is_bitwise_reference_equivalent(context, name):
    instance = fixture(name)
    initial = build_hybrid_load_balanced_initializer(instance, context, WEIGHTS)
    prepared = prepare_repair_problem(instance, context)
    assert analyze_state(instance, context, initial.state) == analyze_state_prepared(
        instance, prepared, initial.state
    )


@pytest.mark.parametrize(
    "name",
    [
        "01_valid_minimal.json",
        "03_valid_explicit_boundary.json",
        "04_valid_shared_zone.json",
    ],
)
def test_candidate_order_vector_state_plan_and_verifier_parity(context, name):
    instance = fixture(name)
    initial = build_hybrid_load_balanced_initializer(instance, context, WEIGHTS)
    prepared = prepare_repair_problem(instance, context)
    reference = repair_destroyed_state_reference_audited(
        instance,
        context,
        initial.state,
        (0,),
        weights=WEIGHTS,
        candidate_evaluation_budget=256,
        prepared=prepared,
    )
    accelerated = repair_destroyed_state_protocol_r(
        instance,
        context,
        initial.state,
        (0,),
        weights=WEIGHTS,
        candidate_evaluation_budget=256,
        prepared=prepared,
    )
    report = parity_report(reference, accelerated)
    assert report["passed"] and all(report["checks"].values())


def test_atomic_destroy_rejects_duplicate_and_out_of_range(context):
    instance = fixture("03_valid_explicit_boundary.json")
    initial = build_hybrid_load_balanced_initializer(instance, context, WEIGHTS)
    for destroyed in ((0, 0), (-1,), (99,)):
        with pytest.raises(ValueError, match="duplicate or non-atomic"):
            repair_destroyed_state_protocol_r(
                instance,
                context,
                initial.state,
                destroyed,
                weights=WEIGHTS,
                candidate_evaluation_budget=256,
            )


def test_trace_local_cache_does_not_change_selected_state(context):
    instance = fixture("03_valid_explicit_boundary.json")
    initial = build_hybrid_load_balanced_initializer(instance, context, WEIGHTS)
    prepared = prepare_repair_problem(instance, context)
    cache = RepairTraceCache()
    first = repair_destroyed_state_protocol_r(
        instance,
        context,
        initial.state,
        (0,),
        weights=WEIGHTS,
        candidate_evaluation_budget=256,
        prepared=prepared,
        cache=cache,
    )
    second = repair_destroyed_state_protocol_r(
        instance,
        context,
        initial.state,
        (0,),
        weights=WEIGHTS,
        candidate_evaluation_budget=256,
        prepared=prepared,
        cache=cache,
    )
    assert second.cache_hits > 0
    assert first.selected_state_sha256 == second.selected_state_sha256
    assert first.candidate_sequence_sha256 == second.candidate_sequence_sha256


def test_candidate_cap_and_fallback_flags_match_reference(context):
    instance = fixture("03_valid_explicit_boundary.json")
    initial = build_hybrid_load_balanced_initializer(instance, context, WEIGHTS)
    prepared = prepare_repair_problem(instance, context)
    reference = repair_destroyed_state_reference_audited(
        instance,
        context,
        initial.state,
        (0, 1),
        weights=WEIGHTS,
        candidate_evaluation_budget=1,
        prepared=prepared,
    )
    accelerated = repair_destroyed_state_protocol_r(
        instance,
        context,
        initial.state,
        (0, 1),
        weights=WEIGHTS,
        candidate_evaluation_budget=1,
        prepared=prepared,
    )
    assert reference.budget_exhausted and accelerated.budget_exhausted
    assert parity_report(reference, accelerated)["passed"]


def _config(iterations=5):
    return AlnsV2Config(
        protocol_id="protocol-r-fixture",
        budget_mode="fixed_iterations",
        iterations=iterations,
        end_to_end_time_s=None,
        safety_watchdog_s=1800.0,
        destroy_ratios=(0.1, 0.25, 0.4),
        repair_candidate_evaluation_budget=256,
        random_seed=3253,
        objective_weights=WEIGHTS,
        update_scheme="segmented",
        segment_length=2,
    )


def _online_config(iterations=5):
    return AlnsV2Config(**{**_config(iterations).__dict__, "update_scheme": "online"})


@pytest.mark.parametrize("mode", ["random_lns", "handcrafted_round_robin", "adaptive_alns"])
def test_fixed_iteration_transition_signature_matches_v2(context, mode):
    instance = fixture("03_valid_explicit_boundary.json")
    reference = run_search_v2(
        instance,
        context,
        build_hybrid_load_balanced_initializer(instance, context, WEIGHTS),
        _config(),
        mode=mode,
        task_group_id="fixture-group",
        difficulty="fixture",
        split="train",
    )
    accelerated = run_search_protocol_r(
        instance,
        context,
        build_hybrid_load_balanced_initializer(instance, context, WEIGHTS),
        _config(),
        mode=mode,
        task_group_id="fixture-group",
        difficulty="fixture",
        split="train",
    )
    assert transition_signature(reference.trace) == transition_signature(
        accelerated.trace
    )
    assert accelerated.trace["iterations_completed"] == 5
    assert accelerated.trace["fixed_iteration_complete"]


def test_online_alns_transition_signature_matches_v2(context):
    instance = fixture("03_valid_explicit_boundary.json")
    config = _online_config()
    reference = run_search_v2(
        instance,
        context,
        build_hybrid_load_balanced_initializer(instance, context, WEIGHTS),
        config,
        mode="adaptive_alns",
        task_group_id="fixture-group",
        difficulty="fixture",
        split="train",
    )
    accelerated = run_search_protocol_r(
        instance,
        context,
        build_hybrid_load_balanced_initializer(instance, context, WEIGHTS),
        config,
        mode="adaptive_alns",
        task_group_id="fixture-group",
        difficulty="fixture",
        split="train",
    )
    assert transition_signature(reference.trace) == transition_signature(
        accelerated.trace
    )


def test_fixed_time_counts_preparation_and_preserves_pre_cutoff_initializer(context):
    instance = fixture("01_valid_minimal.json")
    real = build_hybrid_load_balanced_initializer(instance, context, WEIGHTS)
    provenance = InitializerProvenance(
        requested_initializer="hybrid_load_balanced",
        actual_initializer="hybrid_load_balanced",
        solver_status="fixture",
        has_true_incumbent=True,
        fallback_used=False,
        fallback_reason=None,
        initializer_plan_hash=real.provenance.initializer_plan_hash,
        verifier_feasible=True,
        verifier_failure_reason=None,
        start_monotonic_ns=0,
        completion_monotonic_ns=100_000_000,
        completion_elapsed_s=0.1,
    )

    class Clock:
        value = 100_000_000

        def __call__(self):
            self.value += 100_000_000
            return self.value

    config = AlnsV2Config(
        **{
            **_config(iterations=30).__dict__,
            "budget_mode": "fixed_time",
            "end_to_end_time_s": 0.5,
        }
    )
    outcome = run_search_protocol_r(
        instance,
        context,
        InitializerOutcome(real.state, provenance),
        config,
        mode="random_lns",
        task_group_id="fixture-group",
        difficulty="fixture",
        split="train",
        clock=Clock(),
    )
    assert outcome.trace["termination_reason"] == "end_to_end_time_budget"
    assert outcome.trace["iterations_completed"] == 0
    assert outcome.trace["preparation_runtime_s"] > 0
    assert len(outcome.trace["incumbents"]) == 1
    assert outcome.trace["incumbents"][0]["elapsed_s"] == pytest.approx(0.1)


def test_prepared_problem_rejects_cross_instance_use(context):
    left = fixture("01_valid_minimal.json")
    right = fixture("03_valid_explicit_boundary.json")
    prepared = prepare_repair_problem(left, context)
    state = InitializerState((None,) * len(right.segments), ((right.robots[0].id, ()),))
    with pytest.raises(ValueError, match="another instance"):
        analyze_state_prepared(right, prepared, state)


def test_prepared_hash_supports_incompatible_robot_edges(context):
    config_path = ROOT / "configs/allocation/a4b_protocol_r_freeze_candidate_v1.json"
    from safe_residual_rl.allocation.search.data_protocol_r import (
        build_regular_variant,
        load_protocol_r_config,
    )

    config = load_protocol_r_config(config_path, allow_draft=True)
    instance = build_regular_variant(config, "iid_medium", "train", 0, 0)
    prepared = prepare_repair_problem(instance, context)
    assert len(prepared.prepared_sha256) == 64
    assert any(not __import__("math").isfinite(value) for row in prepared.costs for value in row)
