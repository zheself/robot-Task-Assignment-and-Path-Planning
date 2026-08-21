from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_residual_rl.allocation import (
    SolverProtocol,
    audit_paper_leakage,
    cluster_mean_ci,
    evaluate_acceptance,
    frozen_pairwise_statistics,
    generate_paper_benchmark,
    load_oracle_context,
    load_paper_config,
    load_paper_manifest,
    materialize_paper_benchmark,
    paper_split_counts,
    proxy_admissibility,
    solve_assignment_milp,
    solve_deterministic_lns,
    solve_greedy,
    solve_hungarian,
    solve_load_balanced,
    verify_paper_instances,
    write_paper_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config():
    return load_paper_config(ROOT / "configs/allocation/benchmark_v2.json")


@pytest.fixture(scope="module")
def context():
    return load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")


@pytest.fixture(scope="module")
def generated(config):
    return generate_paper_benchmark(config)


@pytest.fixture(scope="module")
def materialized(config, context, tmp_path_factory):
    root = tmp_path_factory.mktemp("a2-paper-v2")
    manifest, instances = materialize_paper_benchmark(config, context, root)
    path = root / "manifest.json"
    write_paper_manifest(manifest, path)
    return root, path, manifest, instances


def test_preregistered_counts_and_independent_groups(config, generated) -> None:
    counts = {}
    for item in generated:
        counts[item.split] = counts.get(item.split, 0) + 1
    assert counts == {"train": 192, "validation": 48, "frozen_test": 144, "stress": 24}
    assert len(generated) == 408
    assert len({item.task_group_id for item in generated}) == 216
    assert config.statistics["independent_unit"] == "task_group_id"


def test_difficulty_cells_are_frozen(config) -> None:
    counts = {}
    for cell in config.cells:
        counts[cell.split] = counts.get(cell.split, 0) + 1
    assert counts == {"train": 4, "validation": 4, "frozen_test": 6, "stress": 4}
    assert sum(cell.groups * cell.variants for cell in config.cells) == 408
    assert len(config.config_sha256) == 64


def test_generation_is_deterministic(config, generated) -> None:
    second = generate_paper_benchmark(config)
    assert [(item.instance.instance_id, item.seed, item.instance.to_dict()) for item in generated] == [
        (item.instance.instance_id, item.seed, item.instance.to_dict()) for item in second
    ]


def test_sibling_variants_share_geometry_and_split(generated) -> None:
    left, right = next(
        (left, right)
        for left in generated
        for right in generated
        if left.task_group_id == right.task_group_id and left.variant_index == 0 and right.variant_index == 1
    )
    assert left.split == right.split
    assert left.instance.layout_id == right.instance.layout_id
    assert left.instance.robots == right.instance.robots
    assert [item.sampled_curve_m for item in left.instance.segments] == [item.sampled_curve_m for item in right.instance.segments]


def test_proxy_admissibility_matches_preregistered_policy(generated, context) -> None:
    counts = {("admissible_required", True): 0, ("designed_edge_infeasible", False): 0}
    for item in generated:
        result = proxy_admissibility(item.instance, context)
        counts[(item.feasibility_policy, result.admissible)] += 1
    assert counts == {("admissible_required", True): 402, ("designed_edge_infeasible", False): 6}


def test_negative_control_is_detected_by_all_baselines(generated, context, config) -> None:
    instance = next(item.instance for item in generated if item.cell_id == "designed_edge_infeasible")
    protocol = SolverProtocol("a1-solver-protocol-v1", 1.0, 0.0, 0)
    results = [
        solve_greedy(instance, context),
        solve_load_balanced(instance, context),
        solve_hungarian(instance, context),
        solve_assignment_milp(instance, context, protocol),
        solve_deterministic_lns(instance, context, iterations=2, seed=0, objective_weights=dict(config.objective_weights)),
    ]
    assert {item.status for item in results} == {"infeasible"}
    assert all(item.plan is None for item in results)


def test_manifest_roundtrip_leakage_and_hashes(materialized) -> None:
    root, path, manifest, _ = materialized
    assert load_paper_manifest(path) == manifest
    assert paper_split_counts(manifest.records) == {"frozen_test": 144, "stress": 24, "train": 192, "validation": 48}
    assert not audit_paper_leakage(manifest.records)
    assert not verify_paper_instances(manifest, root)


def test_paper_manifest_tampering_is_rejected(materialized, tmp_path) -> None:
    _, path, _, _ = materialized
    data = json.loads(path.read_text())
    data["records"][0]["cell_id"] = "tampered"
    target = tmp_path / "tampered.json"
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_paper_manifest(target)


def test_cluster_bootstrap_is_deterministic() -> None:
    first = cluster_mean_ci([0.0, 0.5, 1.0], 1000, 0.95, 7)
    second = cluster_mean_ci([0.0, 0.5, 1.0], 1000, 0.95, 7)
    assert first == second
    assert first[0] == pytest.approx(0.5)


def test_pairwise_statistics_are_group_clustered_and_holm_adjusted() -> None:
    rows = []
    for group in range(6):
        for variant in range(2):
            instance_id = f"g{group}-v{variant}"
            for method, score in (("candidate", 8.0 + group), ("assignment_milp", 10.0 + group), ("deterministic_lns", 9.0 + group)):
                rows.append({"split":"frozen_test","cell_id":"cell","method":method,"instance_id":instance_id,"task_group_id":f"g{group}","verified":True,"weighted_proxy_score":score})
    result = frozen_pairwise_statistics(rows, ["assignment_milp", "deterministic_lns"], 500, 0.95, 3)
    selected = next(item for item in result if item["method"] == "candidate" and item["reference_method"] == "assignment_milp")
    assert selected["groups_jointly_verified"] == 6
    assert selected["score_difference"] == pytest.approx(-2.0)
    assert selected["wilcoxon_p_holm"] is not None


def test_acceptance_policy_distinguishes_expected_negative_control(config) -> None:
    rows = []
    for split, cell, count in (("train", "train-cell", 2), ("validation", "val-cell", 2), ("frozen_test", "test-cell", 2)):
        for index in range(count):
            for method in ("greedy", "assignment_milp"):
                rows.append({"instance_id":f"{split}-{index}","split":split,"cell_id":cell,"method":method,"status":"feasible","verified":True})
    for method in ("greedy", "assignment_milp"):
        rows.append({"instance_id":"negative-0","split":"stress","cell_id":"designed_edge_infeasible","method":method,"status":"infeasible","verified":False})
    result = evaluate_acceptance(rows, config.acceptance, {"schema_failures":0,"split_leakage":0,"hash_failures":0})
    assert result["passed"]
    assert result["designed_infeasible_detection_rate"] == 1.0


def test_v3_is_preregistered_with_new_independent_groups(config) -> None:
    v3 = load_paper_config(ROOT / "configs/allocation/benchmark_v3.json")
    generated_v2 = generate_paper_benchmark(config)
    generated_v3 = generate_paper_benchmark(v3)
    assert len(generated_v3) == 408
    assert len({item.task_group_id for item in generated_v3}) == 216
    assert not (
        {item.task_group_id for item in generated_v2}
        & {item.task_group_id for item in generated_v3}
    )
    assert not (
        {item.instance.instance_id for item in generated_v2}
        & {item.instance.instance_id for item in generated_v3}
    )
    assert v3.master_seed != config.master_seed
    assert v3.acceptance == config.acceptance
