from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

from safe_residual_rl.allocation.search.protocol_r_pipeline import (
    deterministic_merge_shards,
    validate_dependency_chain,
    validate_trace_matrix,
    validate_worker_environment,
)
from safe_residual_rl.allocation.search.trace import canonical_hash

ROOT = Path(__file__).resolve().parents[2]
CELLS = (
    "iid_small",
    "iid_medium",
    "dense_precedence",
    "resource_bottleneck",
    "tight_windows",
    "scale",
)
METHODS = ("m0", "m1")


def _trace(cell, index, method="m0", mode="fixed_iterations", complete=True):
    trace = {
        "instance_id": f"a4blnsr3-{cell}-{index}",
        "task_group_id": f"a4blnsr3-{cell}-g-{index}",
        "difficulty": cell,
        "method_id": method,
        "random_seed": 3253,
        "budget_mode": mode,
        "fixed_iteration_complete": complete if mode == "fixed_iterations" else False,
        "iterations_completed": 30 if complete and mode == "fixed_iterations" else 0,
        "steps": [],
    }
    trace["trace_sha256"] = canonical_hash(trace)
    return trace


def _shard(cell, traces):
    payload = {
        "version": "fixture",
        "cell": cell,
        "passed": True,
        "traces": traces,
        "config_sha256": "c" * 64,
        "manifest_sha256": "m" * 64,
        "source_sha256": "s" * 64,
        "protocol_document_sha256": "p" * 64,
    }
    payload["shard_sha256"] = canonical_hash(payload)
    return payload


def test_deterministic_merge_order_and_missing_duplicate_rejection():
    shards = [
        _shard(cell, [_trace(cell, 1, "m1"), _trace(cell, 0, "m0")])
        for cell in reversed(CELLS)
    ]
    merged, record = deterministic_merge_shards(
        shards, cells=CELLS, methods=METHODS
    )
    assert merged[0]["difficulty"] == "iid_small"
    assert merged[0]["instance_id"].endswith("-0")
    assert record["trace_count"] == 12
    with pytest.raises(RuntimeError, match="missing duplicate or foreign"):
        deterministic_merge_shards(shards[:-1], cells=CELLS, methods=METHODS)
    duplicate = list(shards)
    duplicate[-1] = duplicate[0]
    with pytest.raises(RuntimeError, match="missing duplicate or foreign"):
        deterministic_merge_shards(duplicate, cells=CELLS, methods=METHODS)


def test_exact_iteration_matrix_gate_counts_all_failures():
    fixed = [_trace("iid_small", index) for index in range(400)]
    timed = [
        _trace("iid_medium", index, mode="fixed_time", complete=False)
        for index in range(120)
    ]
    status = validate_trace_matrix(
        fixed + timed,
        fixed_iteration_expected=400,
        fixed_time_expected=120,
        required_iterations=30,
    )
    assert status["passed"] and status["exact_iteration_complete"] == 400
    failed = [dict(item) for item in fixed + timed]
    failed[0]["fixed_iteration_complete"] = False
    failed[0]["trace_sha256"] = canonical_hash(
        {key: value for key, value in failed[0].items() if key != "trace_sha256"}
    )
    assert not validate_trace_matrix(
        failed,
        fixed_iteration_expected=400,
        fixed_time_expected=120,
        required_iterations=30,
    )["passed"]


def test_worker_single_thread_cpu_affinity_and_cuda_contract():
    environment = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    assert validate_worker_environment(environment, [7])["passed"]
    assert not validate_worker_environment(environment, [7, 8])["passed"]
    changed = dict(environment, MKL_NUM_THREADS="2")
    assert not validate_worker_environment(changed, [7])["passed"]


def test_afterok_chain_and_failed_predecessor_cannot_be_afterany():
    stages = ["preflight", "generate", "profile", "profile_gate"]
    submitted = []
    for index, stage in enumerate(stages):
        submitted.append(
            {
                "stage": stage,
                "job_id": str(100 + index),
                "dependency": None if index == 0 else f"afterok:{99 + index}",
            }
        )
    assert validate_dependency_chain(submitted, stages)["passed"]
    submitted[-1]["dependency"] = "afterany:102"
    assert not validate_dependency_chain(submitted, stages)["passed"]


def test_slurm_resources_are_cpu_only_and_match_draft():
    scripts = sorted((ROOT / "slurm").glob("a4b_protocol_r_*.sbatch"))
    assert len(scripts) == 9
    combined = "\n".join(path.read_text() for path in scripts)
    assert "--partition=normal" in combined and "--account=v-chengwy" in combined
    assert "--gres" not in combined.lower() and "--gpus" not in combined.lower()
    assert "--cpus-per-task=6" in combined and "--mem=32G" in combined
    assert "CUDA_VISIBLE_DEVICES" in (ROOT / "scripts/run_a4b_protocol_r_worker.sh").read_text()


def test_submit_wrapper_contains_unique_guard_same_node_and_linear_afterok():
    text = (ROOT / "scripts/submit_a4b_protocol_r_chain.sh").read_text()
    assert "squeue" in text and "^a4b-r3-" in text
    assert "--nodelist=\"${selected_node}\"" in text
    assert '--dependency="afterok:${dependency}"' in text
    assert "A4B_CONFIRM_PROTOCOL_R_SUBMIT" in text
    assert "EXECUTE_A4B_PROTOCOL_R_V3" in text
    assert "afterany" not in text


def test_draft_cli_refuses_preflight_without_creating_output(tmp_path):
    command = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/run_a4b_protocol_r.py"),
        "preflight",
        "--output",
        str(tmp_path / "a4blnsr3"),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "draft Protocol R cannot generate data or execute search" in result.stderr
    assert not (tmp_path / "a4blnsr3").exists()


def test_runner_source_hash_matrix_has_no_neural_or_checkpoint_stage():
    spec = importlib.util.spec_from_file_location(
        "protocol_r_runner", ROOT / "scripts/run_a4b_protocol_r.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert all((ROOT / path).is_file() for path in module.SOURCE_FILES)
    assert all(len(module.sha256_file(ROOT / path)) == 64 for path in module.SOURCE_FILES)
    joined = " ".join(module.SOURCE_FILES).lower()
    assert "checkpoint" not in joined and "neural" not in joined
