from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_residual_rl.allocation import (
    generate_paper_benchmark,
    load_oracle_context,
    load_paper_config,
    materialize_paper_benchmark,
    verify_paper_instances,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def v4_config():
    return load_paper_config(ROOT / "configs/allocation/benchmark_v4.json")


@pytest.fixture(scope="module")
def v4_materialized(v4_config, tmp_path_factory):
    root = tmp_path_factory.mktemp("a2-paper-v4")
    context = load_oracle_context(ROOT / "configs/allocation/oracle_proxy_v1.json")
    manifest, generated = materialize_paper_benchmark(v4_config, context, root)
    return root, manifest, generated


def test_v4_is_new_and_preregistered(v4_config) -> None:
    v2 = generate_paper_benchmark(
        load_paper_config(ROOT / "configs/allocation/benchmark_v2.json")
    )
    v3 = generate_paper_benchmark(
        load_paper_config(ROOT / "configs/allocation/benchmark_v3.json")
    )
    v4 = generate_paper_benchmark(v4_config)
    assert len(v4) == 408
    assert len({item.task_group_id for item in v4}) == 216
    old_groups = {item.task_group_id for item in v2 + v3}
    old_instances = {item.instance.instance_id for item in v2 + v3}
    assert not old_groups & {item.task_group_id for item in v4}
    assert not old_instances & {item.instance.instance_id for item in v4}
    assert v4_config.acceptance["zero_witness_failures"] is True


def test_v4_has_verified_hashed_witnesses(v4_materialized) -> None:
    root, manifest, generated = v4_materialized
    assert len(generated) == 408
    assert sum(item.witness_relative_path is not None for item in manifest.records) == 402
    assert sum(item.witness_sha256 is not None for item in manifest.records) == 402
    assert not verify_paper_instances(manifest, root)


def test_v4_witness_tampering_is_detected(v4_materialized) -> None:
    root, manifest, _ = v4_materialized
    record = next(item for item in manifest.records if item.witness_relative_path)
    path = root / str(record.witness_relative_path)
    original = path.read_text(encoding="utf-8")
    raw = json.loads(original)
    raw["witness_sha256"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    try:
        failures = verify_paper_instances(manifest, root)
        assert f"WITNESS_HASH_MISMATCH={record.instance_id}" in failures
        assert f"MANIFEST_WITNESS_HASH_MISMATCH={record.instance_id}" in failures
    finally:
        path.write_text(original, encoding="utf-8")
