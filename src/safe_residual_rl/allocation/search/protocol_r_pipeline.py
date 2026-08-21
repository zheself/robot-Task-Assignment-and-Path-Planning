"""Fail-closed shard, matrix, environment and dependency audits for Protocol R."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .trace import canonical_hash


def trace_identity(trace: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        trace["instance_id"],
        trace["method_id"],
        int(trace["random_seed"]),
        trace["budget_mode"],
    )


def trace_sort_key(
    trace: Mapping[str, Any], cells: Sequence[str], methods: Sequence[str]
) -> tuple[object, ...]:
    return (
        cells.index(str(trace["difficulty"])),
        str(trace["instance_id"]),
        methods.index(str(trace["method_id"])),
        int(trace["random_seed"]),
        0 if trace["budget_mode"] == "fixed_iterations" else 1,
    )


def validate_trace_matrix(
    traces: Sequence[Mapping[str, Any]],
    *,
    fixed_iteration_expected: int,
    fixed_time_expected: int,
    required_iterations: int,
) -> dict[str, object]:
    identities = [trace_identity(trace) for trace in traces]
    fixed = [trace for trace in traces if trace["budget_mode"] == "fixed_iterations"]
    timed = [trace for trace in traces if trace["budget_mode"] == "fixed_time"]
    hash_valid = all(
        canonical_hash({key: value for key, value in trace.items() if key != "trace_sha256"})
        == trace.get("trace_sha256")
        for trace in traces
    )
    exact_complete = all(
        bool(trace.get("fixed_iteration_complete"))
        and int(trace.get("iterations_completed", -1)) == required_iterations
        for trace in fixed
    )
    passed = (
        len(fixed) == fixed_iteration_expected
        and len(timed) == fixed_time_expected
        and len(identities) == len(set(identities))
        and hash_valid
        and exact_complete
    )
    return {
        "fixed_iteration_traces": len(fixed),
        "fixed_time_traces": len(timed),
        "unique_identities": len(set(identities)),
        "duplicate_identities": len(identities) - len(set(identities)),
        "exact_iteration_complete": sum(
            bool(trace.get("fixed_iteration_complete"))
            and int(trace.get("iterations_completed", -1)) == required_iterations
            for trace in fixed
        ),
        "hash_valid": hash_valid,
        "passed": passed,
    }


def deterministic_merge_shards(
    shards: Sequence[Mapping[str, Any]],
    *,
    cells: Sequence[str],
    methods: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    actual_cells = [str(shard.get("cell")) for shard in shards]
    if Counter(actual_cells) != Counter(cells):
        raise RuntimeError("missing duplicate or foreign Protocol-R shard")
    shared_fields = (
        "config_sha256",
        "manifest_sha256",
        "source_sha256",
        "protocol_document_sha256",
    )
    for shard in shards:
        saved = shard.get("shard_sha256")
        unsigned = {key: value for key, value in shard.items() if key != "shard_sha256"}
        if canonical_hash(unsigned) != saved or not shard.get("passed"):
            raise RuntimeError(f"invalid or failed Protocol-R shard: {shard.get('cell')}")
    for field in shared_fields:
        values = {shard.get(field) for shard in shards}
        if len(values) != 1 or None in values:
            raise RuntimeError(f"Protocol-R shard provenance mismatch: {field}")
    traces = [dict(trace) for shard in shards for trace in shard.get("traces", [])]
    identities = [trace_identity(trace) for trace in traces]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate Protocol-R trace identity")
    ordered = sorted(traces, key=lambda trace: trace_sort_key(trace, cells, methods))
    record = {
        "version": "a4b-protocol-r-deterministic-merge-v1",
        "cells": list(cells),
        "shard_sha256": {
            str(shard["cell"]): shard["shard_sha256"] for shard in shards
        },
        "trace_count": len(ordered),
        "trace_identity_sha256": canonical_hash([trace_identity(item) for item in ordered]),
        "transition_sha256": canonical_hash(
            [
                (
                    trace_identity(item),
                    [
                        (
                            step["iteration"],
                            step["operator"],
                            step["destroy_set"],
                            step["candidate_state_sha256"],
                            step["accepted"],
                        )
                        for step in item.get("steps", [])
                    ],
                )
                for item in ordered
            ]
        ),
    }
    record["merge_sha256"] = canonical_hash(record)
    return ordered, record


def validate_worker_environment(
    environment: Mapping[str, str], affinity_cpus: Sequence[int]
) -> dict[str, object]:
    expected = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    checks = {key: environment.get(key) == value for key, value in expected.items()}
    checks["one_cpu"] = len(tuple(affinity_cpus)) == 1
    checks["cpu_is_unique_scalar"] = len(set(affinity_cpus)) == len(tuple(affinity_cpus))
    return {"checks": checks, "passed": all(checks.values())}


def validate_dependency_chain(
    submitted: Sequence[Mapping[str, Any]], expected_stages: Sequence[str]
) -> dict[str, object]:
    checks = []
    for index, stage in enumerate(expected_stages):
        matching = [item for item in submitted if item.get("stage") == stage]
        if len(matching) != 1:
            checks.append(False)
            continue
        dependency = matching[0].get("dependency")
        if index == 0:
            checks.append(dependency in (None, ""))
        else:
            previous = next(
                (item for item in submitted if item.get("stage") == expected_stages[index - 1]),
                None,
            )
            checks.append(
                previous is not None
                and dependency == f"afterok:{previous.get('job_id')}"
            )
    return {
        "stages": list(expected_stages),
        "checks": checks,
        "passed": len(checks) == len(expected_stages) and all(checks),
    }


def conjunctive_gate(checks: Mapping[str, bool]) -> dict[str, object]:
    return {"checks": dict(checks), "passed": bool(checks) and all(checks.values())}
