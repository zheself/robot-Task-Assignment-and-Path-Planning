"""Materialise small auditable A0 JSON fixtures with explicit patch operations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def load_auditable_fixture(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    with fixture_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "extends" not in payload:
        return payload
    base = load_auditable_fixture(fixture_path.parent / payload["extends"])
    materialized = copy.deepcopy(base)
    materialized["fixture_id"] = payload["fixture_id"]
    materialized["description"] = payload["description"]
    materialized["expected"] = payload["expected"]
    for operation in payload.get("patch", []):
        _apply_patch(materialized["instance"], operation)
    return materialized


def _apply_patch(document: Any, operation: dict[str, Any]) -> None:
    op = operation["op"]
    if op not in {"add", "replace", "remove"}:
        raise ValueError(f"unsupported fixture patch operation: {op}")
    parts = [_decode(x) for x in operation["path"].split("/")[1:]]
    target = document
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    key = parts[-1]
    if isinstance(target, list):
        if op == "add" and key == "-":
            target.append(copy.deepcopy(operation["value"]))
        elif op == "remove":
            target.pop(int(key))
        else:
            target[int(key)] = copy.deepcopy(operation["value"])
    elif op == "remove":
        del target[key]
    else:
        target[key] = copy.deepcopy(operation["value"])


def _decode(value: str) -> str:
    return value.replace("~1", "/").replace("~0", "~")
