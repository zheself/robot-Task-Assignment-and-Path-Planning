#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safe_residual_rl.data import synthetic_manifest


def main() -> None:
    manifest = synthetic_manifest({"train": 6, "validation": 3, "test": 3})
    target = ROOT / "data" / "manifests" / "synthetic_ur5_pre_advisor_v1.json"
    manifest.write_immutable(target)
    print(f"{target}\nsha256={manifest.sha256}")


if __name__ == "__main__":
    main()
