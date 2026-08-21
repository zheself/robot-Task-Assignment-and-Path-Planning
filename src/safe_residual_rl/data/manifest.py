"""Canonical, hash-addressed split manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


VALID_SPLITS = ("train", "validation", "test")


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class SplitEntry:
    sample_group: str
    session_id: str
    path_id: str
    date_id: str
    split: str
    source: str

    def as_dict(self) -> dict:
        return {
            "date_id": self.date_id,
            "path_id": self.path_id,
            "sample_group": self.sample_group,
            "session_id": self.session_id,
            "source": self.source,
            "split": self.split,
        }


@dataclass(frozen=True)
class SplitManifest:
    manifest_id: str
    robot_id: str
    evidence_level: str
    entries: tuple[SplitEntry, ...]
    schema_version: int = 1

    def payload(self) -> dict:
        return {
            "entries": [entry.as_dict() for entry in sorted(self.entries, key=lambda item: item.sample_group)],
            "evidence_level": self.evidence_level,
            "manifest_id": self.manifest_id,
            "robot_id": self.robot_id,
            "schema_version": self.schema_version,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def document(self) -> dict:
        return {**self.payload(), "sha256": self.sha256}

    def validate(self) -> None:
        if not self.entries:
            raise ValueError("manifest has no entries")
        groups: dict[str, str] = {}
        session_splits: dict[str, str] = {}
        path_splits: dict[str, str] = {}
        for entry in self.entries:
            if entry.split not in VALID_SPLITS:
                raise ValueError(f"invalid split {entry.split!r}")
            if entry.sample_group in groups:
                raise ValueError(f"duplicate sample_group: {entry.sample_group}")
            groups[entry.sample_group] = entry.split
            for label, value, mapping in (
                ("session", entry.session_id, session_splits),
                ("path", entry.path_id, path_splits),
            ):
                previous = mapping.setdefault(value, entry.split)
                if previous != entry.split:
                    raise ValueError(f"{label} leakage: {value} is in {previous} and {entry.split}")
        present = set(groups.values())
        if present != set(VALID_SPLITS):
            raise ValueError(f"manifest must contain {VALID_SPLITS}; found {sorted(present)}")

    def write_immutable(self, path: Path) -> None:
        self.validate()
        serialized = json.dumps(self.document(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != serialized:
                raise FileExistsError(f"refusing to overwrite immutable manifest: {path}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")


def load_manifest(path: Path) -> SplitManifest:
    document = json.loads(path.read_text(encoding="utf-8"))
    recorded_hash = document.pop("sha256")
    manifest = SplitManifest(
        manifest_id=document["manifest_id"],
        robot_id=document["robot_id"],
        evidence_level=document["evidence_level"],
        schema_version=document["schema_version"],
        entries=tuple(SplitEntry(**entry) for entry in document["entries"]),
    )
    manifest.validate()
    if manifest.sha256 != recorded_hash:
        raise ValueError(f"manifest hash mismatch: {path}")
    return manifest


def synthetic_manifest(path_counts: dict[str, int]) -> SplitManifest:
    entries = []
    for split in VALID_SPLITS:
        for index in range(path_counts[split]):
            prefix = "val" if split == "validation" else split
            date_id = "date_A" if split != "test" else "date_B"
            entries.append(
                SplitEntry(
                    sample_group=f"{prefix}_group_{index:02d}",
                    session_id=f"{prefix}_session_{index:02d}",
                    path_id=f"{prefix}_path_{index:02d}",
                    date_id=date_id,
                    split=split,
                    source="deterministic_synthetic_generator_v1",
                )
            )
    manifest = SplitManifest(
        manifest_id="synthetic_ur5_pre_advisor_v1",
        robot_id="ur5",
        evidence_level="SYNTHETIC",
        entries=tuple(entries),
    )
    manifest.validate()
    return manifest
