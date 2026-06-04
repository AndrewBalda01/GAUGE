"""Load, validate and hash-verify a versioned JSONL dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from store.schema import DatasetManifest, MetricKind, TestCase

_DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(name: str, version: str = "v1") -> tuple[list[TestCase], DatasetManifest]:
    """Return (cases, manifest) after verifying the SHA-256 integrity."""
    jsonl_path = _DATASETS_DIR / f"{name}.{version}.jsonl"
    manifest_path = _DATASETS_DIR / f"{name}.{version}.manifest.json"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = DatasetManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))

    actual_hash = _sha256(jsonl_path)
    if actual_hash != manifest.sha256:
        raise ValueError(
            f"Hash mismatch for {jsonl_path.name}: "
            f"expected {manifest.sha256[:16]}… got {actual_hash[:16]}…"
        )

    raw_lines = [l for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    cases: list[TestCase] = []
    for i, line in enumerate(raw_lines, 1):
        data = json.loads(line)
        # coerce string metric names to enum
        if "metrics" in data:
            data["metrics"] = [MetricKind(m) for m in data["metrics"]]
        cases.append(TestCase(**data))

    if len(cases) != manifest.case_count:
        raise ValueError(
            f"Case count mismatch: manifest says {manifest.case_count}, file has {len(cases)}"
        )

    return cases, manifest


def filter_cases(
    cases: list[TestCase],
    tags: list[str] | None = None,
    ids: list[str] | None = None,
) -> list[TestCase]:
    result = cases
    if tags:
        tag_set = set(tags)
        result = [c for c in result if tag_set.intersection(c.tags)]
    if ids:
        id_set = set(ids)
        result = [c for c in result if c.id in id_set]
    return result
