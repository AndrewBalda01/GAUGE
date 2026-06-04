"""Run once to generate/update a dataset manifest file."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_manifest(jsonl_path: Path) -> None:
    cases = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    manifest = {
        "name": jsonl_path.stem.rsplit(".", 1)[0],
        "version": jsonl_path.stem.rsplit(".", 1)[-1],
        "sha256": sha256_file(jsonl_path),
        "case_count": len(cases),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Legal Q&A evaluation dataset — contract, tort, criminal, IP, privacy, property law.",
    }
    out = jsonl_path.with_suffix(".manifest.json")
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Written {out}  ({len(cases)} cases, sha256={manifest['sha256'][:16]}…)")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "legal_qa.v1.jsonl"
    make_manifest(target)
