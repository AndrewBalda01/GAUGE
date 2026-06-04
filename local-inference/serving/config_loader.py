"""Load engine configs from YAML files in serving/configs/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIGS_DIR = Path(__file__).parent / "configs"


def load_config(name: str) -> dict[str, Any]:
    path = _CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def list_configs() -> list[dict[str, Any]]:
    configs = []
    for p in sorted(_CONFIGS_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        cfg["_file"] = p.stem
        configs.append(cfg)
    return configs


def load_all_configs(exclude_mock: bool = False) -> list[dict[str, Any]]:
    configs = list_configs()
    if exclude_mock:
        configs = [c for c in configs if c.get("backend") != "mock"]
    return configs
