from __future__ import annotations
import os
from pathlib import Path
from typing import Any, MutableMapping

import yaml


def load_config(path: str | None = None) -> dict[str, Any]:
    path = path or os.environ.get("BGFM_CONFIG")
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"BGFM config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML config must be a mapping")
    return cfg


def load_section(name: str, path: str | None = None) -> dict[str, Any]:
    section = load_config(path).get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return section


def apply_globals(namespace: MutableMapping[str, Any], section: dict[str, Any]) -> None:
    for key, value in section.items():
        candidates = [key, str(key).upper()]
        target = next((k for k in candidates if k in namespace), None)
        if target is not None:
            namespace[target] = value


def apply_mapping(target: MutableMapping[str, Any], section: dict[str, Any]) -> None:
    existing = {str(k).lower(): k for k in target.keys()}
    for key, value in section.items():
        actual = existing.get(str(key).lower(), key)
        target[actual] = value
