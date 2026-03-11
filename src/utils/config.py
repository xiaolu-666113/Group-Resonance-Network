"""Configuration utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


ConfigDict = Dict[str, Any]


def load_config(config_path: str | Path) -> ConfigDict:
    """Load YAML config as dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping, got: {type(cfg)}")
    return cfg


def deep_update(base: ConfigDict, updates: ConfigDict) -> ConfigDict:
    """Recursively update dictionary without mutating the input."""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def set_by_dotted_path(config: ConfigDict, dotted_key: str, value: Any) -> None:
    """Set nested value in-place, e.g. `model.embedding_dim=128`."""
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def apply_overrides(config: ConfigDict, overrides: list[str]) -> ConfigDict:
    """Apply CLI overrides in KEY=VALUE format."""
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}")
        key, value_text = item.split("=", 1)
        value: Any
        try:
            value = yaml.safe_load(value_text)
        except Exception:
            value = value_text
        set_by_dotted_path(result, key, value)
    return result
