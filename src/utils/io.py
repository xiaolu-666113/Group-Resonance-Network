"""I/O helpers for experiment artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import torch


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Write JSON with indentation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def append_csv_row(path: str | Path, row: Dict[str, Any], fieldnames: Iterable[str] | None = None) -> None:
    """Append one row to CSV, creating header on first write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(row.keys())

    file_exists = p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    """Serialize checkpoint to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, p)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    """Load checkpoint dictionary."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return torch.load(p, map_location=map_location)
