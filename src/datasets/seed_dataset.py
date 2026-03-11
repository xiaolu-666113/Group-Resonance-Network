"""SEED dataset adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base_dataset import BaseEEGDataset, load_npz


SEED_CLASS_NAMES = ["Negative", "Neutral", "Positive"]


def _normalize_seed_labels(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim > 1:
        arr = arr.squeeze()
    unique = set(np.unique(arr).tolist())
    if unique.issubset({-1, 0, 1}):
        mapper = {-1: 0, 0: 1, 1: 2}
        return np.vectorize(mapper.get)(arr).astype(np.int64)
    if unique.issubset({0, 1, 2}):
        return arr.astype(np.int64)
    raise ValueError(f"Unsupported SEED labels: {sorted(unique)}")


class SEEDDataset(BaseEEGDataset):
    """SEED dataset in canonical format."""

    @classmethod
    def from_npz(cls, path: str | Path, input_mode: str = "features") -> "SEEDDataset":
        data = load_npz(path)
        required = {"x", "y", "subject_id"}
        missing = required.difference(data.keys())
        if missing:
            raise KeyError(f"Missing required SEED keys: {missing}")

        y = _normalize_seed_labels(data["y"])
        return cls(
            x=data["x"],
            y=y,
            subject_id=data["subject_id"],
            name="SEED",
            class_names=SEED_CLASS_NAMES,
            input_mode=input_mode,
            session_id=data.get("session_id"),
            trial_id=data.get("trial_id"),
        )
