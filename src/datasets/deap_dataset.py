"""DEAP dataset adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base_dataset import BaseEEGDataset, load_npz


DEAP_CLASS_NAMES = ["Low", "High"]


def _binarize(values: np.ndarray, threshold: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return (arr >= threshold).astype(np.int64)


def _resolve_deap_labels(data: dict[str, np.ndarray], task: str, threshold: float) -> np.ndarray:
    task = task.lower()
    if task not in {"valence", "arousal"}:
        raise ValueError(f"Unsupported DEAP task: {task}")

    if task in data:
        return _binarize(data[task], threshold=threshold)

    if "y" in data:
        y = np.asarray(data["y"])
        if y.ndim == 1:
            unique = set(np.unique(y).tolist())
            if unique.issubset({0, 1}):
                return y.astype(np.int64)
            return _binarize(y, threshold=threshold)
        if y.ndim == 2 and y.shape[1] >= 2:
            col = 0 if task == "valence" else 1
            col_vals = y[:, col]
            unique = set(np.unique(col_vals).tolist())
            if unique.issubset({0, 1}):
                return col_vals.astype(np.int64)
            return _binarize(col_vals, threshold=threshold)

    raise KeyError(
        "Cannot resolve DEAP labels. Provide one of: "
        "`valence`/`arousal` arrays, or `y` with shape [N] or [N,2]."
    )


class DEAPDataset(BaseEEGDataset):
    """DEAP dataset in canonical format."""

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        task: str = "valence",
        threshold: float = 5.0,
        input_mode: str = "features",
    ) -> "DEAPDataset":
        data = load_npz(path)
        if "x" not in data or "subject_id" not in data:
            raise KeyError("DEAP npz must contain keys: x, subject_id and label keys")

        y = _resolve_deap_labels(data, task=task, threshold=threshold)
        return cls(
            x=data["x"],
            y=y,
            subject_id=data["subject_id"],
            name=f"DEAP-{task}",
            class_names=DEAP_CLASS_NAMES,
            input_mode=input_mode,
            session_id=data.get("session_id"),
            trial_id=data.get("trial_id"),
        )
