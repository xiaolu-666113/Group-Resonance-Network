"""Base dataset implementation for canonical EEG format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetMeta:
    """Metadata container for EEG datasets."""

    name: str
    class_names: list[str]
    input_mode: str


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    """Load npz file into plain dictionary."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    data = np.load(p, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _to_feature_tensor(x: np.ndarray, input_mode: str) -> np.ndarray:
    """Normalize EEG tensor into [N, T, C, Bf]."""
    arr = np.asarray(x)
    if arr.ndim == 4:
        # Common case: [N, T, C, Bf]
        if arr.shape[-1] <= 16 and arr.shape[2] <= 256:
            return arr.astype(np.float32)
        # Alternate case: [N, C, Bf, T]
        if arr.shape[1] <= 256 and arr.shape[2] <= 16:
            return np.transpose(arr, (0, 3, 1, 2)).astype(np.float32)
        raise ValueError(
            f"Unsupported 4D tensor shape {arr.shape}. Expected [N,T,C,Bf] or [N,C,Bf,T]."
        )

    if arr.ndim == 3:
        if input_mode == "raw":
            # [N, C, T] -> [N, T, C, 1]
            if arr.shape[1] <= 256 and arr.shape[2] > arr.shape[1]:
                out = np.transpose(arr, (0, 2, 1))
            # [N, T, C] -> [N, T, C, 1]
            elif arr.shape[2] <= 256 and arr.shape[1] > arr.shape[2]:
                out = arr
            else:
                # fallback: assume second axis is channels when small
                out = np.transpose(arr, (0, 2, 1)) if arr.shape[1] <= 256 else arr
            return out[..., None].astype(np.float32)

        # feature mode but missing band axis: [N, T, C]
        if arr.shape[2] <= 256:
            return arr[..., None].astype(np.float32)
        raise ValueError(f"Unsupported 3D feature tensor shape {arr.shape}")

    raise ValueError(f"Unsupported EEG tensor shape {arr.shape}")


class BaseEEGDataset(Dataset):
    """Base EEG dataset with canonical sample dictionary output.

    Output sample dictionary keys:
    - x: torch.float32 [T, C, Bf]
    - y: torch.long []
    - subject_id: torch.long []
    - index: torch.long []
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        subject_id: np.ndarray,
        name: str,
        class_names: list[str],
        input_mode: str = "features",
        session_id: Optional[np.ndarray] = None,
        trial_id: Optional[np.ndarray] = None,
    ) -> None:
        self.meta = DatasetMeta(name=name, class_names=class_names, input_mode=input_mode)
        self.x = _to_feature_tensor(x, input_mode=input_mode)
        self.y = np.asarray(y, dtype=np.int64)
        self.subject_id = np.asarray(subject_id, dtype=np.int64)
        self.session_id = None if session_id is None else np.asarray(session_id, dtype=np.int64)
        self.trial_id = None if trial_id is None else np.asarray(trial_id, dtype=np.int64)

        n = self.x.shape[0]
        assert self.y.shape[0] == n, f"Label count mismatch: {self.y.shape[0]} vs {n}"
        assert self.subject_id.shape[0] == n, f"Subject count mismatch: {self.subject_id.shape[0]} vs {n}"

        self._flat_feature_cache: np.ndarray | None = None

    @property
    def num_classes(self) -> int:
        return len(self.meta.class_names)

    @property
    def class_names(self) -> list[str]:
        return self.meta.class_names

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[index]).float(),
            "y": torch.tensor(self.y[index], dtype=torch.long),
            "subject_id": torch.tensor(self.subject_id[index], dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }

    def get_x(self, indices: np.ndarray) -> np.ndarray:
        """Fetch feature tensor by global indices."""
        return self.x[indices]

    def get_y(self, indices: np.ndarray) -> np.ndarray:
        """Fetch labels by global indices."""
        return self.y[indices]

    def get_subject_ids(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Fetch subject ids by indices or all if None."""
        if indices is None:
            return self.subject_id
        return self.subject_id[indices]

    def flat_features(self) -> np.ndarray:
        """Return flattened feature vectors [N, D] for nearest-reference strategy."""
        if self._flat_feature_cache is None:
            self._flat_feature_cache = self.x.reshape(self.x.shape[0], -1).astype(np.float32)
        return self._flat_feature_cache
