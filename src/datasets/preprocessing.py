"""Preprocessing utilities for raw EEG to DE feature caching."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from src.datasets.base_dataset import load_npz
from src.utils.signal_ops import DEFAULT_BANDS, compute_de_features


def preprocess_raw_npz_to_de(
    input_path: str | Path,
    output_path: str | Path,
    fs: float,
    bands: Dict[str, Tuple[float, float]] | None = None,
    window_sec: float = 1.0,
    step_sec: float = 0.5,
    overwrite: bool = False,
) -> Path:
    """Convert canonical raw EEG npz into cached DE features npz.

    Input keys required:
    - x: [N, C, T] or [N, T, C]
    - y: [N]
    - subject_id: [N]
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    data = load_npz(input_path)
    required = {"x", "y", "subject_id"}
    missing = required.difference(data.keys())
    if missing:
        raise KeyError(f"Missing required keys in raw dataset: {missing}")

    x = np.asarray(data["x"])  # [N, C, T] or [N, T, C]
    if x.ndim != 3:
        raise ValueError(f"Expected raw x shape [N,C,T] or [N,T,C], got {x.shape}")

    bands = bands or DEFAULT_BANDS
    features = []
    for i in tqdm(range(x.shape[0]), desc="Preprocessing raw EEG"):
        de = compute_de_features(
            raw_signal=x[i],
            fs=fs,
            bands=bands,
            window_sec=window_sec,
            step_sec=step_sec,
        )
        features.append(de)

    # handle variable window counts by truncating to the minimum length
    t_min = min(feat.shape[0] for feat in features)
    x_feat = np.stack([feat[:t_min] for feat in features], axis=0).astype(np.float32)

    save_dict = {
        "x": x_feat,
        "y": np.asarray(data["y"]),
        "subject_id": np.asarray(data["subject_id"]),
    }
    for key in ("session_id", "trial_id", "valence", "arousal"):
        if key in data:
            save_dict[key] = np.asarray(data[key])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    return output_path
