"""Signal processing tests."""

from __future__ import annotations

import numpy as np

from src.utils.signal_ops import (
    build_resonance_map,
    compute_de_features,
    compute_plv_matrix,
    to_channel_series,
)


def test_compute_de_features_shape() -> None:
    raw = np.random.randn(32, 512).astype(np.float32)
    de = compute_de_features(raw_signal=raw, fs=128.0, window_sec=1.0, step_sec=0.5)
    assert de.ndim == 3
    assert de.shape[1] == 32
    assert de.shape[2] == 5


def test_plv_range() -> None:
    a = np.random.randn(4, 200)
    b = np.random.randn(4, 200)
    plv = compute_plv_matrix(a, b)
    assert plv.shape == (4, 4)
    assert np.all(plv >= 0.0)
    assert np.all(plv <= 1.0)


def test_resonance_map_shape_and_finite() -> None:
    q = np.random.randn(10, 16, 5).astype(np.float32)
    r = np.random.randn(10, 16, 5).astype(np.float32)
    m = build_resonance_map(q, r, fs=128.0, nperseg=64)
    assert m.shape == (16, 16, 2)
    assert np.isfinite(m).all()


def test_to_channel_series_feature_shape() -> None:
    sample = np.random.randn(10, 32, 5).astype(np.float32)
    cs = to_channel_series(sample)
    assert cs.shape == (32, 50)
