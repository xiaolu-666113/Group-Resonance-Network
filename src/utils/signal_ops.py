"""Signal processing operations used by GRN."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.signal import butter, coherence, filtfilt, hilbert


DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 31.0),
    "gamma": (31.0, 50.0),
}


def _ensure_channel_first(x: np.ndarray) -> np.ndarray:
    """Return signal as [C, T]."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D signal, got shape {arr.shape}")
    c_first = arr
    if arr.shape[0] > arr.shape[1]:
        c_first = arr.T
    return c_first


def bandpass_filter(signal_ct: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Band-pass filter for [C, T] signal."""
    if low <= 0 or high <= low:
        raise ValueError(f"Invalid band range: [{low}, {high}]")
    nyq = 0.5 * fs
    high = min(high, nyq - 1e-3)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal_ct, axis=-1)


def differential_entropy(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute differential entropy assuming Gaussianity.

    x: [C, T_window]
    returns: [C]
    """
    var = np.var(x, axis=-1, ddof=1)
    return 0.5 * np.log(2.0 * np.pi * np.e * np.maximum(var, eps))


def compute_de_features(
    raw_signal: np.ndarray,
    fs: float,
    bands: Dict[str, Tuple[float, float]] | None = None,
    window_sec: float = 1.0,
    step_sec: float = 0.5,
) -> np.ndarray:
    """Compute band-wise DE features from raw EEG.

    raw_signal: [C, T] or [T, C]
    returns: [T_win, C, B]
    """
    bands = bands or DEFAULT_BANDS
    x_ct = _ensure_channel_first(raw_signal)
    c, t = x_ct.shape

    win = max(1, int(round(window_sec * fs)))
    step = max(1, int(round(step_sec * fs)))
    if t < win:
        raise ValueError(f"Signal length {t} shorter than window {win}")

    band_names = list(bands.keys())
    filtered = []
    for name in band_names:
        low, high = bands[name]
        filtered.append(bandpass_filter(x_ct, fs=fs, low=low, high=high))
    filtered_arr = np.stack(filtered, axis=-1)  # [C, T, B]

    windows = []
    for start in range(0, t - win + 1, step):
        seg = filtered_arr[:, start : start + win, :]  # [C, win, B]
        de_cb = np.stack(
            [differential_entropy(seg[:, :, b]) for b in range(seg.shape[-1])],
            axis=-1,
        )
        windows.append(de_cb)

    out = np.stack(windows, axis=0)  # [T_win, C, B]
    return out.astype(np.float32)


def to_channel_series(sample: np.ndarray) -> np.ndarray:
    """Convert feature/raw sample to [C, L] channel series for resonance ops.

    Supported:
    - [T, C, B]
    - [C, T]
    - [T, C]
    """
    arr = np.asarray(sample, dtype=np.float64)
    if arr.ndim == 3:
        # [T, C, B] -> [C, T*B]
        if arr.shape[1] <= 512:
            t, c, b = arr.shape
            return np.transpose(arr, (1, 0, 2)).reshape(c, t * b)
        raise ValueError(f"Unsupported 3D sample shape: {arr.shape}")
    if arr.ndim == 2:
        return _ensure_channel_first(arr)
    raise ValueError(f"Unsupported sample dimensions: {arr.shape}")


def compute_plv_matrix(a_cl: np.ndarray, b_cl: np.ndarray) -> np.ndarray:
    """Compute PLV matrix between channel sets.

    a_cl, b_cl: [C, L]
    returns: [C, C]
    """
    if a_cl.shape[1] != b_cl.shape[1]:
        length = min(a_cl.shape[1], b_cl.shape[1])
        a_cl = a_cl[:, :length]
        b_cl = b_cl[:, :length]

    phase_a = np.angle(hilbert(a_cl, axis=1))
    phase_b = np.angle(hilbert(b_cl, axis=1))
    ea = np.exp(1j * phase_a)
    eb = np.exp(1j * phase_b)
    plv = np.abs((ea @ np.conj(eb).T) / ea.shape[1])
    return np.clip(plv.real, 0.0, 1.0)


def compute_coherence_matrix(
    a_cl: np.ndarray,
    b_cl: np.ndarray,
    fs: float,
    nperseg: int = 128,
) -> np.ndarray:
    """Compute mean coherence matrix between all channel pairs."""
    if a_cl.shape[1] != b_cl.shape[1]:
        length = min(a_cl.shape[1], b_cl.shape[1])
        a_cl = a_cl[:, :length]
        b_cl = b_cl[:, :length]

    c = a_cl.shape[0]
    seg = int(min(max(8, nperseg), a_cl.shape[1]))
    out = np.zeros((c, c), dtype=np.float64)
    for i in range(c):
        for j in range(c):
            _, coh = coherence(a_cl[i], b_cl[j], fs=fs, nperseg=seg)
            coh = np.nan_to_num(coh, nan=0.0, posinf=1.0, neginf=0.0)
            out[i, j] = float(np.mean(coh))
    return np.clip(out, 0.0, 1.0)


def build_resonance_map(
    query_sample: np.ndarray,
    ref_sample: np.ndarray,
    fs: float,
    nperseg: int = 128,
) -> np.ndarray:
    """Build one resonance map [C, C, 2] containing PLV and coherence."""
    q_cl = to_channel_series(query_sample)
    r_cl = to_channel_series(ref_sample)
    if q_cl.shape[0] != r_cl.shape[0]:
        raise ValueError(
            f"Channel mismatch for resonance map: {q_cl.shape[0]} vs {r_cl.shape[0]}"
        )
    plv = compute_plv_matrix(q_cl, r_cl)
    coh = compute_coherence_matrix(q_cl, r_cl, fs=fs, nperseg=nperseg)
    return np.stack([plv, coh], axis=-1).astype(np.float32)


def build_resonance_tensor(
    batch_samples: np.ndarray,
    ref_samples: np.ndarray,
    fs: float,
    nperseg: int = 128,
) -> np.ndarray:
    """Build resonance tensor for one batch.

    batch_samples: [B, ...]
    ref_samples: [B, K, ...]
    returns: [B, K, C, C, 2]
    """
    b = batch_samples.shape[0]
    k = ref_samples.shape[1]
    maps = []
    for i in range(b):
        per_ref = []
        for j in range(k):
            per_ref.append(build_resonance_map(batch_samples[i], ref_samples[i, j], fs=fs, nperseg=nperseg))
        maps.append(np.stack(per_ref, axis=0))
    return np.stack(maps, axis=0)
