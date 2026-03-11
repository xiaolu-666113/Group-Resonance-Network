#!/usr/bin/env python3
"""Quick sanity checks for core GRN modules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.reference_selector import (
    ReferenceSelectionConfig,
    ReferenceSelector,
    assert_reference_leakage_free,
)
from src.models.grn import GRNModel
from src.utils.signal_ops import build_resonance_map


def main() -> None:
    print("[1/3] Model forward shape check")
    model = GRNModel(num_classes=3, use_prototypes=True, use_resonance=True)
    x = torch.randn(4, 10, 32, 5)
    m_res = torch.randn(4, 3, 32, 32, 2)
    out = model(x, m_res)
    assert out["logits"].shape == (4, 3)
    assert out["F"].shape[0] == 4

    print("[2/3] Reference leakage check")
    subject_ids = np.array([0, 0, 1, 1, 2, 2])
    labels = np.array([0, 1, 0, 1, 0, 1])
    selector = ReferenceSelector(
        allowed_indices=np.array([0, 1, 2, 3]),
        labels=labels,
        subject_ids=subject_ids,
        config=ReferenceSelectionConfig(strategy="random", k_refs=2, seed=42),
        forbidden_subjects={2},
    )
    refs = selector.select(np.array([4, 5]))
    assert_reference_leakage_free(refs, subject_ids, forbidden_subjects={2})

    print("[3/3] Resonance map check")
    q = np.random.randn(10, 32, 5).astype(np.float32)
    r = np.random.randn(10, 32, 5).astype(np.float32)
    m = build_resonance_map(q, r, fs=200.0, nperseg=64)
    assert m.shape == (32, 32, 2)
    assert np.all(np.isfinite(m))

    print("Sanity check passed.")


if __name__ == "__main__":
    main()
