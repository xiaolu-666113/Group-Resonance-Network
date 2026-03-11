"""Leakage safety tests for reference selection."""

from __future__ import annotations

import numpy as np

from src.datasets.reference_selector import (
    ReferenceSelectionConfig,
    ReferenceSelector,
    assert_reference_leakage_free,
)


def test_reference_selector_no_forbidden_subjects() -> None:
    subject_ids = np.array([0, 0, 1, 1, 2, 2])
    labels = np.array([0, 1, 0, 1, 0, 1])
    allowed_indices = np.array([0, 1, 2, 3])

    selector = ReferenceSelector(
        allowed_indices=allowed_indices,
        labels=labels,
        subject_ids=subject_ids,
        config=ReferenceSelectionConfig(strategy="random", k_refs=3, seed=7),
        forbidden_subjects={2},
    )

    refs = selector.select(np.array([4, 5]))
    assert refs.shape == (2, 3)
    assert_reference_leakage_free(refs, subject_ids=subject_ids, forbidden_subjects={2})


def test_nearest_selector_shapes() -> None:
    n = 10
    subject_ids = np.array([0] * 5 + [1] * 5)
    labels = np.array([0, 1] * 5)
    features = np.random.randn(n, 8).astype(np.float32)

    selector = ReferenceSelector(
        allowed_indices=np.arange(5),
        labels=labels,
        subject_ids=subject_ids,
        config=ReferenceSelectionConfig(strategy="nearest", k_refs=2, seed=13),
        features=features,
        forbidden_subjects={1},
    )

    refs = selector.select(np.array([6, 7]))
    assert refs.shape == (2, 2)
