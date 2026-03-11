"""Protocol split utilities for EEG experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class FoldSplit:
    """Train/validation/test split by global indices."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    fold_name: str


def _safe_train_test_split(
    indices: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(indices) < 2:
        return indices, np.array([], dtype=np.int64)

    uniq = np.unique(labels)
    stratify = labels if len(uniq) > 1 else None
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        # Fallback when per-class sample count is too small for stratification.
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def split_train_val(
    train_pool_idx: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split train pool into train/validation."""
    if val_ratio <= 0:
        return np.asarray(train_pool_idx, dtype=np.int64), np.array([], dtype=np.int64)
    train_labels = labels[train_pool_idx]
    return _safe_train_test_split(train_pool_idx, train_labels, test_size=val_ratio, seed=seed)


def build_subject_dependent_split(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> FoldSplit:
    """Build one subject-dependent split by splitting within each subject."""
    subject_ids = np.asarray(subject_ids)
    labels = np.asarray(labels)

    tr_all: list[np.ndarray] = []
    te_all: list[np.ndarray] = []

    for subj in np.unique(subject_ids):
        idx = np.where(subject_ids == subj)[0]
        tr, te = _safe_train_test_split(idx, labels[idx], test_size=test_ratio, seed=seed)
        tr_all.append(tr)
        te_all.append(te)

    train_pool = np.concatenate(tr_all) if tr_all else np.array([], dtype=np.int64)
    test_idx = np.concatenate(te_all) if te_all else np.array([], dtype=np.int64)
    train_idx, val_idx = split_train_val(train_pool, labels, val_ratio=val_ratio, seed=seed)

    return FoldSplit(
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
        fold_name="sd",
    )


def build_loso_splits(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> list[FoldSplit]:
    """Build LOSO folds (leave-one-subject-out)."""
    subject_ids = np.asarray(subject_ids)
    labels = np.asarray(labels)

    folds: list[FoldSplit] = []
    for subj in np.unique(subject_ids):
        test_idx = np.where(subject_ids == subj)[0]
        train_pool = np.where(subject_ids != subj)[0]
        train_idx, val_idx = split_train_val(train_pool, labels, val_ratio=val_ratio, seed=seed)
        folds.append(
            FoldSplit(
                train_idx=np.asarray(train_idx, dtype=np.int64),
                val_idx=np.asarray(val_idx, dtype=np.int64),
                test_idx=np.asarray(test_idx, dtype=np.int64),
                fold_name=f"loso_subject_{int(subj)}",
            )
        )
    return folds
