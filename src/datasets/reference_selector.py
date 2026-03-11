"""Reference sample selection with leakage safeguards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ReferenceSelectionConfig:
    """Config for reference selection."""

    strategy: str = "random"
    k_refs: int = 3
    seed: int = 42


class ReferenceSelector:
    """Select training-only references for each query sample.

    Supported strategies:
    - random
    - class_balanced
    - nearest
    """

    def __init__(
        self,
        allowed_indices: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray,
        config: ReferenceSelectionConfig,
        features: np.ndarray | None = None,
        forbidden_subjects: Iterable[int] | None = None,
    ) -> None:
        self.allowed_indices = np.asarray(allowed_indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.subject_ids = np.asarray(subject_ids, dtype=np.int64)
        self.config = config
        self.strategy = config.strategy.lower()
        self.k_refs = int(config.k_refs)
        self.seed = int(config.seed)
        self.features = features
        self.forbidden_subjects = set(int(x) for x in (forbidden_subjects or []))

        if len(self.allowed_indices) == 0:
            raise ValueError("Reference pool cannot be empty")

        if self.strategy == "nearest" and self.features is None:
            raise ValueError("Nearest strategy requires feature matrix [N, D]")

        # Class index mapping for class-balanced sampling.
        self.class_to_indices: dict[int, np.ndarray] = {}
        for cls in np.unique(self.labels[self.allowed_indices]):
            cls_idx = self.allowed_indices[self.labels[self.allowed_indices] == cls]
            self.class_to_indices[int(cls)] = cls_idx

        self._assert_forbidden_excluded(self.allowed_indices)

    def _rng(self, query_idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + int(query_idx) * 10007)

    def _exclude_query(self, pool: np.ndarray, query_idx: int) -> np.ndarray:
        return pool[pool != query_idx]

    def _sample_random(self, pool: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        if len(pool) == 0:
            pool = self.allowed_indices
        replace = len(pool) < k
        return rng.choice(pool, size=k, replace=replace).astype(np.int64)

    def _sample_class_balanced(self, query_idx: int, k: int, rng: np.random.Generator) -> np.ndarray:
        classes = sorted(self.class_to_indices.keys())
        if not classes:
            return self._sample_random(self.allowed_indices, k, rng)

        per_class = max(1, int(np.ceil(k / len(classes))))
        refs: list[int] = []
        for cls in classes:
            pool = self._exclude_query(self.class_to_indices[cls], query_idx)
            if len(pool) == 0:
                continue
            take = min(per_class, len(pool))
            picked = rng.choice(pool, size=take, replace=False)
            refs.extend(picked.tolist())

        if len(refs) < k:
            extra_pool = self._exclude_query(self.allowed_indices, query_idx)
            extra = self._sample_random(extra_pool, k - len(refs), rng)
            refs.extend(extra.tolist())

        refs = np.asarray(refs[:k], dtype=np.int64)
        return refs

    def _sample_nearest(self, query_idx: int, k: int) -> np.ndarray:
        assert self.features is not None
        pool = self._exclude_query(self.allowed_indices, query_idx)
        if len(pool) == 0:
            pool = self.allowed_indices

        query_vec = self.features[query_idx]
        pool_vecs = self.features[pool]
        dists = np.linalg.norm(pool_vecs - query_vec[None, :], axis=1)
        order = np.argsort(dists)
        top = pool[order[:k]]
        if len(top) < k:
            pad = np.resize(top, k)
            return pad.astype(np.int64)
        return top.astype(np.int64)

    def select(self, query_indices: np.ndarray) -> np.ndarray:
        """Select reference indices for each query.

        query_indices: [B]
        returns: [B, K]
        """
        q = np.asarray(query_indices, dtype=np.int64)
        selected = np.zeros((len(q), self.k_refs), dtype=np.int64)

        for i, idx in enumerate(q):
            rng = self._rng(int(idx))
            if self.strategy == "random":
                pool = self._exclude_query(self.allowed_indices, int(idx))
                refs = self._sample_random(pool, self.k_refs, rng)
            elif self.strategy == "class_balanced":
                refs = self._sample_class_balanced(int(idx), self.k_refs, rng)
            elif self.strategy == "nearest":
                refs = self._sample_nearest(int(idx), self.k_refs)
            else:
                raise ValueError(f"Unknown reference strategy: {self.strategy}")

            self._assert_forbidden_excluded(refs)
            selected[i] = refs

        return selected

    def _assert_forbidden_excluded(self, refs: np.ndarray) -> None:
        if not self.forbidden_subjects:
            return
        ref_subjects = set(self.subject_ids[np.asarray(refs, dtype=np.int64)].tolist())
        overlap = ref_subjects.intersection(self.forbidden_subjects)
        if overlap:
            raise AssertionError(
                f"Reference leakage detected for forbidden subjects: {sorted(overlap)}"
            )


def assert_reference_leakage_free(
    reference_indices: np.ndarray,
    subject_ids: np.ndarray,
    forbidden_subjects: Iterable[int],
) -> None:
    """Sanity-check helper for tests and scripts."""
    refs = np.asarray(reference_indices, dtype=np.int64).reshape(-1)
    subjects = set(np.asarray(subject_ids, dtype=np.int64)[refs].tolist())
    forbidden = set(int(x) for x in forbidden_subjects)
    overlap = subjects.intersection(forbidden)
    if overlap:
        raise AssertionError(f"Leakage found for subjects: {sorted(overlap)}")
