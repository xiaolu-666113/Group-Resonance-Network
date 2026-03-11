"""Metrics utilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


EPS = 1e-8


def per_class_accuracy(conf_mat: np.ndarray) -> np.ndarray:
    """Compute per-class accuracy from confusion matrix."""
    class_totals = conf_mat.sum(axis=1)
    correct = np.diag(conf_mat)
    return correct / np.maximum(class_totals, EPS)


def classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: list[int] | None = None,
) -> Dict[str, Any]:
    """Compute core classification metrics for experiments."""
    y_true_np = np.asarray(list(y_true), dtype=np.int64)
    y_pred_np = np.asarray(list(y_pred), dtype=np.int64)
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true_np, y_pred_np])).tolist())

    conf_mat = confusion_matrix(y_true_np, y_pred_np, labels=labels)
    per_cls = per_class_accuracy(conf_mat)

    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, labels=labels, average="macro")),
        "per_class_accuracy": per_cls.tolist(),
        "confusion_matrix": conf_mat.tolist(),
    }


def summarize_fold_metrics(metrics_list: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate fold-level metrics into mean/std summary."""
    if not metrics_list:
        raise ValueError("metrics_list must not be empty")

    acc = np.array([m["accuracy"] for m in metrics_list], dtype=np.float64)
    f1 = np.array([m["macro_f1"] for m in metrics_list], dtype=np.float64)
    per_class = np.array([m["per_class_accuracy"] for m in metrics_list], dtype=np.float64)
    conf = np.array([m["confusion_matrix"] for m in metrics_list], dtype=np.float64)

    return {
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1) if len(acc) > 1 else 0.0),
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_std": float(f1.std(ddof=1) if len(f1) > 1 else 0.0),
        "per_class_accuracy_mean": per_class.mean(axis=0).tolist(),
        "per_class_accuracy_std": (per_class.std(axis=0, ddof=1) if len(per_class) > 1 else np.zeros_like(per_class[0])).tolist(),
        "confusion_matrix_sum": conf.sum(axis=0).astype(int).tolist(),
    }
