"""Plotting utilities for publication-ready figures."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _setup_plot_style(font_size: int = 16) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "figure.dpi": 150,
        }
    )


def plot_confusion_matrix(
    conf_mat: np.ndarray,
    class_names: Sequence[str],
    out_png: str | Path,
    out_pdf: str | Path,
    normalize: bool = False,
) -> None:
    """Render confusion matrix with large fonts and no figure title."""
    _setup_plot_style(font_size=18)
    matrix = conf_mat.astype(np.float64)
    if normalize:
        row_sum = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / np.maximum(row_sum, 1e-12)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    thresh = matrix.max() / 2.0 if matrix.size > 0 else 0.0
    fmt = ".2f" if normalize else "d"
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = format(val, fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
            )

    fig.tight_layout()
    out_png = Path(out_png)
    out_pdf = Path(out_pdf)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    epochs: Sequence[int],
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    train_acc: Sequence[float],
    val_acc: Sequence[float],
    out_png: str | Path,
    out_pdf: str | Path,
) -> None:
    """Render train/validation loss and accuracy curves."""
    _setup_plot_style(font_size=16)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, train_acc, label="Train Acc", linewidth=2)
    axes[1].plot(epochs, val_acc, label="Val Acc", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    out_png = Path(out_png)
    out_pdf = Path(out_pdf)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
