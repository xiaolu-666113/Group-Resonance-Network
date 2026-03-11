#!/usr/bin/env python3
"""Generate training curve figure (PNG + PDF)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.plotting import plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create training curve figure")
    parser.add_argument("--history-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_path = Path(args.history_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    with history_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]) + 1)
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))

    plot_training_curves(
        epochs=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        train_acc=train_acc,
        val_acc=val_acc,
        out_png=out_dir / "training_curves.png",
        out_pdf=out_dir / "training_curves.pdf",
    )


if __name__ == "__main__":
    main()
