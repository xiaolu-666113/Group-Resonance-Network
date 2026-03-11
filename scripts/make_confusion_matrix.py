#!/usr/bin/env python3
"""Generate confusion matrix figure (PNG + PDF)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_json
from src.utils.plotting import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create confusion matrix figure")
    parser.add_argument("--input", type=str, required=True, help="Path to metrics json or predictions npz")
    parser.add_argument("--out-dir", type=str, default="outputs/figures")
    parser.add_argument("--class-names", type=str, default="", help="Comma-separated class names")
    parser.add_argument("--normalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".json":
        data = load_json(in_path)
        if "confusion_matrix" not in data:
            raise KeyError("JSON must contain confusion_matrix")
        conf = np.asarray(data["confusion_matrix"], dtype=np.int64)
    elif in_path.suffix.lower() == ".npz":
        data = np.load(in_path)
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        conf = confusion_matrix(y_true, y_pred, labels=labels)
    else:
        raise ValueError("Input must be .json or .npz")

    if args.class_names:
        class_names = [x.strip() for x in args.class_names.split(",")]
    else:
        class_names = [str(i) for i in range(conf.shape[0])]

    plot_confusion_matrix(
        conf_mat=conf,
        class_names=class_names,
        out_png=out_dir / "confusion_matrix.png",
        out_pdf=out_dir / "confusion_matrix.pdf",
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
