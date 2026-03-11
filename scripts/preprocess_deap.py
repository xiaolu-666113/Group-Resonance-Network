#!/usr/bin/env python3
"""Preprocess raw DEAP EEG into cached DE features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.preprocessing import preprocess_raw_npz_to_de


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw DEAP data")
    parser.add_argument("--input", type=str, required=True, help="Raw DEAP npz path")
    parser.add_argument("--output", type=str, required=True, help="Output DE npz path")
    parser.add_argument("--fs", type=float, default=128.0)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = preprocess_raw_npz_to_de(
        input_path=Path(args.input),
        output_path=Path(args.output),
        fs=args.fs,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        overwrite=args.overwrite,
    )
    print(f"Saved preprocessed DEAP features to: {out}")


if __name__ == "__main__":
    main()
