#!/usr/bin/env python3
"""Run complete LOSO protocol and aggregate fold metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.samplers import build_loso_splits
from src.trainers.trainer import run_fold
from src.utils.config import apply_overrides, load_config
from src.utils.io import save_json
from src.utils.metrics import summarize_fold_metrics
from src.utils.seed import set_seed
from src.utils.factory import build_dataset_from_config, create_experiment_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LOSO experiment for GRN")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--override", type=str, nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.override)

    seed = int(config["experiment"].get("seed", 42))
    deterministic = bool(config["training"].get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    dataset = build_dataset_from_config(config)
    device = resolve_device(args.device)
    run_dir = create_experiment_dir(config, tag="loso")
    save_json(config, run_dir / "resolved_config.json")

    folds = build_loso_splits(
        subject_ids=dataset.subject_id,
        labels=dataset.y,
        val_ratio=float(config["dataset"].get("val_ratio", 0.1)),
        seed=seed,
    )
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    fold_results = []
    test_metrics_list = []

    for i, split in enumerate(folds):
        fold_seed = seed + i
        set_seed(fold_seed, deterministic=deterministic)
        result = run_fold(
            config=config,
            dataset=dataset,
            split=split,
            run_dir=run_dir,
            device=device,
            resume_path=config["training"].get("resume") or None,
        )
        fold_results.append(result.__dict__)
        test_metrics_list.append(result.test_metrics)

    summary_stats = summarize_fold_metrics(test_metrics_list)
    summary = {
        "protocol": "loso",
        "num_folds": len(fold_results),
        "folds": fold_results,
        "summary": summary_stats,
        "run_dir": str(run_dir),
    }

    save_json(summary, run_dir / "summary.json")

    csv_file = run_dir / "fold_metrics.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "accuracy", "macro_f1"])
        writer.writeheader()
        for fr in fold_results:
            writer.writerow(
                {
                    "fold": fr["fold_name"],
                    "accuracy": fr["test_metrics"]["accuracy"],
                    "macro_f1": fr["test_metrics"]["macro_f1"],
                }
            )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
