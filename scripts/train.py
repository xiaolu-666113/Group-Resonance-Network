#!/usr/bin/env python3
"""Train GRN with subject-dependent or single-fold LOSO protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.samplers import build_loso_splits, build_subject_dependent_split
from src.trainers.trainer import run_fold
from src.utils.config import apply_overrides, load_config
from src.utils.io import save_json
from src.utils.seed import set_seed
from src.utils.factory import build_dataset_from_config, create_experiment_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Group Resonance Network")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Config overrides in key=value format",
    )
    parser.add_argument(
        "--test-subject",
        type=int,
        default=None,
        help="When protocol=loso, choose one test subject id",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    seed = int(config["experiment"].get("seed", 42))
    deterministic = bool(config["training"].get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    dataset = build_dataset_from_config(config)
    device = resolve_device(args.device)
    protocol = str(config["dataset"].get("protocol", "sd")).lower()
    run_dir = create_experiment_dir(config)

    save_json(config, run_dir / "resolved_config.json")

    if protocol == "sd":
        split = build_subject_dependent_split(
            subject_ids=dataset.subject_id,
            labels=dataset.y,
            test_ratio=float(config["dataset"].get("test_ratio", 0.2)),
            val_ratio=float(config["dataset"].get("val_ratio", 0.1)),
            seed=seed,
        )
        result = run_fold(
            config=config,
            dataset=dataset,
            split=split,
            run_dir=run_dir,
            device=device,
            resume_path=config["training"].get("resume") or None,
        )
        summary = {
            "protocol": "sd",
            "folds": [result.__dict__],
            "run_dir": str(run_dir),
        }
        save_json(summary, run_dir / "summary.json")
        print(json.dumps(summary, indent=2))
        return

    if protocol == "loso":
        folds = build_loso_splits(
            subject_ids=dataset.subject_id,
            labels=dataset.y,
            val_ratio=float(config["dataset"].get("val_ratio", 0.1)),
            seed=seed,
        )
        if args.test_subject is not None:
            selected = []
            for f in folds:
                # fold name format: loso_subject_{id}
                subj = int(f.fold_name.split("_")[-1])
                if subj == int(args.test_subject):
                    selected.append(f)
            if not selected:
                raise ValueError(f"Subject {args.test_subject} not found in LOSO folds")
            folds = selected
        else:
            folds = folds[:1]

        fold_results = []
        for i, split in enumerate(folds):
            fold_seed = seed + i
            set_seed(fold_seed, deterministic=deterministic)
            out = run_fold(
                config=config,
                dataset=dataset,
                split=split,
                run_dir=run_dir,
                device=device,
                resume_path=config["training"].get("resume") or None,
            )
            fold_results.append(out.__dict__)

        summary = {
            "protocol": "loso",
            "num_folds": len(fold_results),
            "folds": fold_results,
            "run_dir": str(run_dir),
        }
        save_json(summary, run_dir / "summary.json")
        print(json.dumps(summary, indent=2))
        return

    raise ValueError(f"Unsupported protocol: {protocol}")


if __name__ == "__main__":
    main()
