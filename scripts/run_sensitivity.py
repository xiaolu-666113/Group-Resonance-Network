#!/usr/bin/env python3
"""Run sensitivity analysis over K_r and prototype count M."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.samplers import build_loso_splits, build_subject_dependent_split
from src.trainers.trainer import run_fold
from src.utils.config import apply_overrides, load_config
from src.utils.io import save_json
from src.utils.metrics import summarize_fold_metrics
from src.utils.seed import set_seed
from src.utils.factory import build_dataset_from_config, create_experiment_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sensitivity analysis for GRN")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--override", type=str, nargs="*", default=[])
    parser.add_argument("--k-refs", type=int, nargs="*", default=[1, 3, 5])
    parser.add_argument("--prototypes", type=int, nargs="*", default=[4, 8, 12])
    parser.add_argument("--max-folds", type=int, default=None, help="Only for LOSO protocol")
    return parser.parse_args()


def deep_set(cfg: dict, keys: list[str], value):
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def main() -> None:
    args = parse_args()
    base_cfg = apply_overrides(load_config(args.config), args.override)
    device = resolve_device(args.device)

    all_results = []

    for k in args.k_refs:
        for m in args.prototypes:
            cfg = deepcopy(base_cfg)
            deep_set(cfg, ["resonance", "k_refs"], int(k))
            deep_set(cfg, ["model", "prototypes"], int(m))
            cfg["experiment"]["name"] = f"{base_cfg['experiment'].get('name', 'grn')}_kr{k}_m{m}"

            seed = int(cfg["experiment"].get("seed", 42))
            deterministic = bool(cfg["training"].get("deterministic", True))
            set_seed(seed, deterministic=deterministic)

            dataset = build_dataset_from_config(cfg)
            run_dir = create_experiment_dir(cfg, tag=f"kr{k}_m{m}")
            save_json(cfg, run_dir / "resolved_config.json")

            protocol = str(cfg["dataset"].get("protocol", "sd")).lower()
            fold_results = []
            test_metrics = []

            if protocol == "sd":
                split = build_subject_dependent_split(
                    subject_ids=dataset.subject_id,
                    labels=dataset.y,
                    test_ratio=float(cfg["dataset"].get("test_ratio", 0.2)),
                    val_ratio=float(cfg["dataset"].get("val_ratio", 0.1)),
                    seed=seed,
                )
                res = run_fold(cfg, dataset, split, run_dir=run_dir, device=device)
                fold_results.append(res.__dict__)
                test_metrics.append(res.test_metrics)
            elif protocol == "loso":
                folds = build_loso_splits(
                    subject_ids=dataset.subject_id,
                    labels=dataset.y,
                    val_ratio=float(cfg["dataset"].get("val_ratio", 0.1)),
                    seed=seed,
                )
                if args.max_folds is not None:
                    folds = folds[: args.max_folds]

                for i, split in enumerate(folds):
                    set_seed(seed + i, deterministic=deterministic)
                    res = run_fold(cfg, dataset, split, run_dir=run_dir, device=device)
                    fold_results.append(res.__dict__)
                    test_metrics.append(res.test_metrics)
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")

            summary = summarize_fold_metrics(test_metrics)
            item = {
                "k_refs": int(k),
                "prototypes": int(m),
                "run_dir": str(run_dir),
                "summary": summary,
                "folds": fold_results,
            }
            save_json(item, run_dir / "summary.json")
            all_results.append(item)

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
