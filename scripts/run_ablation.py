#!/usr/bin/env python3
"""Run ablation experiments for GRN variants."""

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


ABLATIONS = {
    "individual_only": {
        "model": {"use_prototypes": False, "use_resonance": False},
        "loss": {"lambda_proto": 0.0},
    },
    "prototypes_only": {
        "model": {"use_prototypes": True, "use_resonance": False},
    },
    "resonance_only": {
        "model": {"use_prototypes": False, "use_resonance": True},
        "loss": {"lambda_proto": 0.0},
    },
    "full_model": {
        "model": {"use_prototypes": True, "use_resonance": True},
    },
    "full_no_proto_reg": {
        "model": {"use_prototypes": True, "use_resonance": True},
        "loss": {"lambda_proto": 0.0},
    },
}


def deep_merge(base: dict, update: dict) -> dict:
    out = deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRN ablation study")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--override", type=str, nargs="*", default=[])
    parser.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=list(ABLATIONS.keys()),
        help="Ablation variants to run",
    )
    parser.add_argument("--max-folds", type=int, default=None, help="Only for LOSO protocol")
    return parser.parse_args()


def run_one_variant(base_cfg: dict, variant: str, device, max_folds: int | None) -> dict:
    cfg = deep_merge(base_cfg, ABLATIONS[variant])
    cfg["experiment"]["name"] = f"{base_cfg['experiment'].get('name', 'grn')}_{variant}"

    seed = int(cfg["experiment"].get("seed", 42))
    deterministic = bool(cfg["training"].get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    dataset = build_dataset_from_config(cfg)
    run_dir = create_experiment_dir(cfg, tag=variant)
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
        if max_folds is not None:
            folds = folds[:max_folds]

        for i, split in enumerate(folds):
            set_seed(seed + i, deterministic=deterministic)
            res = run_fold(cfg, dataset, split, run_dir=run_dir, device=device)
            fold_results.append(res.__dict__)
            test_metrics.append(res.test_metrics)
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    summary = summarize_fold_metrics(test_metrics)
    out = {
        "variant": variant,
        "protocol": protocol,
        "run_dir": str(run_dir),
        "folds": fold_results,
        "summary": summary,
    }
    save_json(out, run_dir / "summary.json")
    return out


def main() -> None:
    args = parse_args()
    base_cfg = apply_overrides(load_config(args.config), args.override)
    device = resolve_device(args.device)

    results = []
    for variant in args.variants:
        if variant not in ABLATIONS:
            raise ValueError(f"Unknown ablation variant: {variant}")
        results.append(run_one_variant(base_cfg, variant, device, args.max_folds))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
