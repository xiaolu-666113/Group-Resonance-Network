#!/usr/bin/env python3
"""Evaluate a trained checkpoint without additional training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.reference_selector import ReferenceSelectionConfig, ReferenceSelector
from src.datasets.samplers import build_loso_splits, build_subject_dependent_split
from src.models.grn import GRNModel
from src.trainers.trainer import GRNTrainer, ResonanceProvider
from src.utils.config import apply_overrides, load_config
from src.utils.io import load_checkpoint, save_json
from src.utils.seed import set_seed
from src.utils.factory import build_dataset_from_config, create_experiment_dir, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GRN checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--override", type=str, nargs="*", default=[])
    parser.add_argument("--protocol", type=str, default=None, choices=["sd", "loso"])
    parser.add_argument("--test-subject", type=int, default=None, help="Required for LOSO single subject evaluation")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    return parser.parse_args()


def build_model(config: dict, num_classes: int) -> GRNModel:
    mcfg = config["model"]
    return GRNModel(
        num_classes=num_classes,
        embedding_dim=int(mcfg.get("embedding_dim", 256)),
        encoder_hidden_dim=int(mcfg.get("encoder_hidden_dim", 256)),
        encoder_transformer_layers=int(mcfg.get("transformer_layers", 2)),
        encoder_transformer_heads=int(mcfg.get("transformer_heads", 4)),
        dropout=float(mcfg.get("dropout", 0.2)),
        num_prototypes=int(mcfg.get("prototypes", 8)),
        prototype_temperature=float(mcfg.get("temperature", 0.07)),
        prototype_similarity=str(mcfg.get("prototype_similarity", "cosine")),
        resonance_channels=int(mcfg.get("resonance_channels", 32)),
        fusion_hidden_dim=int(mcfg.get("fusion_hidden_dim", 512)),
        use_prototypes=bool(mcfg.get("use_prototypes", True)),
        use_resonance=bool(mcfg.get("use_resonance", True),),
    )


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.override)

    seed = int(config["experiment"].get("seed", 42))
    deterministic = bool(config["training"].get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    dataset = build_dataset_from_config(config)
    protocol = args.protocol or str(config["dataset"].get("protocol", "sd")).lower()

    if protocol == "sd":
        split = build_subject_dependent_split(
            subject_ids=dataset.subject_id,
            labels=dataset.y,
            test_ratio=float(config["dataset"].get("test_ratio", 0.2)),
            val_ratio=float(config["dataset"].get("val_ratio", 0.1)),
            seed=seed,
        )
    else:
        folds = build_loso_splits(
            subject_ids=dataset.subject_id,
            labels=dataset.y,
            val_ratio=float(config["dataset"].get("val_ratio", 0.1)),
            seed=seed,
        )
        if args.test_subject is None:
            raise ValueError("--test-subject is required when protocol=loso")
        matched = [f for f in folds if int(f.fold_name.split("_")[-1]) == int(args.test_subject)]
        if not matched:
            raise ValueError(f"Subject {args.test_subject} not found in LOSO folds")
        split = matched[0]

    device = resolve_device(args.device)
    model = build_model(config, num_classes=dataset.num_classes).to(device)
    state = load_checkpoint(args.checkpoint, map_location=device.type)
    model.load_state_dict(state["model"])

    # Build evaluator wrapper with dummy optimizer (not used in evaluation mode).
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = GRNTrainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=None,
        lambda_proto=float(config["loss"].get("lambda_proto", 0.1)),
        use_proto_regularizer=bool(config["model"].get("use_prototypes", True)),
        entropy_weight=float(config["loss"].get("prototype_entropy_weight", 0.01)),
        mixed_precision=bool(config["training"].get("mixed_precision", False)),
        grad_clip_norm=0.0,
        logger=None,
        output_dir=Path("."),
    )

    # Build dataloader manually.
    from torch.utils.data import DataLoader, Subset

    idx = split.test_idx if args.split == "test" else split.val_idx
    if len(idx) == 0:
        raise ValueError(f"Requested split {args.split} is empty")

    loader = DataLoader(
        Subset(dataset, idx.tolist()),
        batch_size=int(config["training"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(config["dataset"].get("num_workers", 0)),
        drop_last=False,
    )

    provider = None
    if bool(config["model"].get("use_resonance", True)):
        rcfg = config["resonance"]
        strategy = str(rcfg.get("selector_strategy", "random"))
        features = dataset.flat_features() if strategy == "nearest" else None
        forbidden = set(dataset.get_subject_ids(split.test_idx).tolist()) if protocol == "loso" else set()
        selector = ReferenceSelector(
            allowed_indices=split.train_idx,
            labels=dataset.y,
            subject_ids=dataset.subject_id,
            config=ReferenceSelectionConfig(
                strategy=strategy,
                k_refs=int(rcfg.get("k_refs", 3)),
                seed=seed,
            ),
            features=features,
            forbidden_subjects=forbidden,
        )
        provider = ResonanceProvider(
            dataset=dataset,
            selector=selector,
            mode=str(rcfg.get("mode", "precompute")),
            fs=float(rcfg.get("fs", config["dataset"].get("sampling_rate", 200))),
            coherence_nperseg=int(rcfg.get("coherence_nperseg", 128)),
        )
        if str(rcfg.get("mode", "precompute")).lower() == "precompute":
            provider.prepare_cache(np.asarray(idx, dtype=np.int64), show_progress=True)

    metrics = trainer.evaluate(loader, resonance_provider=provider, desc=f"Eval-{args.split}")
    metrics.pop("y_true", None)
    metrics.pop("y_pred", None)

    run_dir = create_experiment_dir(config, tag="eval")
    save_json(metrics, run_dir / f"{args.split}_metrics.json")
    print(json.dumps({"split": args.split, "metrics": metrics, "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
