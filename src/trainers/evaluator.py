"""Evaluation helpers for trained checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
import torch

from src.datasets.base_dataset import BaseEEGDataset
from src.datasets.reference_selector import ReferenceSelectionConfig, ReferenceSelector
from src.datasets.samplers import FoldSplit
from src.trainers.trainer import GRNTrainer, ResonanceProvider, _create_model, _make_loader
from src.utils.io import load_checkpoint


def evaluate_checkpoint(
    config: Dict[str, Any],
    dataset: BaseEEGDataset,
    split: FoldSplit,
    checkpoint_path: str | Path,
    device: torch.device,
    split_name: Literal["val", "test"] = "test",
) -> Dict[str, Any]:
    """Evaluate a saved checkpoint on validation or test split.

    This function does not run any additional training.
    """
    target_idx = split.test_idx if split_name == "test" else split.val_idx
    if len(target_idx) == 0:
        raise ValueError(f"Requested split '{split_name}' is empty")

    model = _create_model(config, num_classes=dataset.num_classes).to(device)
    state = load_checkpoint(checkpoint_path, map_location=device.type)
    model.load_state_dict(state["model"])

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

    loader = _make_loader(
        dataset=dataset,
        indices=target_idx,
        batch_size=int(config["training"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(config["dataset"].get("num_workers", 0)),
        seed=int(config["experiment"].get("seed", 42)),
    )

    provider = None
    if bool(config["model"].get("use_resonance", True)):
        rcfg = config["resonance"]
        strategy = str(rcfg.get("selector_strategy", "random")).lower()
        features = dataset.flat_features() if strategy == "nearest" else None

        protocol = str(config["dataset"].get("protocol", "sd")).lower()
        forbidden_subjects = (
            set(dataset.get_subject_ids(split.test_idx).tolist()) if protocol == "loso" else set()
        )
        selector = ReferenceSelector(
            allowed_indices=split.train_idx,
            labels=dataset.y,
            subject_ids=dataset.subject_id,
            config=ReferenceSelectionConfig(
                strategy=strategy,
                k_refs=int(rcfg.get("k_refs", 3)),
                seed=int(config["experiment"].get("seed", 42)),
            ),
            features=features,
            forbidden_subjects=forbidden_subjects,
        )
        provider = ResonanceProvider(
            dataset=dataset,
            selector=selector,
            mode=str(rcfg.get("mode", "precompute")),
            fs=float(rcfg.get("fs", config["dataset"].get("sampling_rate", 200))),
            coherence_nperseg=int(rcfg.get("coherence_nperseg", 128)),
        )
        if str(rcfg.get("mode", "precompute")).lower() == "precompute":
            provider.prepare_cache(np.asarray(target_idx, dtype=np.int64), show_progress=True)

    metrics = trainer.evaluate(loader=loader, resonance_provider=provider, desc=f"Eval-{split_name}")
    return metrics
