"""Factory helpers for dataset/model experiment setup."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch

from src.datasets.deap_dataset import DEAPDataset
from src.datasets.seed_dataset import SEEDDataset


def build_dataset_from_config(config: Dict[str, Any]):
    """Instantiate dataset adapter from config."""
    dcfg = config["dataset"]
    name = str(dcfg.get("name", "seed")).lower()
    data_path = dcfg["data_path"]
    input_mode = str(dcfg.get("input_mode", "features"))

    if name == "seed":
        return SEEDDataset.from_npz(data_path, input_mode=input_mode)
    if name == "deap":
        task = str(dcfg.get("deap_task", "valence"))
        threshold = float(dcfg.get("deap_threshold", 5.0))
        return DEAPDataset.from_npz(data_path, task=task, threshold=threshold, input_mode=input_mode)
    raise ValueError(f"Unsupported dataset name: {name}")


def resolve_device(device_arg: str | None = None) -> torch.device:
    """Resolve torch device from CLI/config."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_experiment_dir(config: Dict[str, Any], tag: str | None = None) -> Path:
    """Create output experiment directory."""
    exp_cfg = config.get("experiment", {})
    name = str(exp_cfg.get("name", "grn_experiment"))
    out_root = Path(exp_cfg.get("output_root", "outputs"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    run_dir = out_root / f"{name}{suffix}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
