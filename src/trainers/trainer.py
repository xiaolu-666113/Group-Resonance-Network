"""Training pipeline for Group Resonance Network."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.datasets.base_dataset import BaseEEGDataset
from src.datasets.reference_selector import ReferenceSelectionConfig, ReferenceSelector
from src.datasets.samplers import FoldSplit
from src.losses.classification import ClassificationLoss
from src.losses.prototype_loss import PrototypeRegularizationLoss
from src.models.grn import GRNModel
from src.utils.io import ensure_dir, load_checkpoint, save_checkpoint, save_json
from src.utils.logging import create_logger
from src.utils.metrics import classification_metrics
from src.utils.seed import seed_worker
from src.utils.signal_ops import build_resonance_tensor


@dataclass
class FoldResult:
    """Result bundle for one fold."""

    fold_name: str
    best_epoch: int
    best_val_accuracy: float
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    best_checkpoint: str
    last_checkpoint: str
    history_file: str
    predictions_file: str


class ResonanceProvider:
    """Provide leakage-safe resonance tensors either online or precomputed."""

    def __init__(
        self,
        dataset: BaseEEGDataset,
        selector: ReferenceSelector,
        mode: str,
        fs: float,
        coherence_nperseg: int,
    ) -> None:
        self.dataset = dataset
        self.selector = selector
        self.mode = mode.lower()
        self.fs = float(fs)
        self.coherence_nperseg = int(coherence_nperseg)
        self.cache: dict[int, np.ndarray] = {}

        if self.mode not in {"online", "precompute"}:
            raise ValueError(f"Unsupported resonance mode: {self.mode}")

    def _compute_single(self, query_idx: int) -> np.ndarray:
        refs = self.selector.select(np.asarray([query_idx], dtype=np.int64))[0]
        q = self.dataset.get_x(np.asarray([query_idx], dtype=np.int64))
        r = self.dataset.get_x(refs)[None, ...]
        m = build_resonance_tensor(
            batch_samples=q,
            ref_samples=r,
            fs=self.fs,
            nperseg=self.coherence_nperseg,
        )
        return m[0].astype(np.float32)

    def prepare_cache(self, query_indices: np.ndarray, show_progress: bool = True) -> None:
        """Precompute resonance maps for fixed query indices."""
        if self.mode != "precompute":
            return
        query_indices = np.asarray(query_indices, dtype=np.int64)
        iterator = tqdm(query_indices, desc="Precomputing resonance", leave=False) if show_progress else query_indices
        for idx in iterator:
            q = int(idx)
            if q not in self.cache:
                self.cache[q] = self._compute_single(q)

    def get_batch(self, batch_indices: np.ndarray, batch_x: torch.Tensor) -> torch.Tensor:
        """Return resonance tensor [B, K, C, C, 2] as torch float."""
        batch_indices = np.asarray(batch_indices, dtype=np.int64)

        if self.mode == "precompute":
            out = []
            for idx in batch_indices:
                i = int(idx)
                if i not in self.cache:
                    self.cache[i] = self._compute_single(i)
                out.append(self.cache[i])
            arr = np.stack(out, axis=0)
            return torch.from_numpy(arr).float()

        refs = self.selector.select(batch_indices)
        b, k = refs.shape
        ref_samples = self.dataset.get_x(refs.reshape(-1)).reshape(b, k, *self.dataset.x.shape[1:])
        arr = build_resonance_tensor(
            batch_samples=batch_x.detach().cpu().numpy(),
            ref_samples=ref_samples,
            fs=self.fs,
            nperseg=self.coherence_nperseg,
        )
        return torch.from_numpy(arr).float()


def _make_loader(
    dataset: BaseEEGDataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    subset = Subset(dataset, np.asarray(indices, dtype=np.int64).tolist())
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def _build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    opt_name = cfg.get("optimizer", "adam").lower()
    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    if opt_name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    epochs: int,
):
    sch_name = cfg.get("scheduler", "none").lower()
    if sch_name == "none":
        return None
    if sch_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if sch_name == "step":
        return StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 20)),
            gamma=float(cfg.get("gamma", 0.5)),
        )
    if sch_name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg.get("gamma", 0.5)),
            patience=int(cfg.get("plateau_patience", 5)),
        )
    raise ValueError(f"Unsupported scheduler: {sch_name}")


class GRNTrainer:
    """Trainer with early stopping, AMP, checkpointing, and metrics export."""

    def __init__(
        self,
        model: GRNModel,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler,
        lambda_proto: float,
        use_proto_regularizer: bool,
        entropy_weight: float,
        mixed_precision: bool,
        grad_clip_norm: float,
        logger,
        output_dir: Path,
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lambda_proto = float(lambda_proto)
        self.use_proto_regularizer = bool(use_proto_regularizer)
        self.grad_clip_norm = float(grad_clip_norm)
        self.logger = logger
        self.output_dir = output_dir

        self.cls_loss = ClassificationLoss()
        self.proto_loss = PrototypeRegularizationLoss(entropy_weight=entropy_weight)

        self.amp_enabled = mixed_precision and device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.amp_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def _run_epoch(
        self,
        loader: DataLoader,
        resonance_provider: ResonanceProvider | None,
        training: bool,
        desc: str,
    ) -> Dict[str, Any]:
        self.model.train(training)

        total_loss = 0.0
        total_cls = 0.0
        total_proto = 0.0
        n_samples = 0
        y_true: list[int] = []
        y_pred: list[int] = []

        iterator = tqdm(loader, desc=desc, leave=False)
        for batch in iterator:
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            idx = batch["index"].cpu().numpy()

            m_res = None
            if self.model.use_resonance:
                if resonance_provider is None:
                    raise ValueError("Resonance is enabled but provider is None")
                m_res = resonance_provider.get_batch(batch_indices=idx, batch_x=x).to(self.device)

            with torch.set_grad_enabled(training):
                with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    out = self.model(x, resonance_tensor=m_res)
                    logits = out["logits"]
                    cls_loss = self.cls_loss(logits, y)
                    proto_reg = torch.tensor(0.0, device=self.device)
                    if (
                        self.use_proto_regularizer
                        and self.model.use_prototypes
                        and out["R"] is not None
                        and out["proto_attn"] is not None
                        and self.lambda_proto > 0
                    ):
                        proto_reg = self.proto_loss(out["F"], out["R"], out["proto_attn"])
                    loss = cls_loss + self.lambda_proto * proto_reg

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            bs = y.shape[0]
            total_loss += float(loss.item()) * bs
            total_cls += float(cls_loss.item()) * bs
            total_proto += float(proto_reg.item()) * bs
            n_samples += bs

            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

        if n_samples == 0:
            return {
                "loss": 0.0,
                "cls_loss": 0.0,
                "proto_loss": 0.0,
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "per_class_accuracy": [],
                "confusion_matrix": [],
                "y_true": [],
                "y_pred": [],
            }

        metrics = classification_metrics(y_true, y_pred)
        metrics.update(
            {
                "loss": total_loss / n_samples,
                "cls_loss": total_cls / n_samples,
                "proto_loss": total_proto / n_samples,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_provider: ResonanceProvider | None,
        eval_provider: ResonanceProvider | None,
        epochs: int,
        patience: int,
        ckpt_dir: Path,
        resume_path: str | None = None,
    ) -> Dict[str, Any]:
        ckpt_dir = ensure_dir(ckpt_dir)
        best_ckpt = ckpt_dir / "best.pt"
        last_ckpt = ckpt_dir / "last.pt"

        start_epoch = 0
        best_val_acc = -1.0
        best_epoch = -1
        best_val_metrics: Dict[str, Any] = {}
        history: list[Dict[str, Any]] = []
        stale_epochs = 0

        if resume_path:
            state = load_checkpoint(resume_path, map_location=self.device.type)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            if self.scheduler is not None and state.get("scheduler") is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            if state.get("scaler") is not None:
                self.scaler.load_state_dict(state["scaler"])
            history = state.get("history", [])
            start_epoch = int(state.get("epoch", -1)) + 1
            best_val_acc = float(state.get("best_val_acc", best_val_acc))
            best_epoch = int(state.get("best_epoch", best_epoch))
            best_val_metrics = state.get("best_val_metrics", {})
            stale_epochs = int(state.get("stale_epochs", 0))
            self.logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

        for epoch in range(start_epoch, epochs):
            train_stats = self._run_epoch(
                loader=train_loader,
                resonance_provider=train_provider,
                training=True,
                desc=f"Train {epoch + 1}/{epochs}",
            )
            val_stats = self._run_epoch(
                loader=val_loader,
                resonance_provider=eval_provider,
                training=False,
                desc=f"Val {epoch + 1}/{epochs}",
            )

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_stats["loss"])
                else:
                    self.scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["accuracy"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["accuracy"],
                "train_macro_f1": train_stats["macro_f1"],
                "val_macro_f1": val_stats["macro_f1"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            history.append(row)

            state = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler": self.scaler.state_dict() if self.amp_enabled else None,
                "history": history,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "best_val_metrics": best_val_metrics,
                "stale_epochs": stale_epochs,
            }
            save_checkpoint(state, last_ckpt)

            if val_stats["accuracy"] > best_val_acc:
                best_val_acc = float(val_stats["accuracy"])
                best_epoch = epoch
                best_val_metrics = val_stats
                stale_epochs = 0
                state.update(
                    {
                        "best_val_acc": best_val_acc,
                        "best_epoch": best_epoch,
                        "best_val_metrics": best_val_metrics,
                    }
                )
                save_checkpoint(state, best_ckpt)
            else:
                stale_epochs += 1

            self.logger.info(
                "epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f best_val=%.4f",
                epoch,
                train_stats["loss"],
                train_stats["accuracy"],
                val_stats["loss"],
                val_stats["accuracy"],
                best_val_acc,
            )

            if stale_epochs >= patience:
                self.logger.info("Early stopping at epoch %d", epoch)
                break

        # ensure best checkpoint exists even if validation never improved
        if not best_ckpt.exists() and last_ckpt.exists():
            save_checkpoint(load_checkpoint(last_ckpt, map_location=self.device.type), best_ckpt)

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_val_metrics": best_val_metrics,
            "best_checkpoint": str(best_ckpt),
            "last_checkpoint": str(last_ckpt),
        }

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        resonance_provider: ResonanceProvider | None,
        desc: str = "Eval",
    ) -> Dict[str, Any]:
        return self._run_epoch(
            loader=loader,
            resonance_provider=resonance_provider,
            training=False,
            desc=desc,
        )


def _create_model(config: Dict[str, Any], num_classes: int) -> GRNModel:
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
        use_resonance=bool(mcfg.get("use_resonance", True)),
    )


def _build_reference_provider(
    dataset: BaseEEGDataset,
    train_idx: np.ndarray,
    query_indices: np.ndarray,
    config: Dict[str, Any],
    forbidden_subjects: set[int],
    show_progress: bool,
) -> ResonanceProvider:
    rcfg = config["resonance"]
    strategy = str(rcfg.get("selector_strategy", "random"))
    k_refs = int(rcfg.get("k_refs", 3))
    seed = int(config["experiment"].get("seed", 42))
    mode = str(rcfg.get("mode", "precompute"))
    fs = float(rcfg.get("fs", config["dataset"].get("sampling_rate", 200)))
    coherence_nperseg = int(rcfg.get("coherence_nperseg", 128))

    features = dataset.flat_features() if strategy == "nearest" else None
    selector = ReferenceSelector(
        allowed_indices=np.asarray(train_idx, dtype=np.int64),
        labels=dataset.y,
        subject_ids=dataset.subject_id,
        config=ReferenceSelectionConfig(strategy=strategy, k_refs=k_refs, seed=seed),
        features=features,
        forbidden_subjects=forbidden_subjects,
    )
    provider = ResonanceProvider(
        dataset=dataset,
        selector=selector,
        mode=mode,
        fs=fs,
        coherence_nperseg=coherence_nperseg,
    )
    if mode.lower() == "precompute":
        provider.prepare_cache(np.asarray(query_indices, dtype=np.int64), show_progress=show_progress)
    return provider


def run_fold(
    config: Dict[str, Any],
    dataset: BaseEEGDataset,
    split: FoldSplit,
    run_dir: str | Path,
    device: torch.device,
    resume_path: str | None = None,
) -> FoldResult:
    """Run one fold end-to-end: train, validate, test, and export artifacts."""
    run_dir = Path(run_dir)
    fold_dir = ensure_dir(run_dir / split.fold_name)
    ckpt_dir = ensure_dir(fold_dir / "checkpoints")
    log_dir = ensure_dir(fold_dir / "logs")
    result_dir = ensure_dir(fold_dir / "results")

    logger = create_logger(f"grn.{split.fold_name}", log_dir / "train.log")

    batch_size = int(config["training"].get("batch_size", 64))
    num_workers = int(config["dataset"].get("num_workers", 0))
    seed = int(config["experiment"].get("seed", 42))

    val_idx = split.val_idx
    if len(val_idx) == 0:
        logger.info("Validation split is empty. Reusing test split for validation metrics.")
        val_idx = split.test_idx

    train_loader = _make_loader(
        dataset, split.train_idx, batch_size=batch_size, shuffle=True, num_workers=num_workers, seed=seed
    )
    val_loader = _make_loader(
        dataset, val_idx, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed
    )
    test_loader = _make_loader(
        dataset, split.test_idx, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed
    )

    model = _create_model(config, num_classes=dataset.num_classes).to(device)
    optimizer = _build_optimizer(model, config["training"])
    scheduler = _build_scheduler(optimizer, config["training"], epochs=int(config["training"].get("epochs", 80)))

    use_resonance = bool(config["model"].get("use_resonance", True))
    train_provider = None
    eval_provider = None
    if use_resonance:
        protocol = str(config["dataset"].get("protocol", "sd")).lower()
        forbidden_subjects = (
            set(dataset.get_subject_ids(split.test_idx).tolist()) if protocol == "loso" else set()
        )
        train_provider = _build_reference_provider(
            dataset=dataset,
            train_idx=split.train_idx,
            query_indices=split.train_idx,
            config=config,
            forbidden_subjects=forbidden_subjects,
            show_progress=False,
        )
        eval_queries = np.unique(np.concatenate([val_idx, split.test_idx], axis=0))
        eval_provider = _build_reference_provider(
            dataset=dataset,
            train_idx=split.train_idx,
            query_indices=eval_queries,
            config=config,
            forbidden_subjects=forbidden_subjects,
            show_progress=False,
        )

    trainer = GRNTrainer(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        lambda_proto=float(config["loss"].get("lambda_proto", 0.1)),
        use_proto_regularizer=bool(config["model"].get("use_prototypes", True)),
        entropy_weight=float(config["loss"].get("prototype_entropy_weight", 0.01)),
        mixed_precision=bool(config["training"].get("mixed_precision", False)),
        grad_clip_norm=float(config["training"].get("grad_clip_norm", 1.0)),
        logger=logger,
        output_dir=fold_dir,
    )

    fit_out = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        train_provider=train_provider,
        eval_provider=eval_provider,
        epochs=int(config["training"].get("epochs", 80)),
        patience=int(config["training"].get("early_stopping_patience", 10)),
        ckpt_dir=ckpt_dir,
        resume_path=resume_path,
    )

    # Load best checkpoint for test evaluation.
    best_state = load_checkpoint(fit_out["best_checkpoint"], map_location=device.type)
    model.load_state_dict(best_state["model"])

    val_metrics = trainer.evaluate(val_loader, eval_provider, desc="Val(best)")
    test_metrics = trainer.evaluate(test_loader, eval_provider, desc="Test")

    history_file = result_dir / "history.csv"
    with history_file.open("w", newline="", encoding="utf-8") as f:
        if fit_out["history"]:
            writer = csv.DictWriter(f, fieldnames=list(fit_out["history"][0].keys()))
            writer.writeheader()
            writer.writerows(fit_out["history"])

    pred_file = result_dir / "test_predictions.npz"
    np.savez_compressed(
        pred_file,
        y_true=np.asarray(test_metrics["y_true"], dtype=np.int64),
        y_pred=np.asarray(test_metrics["y_pred"], dtype=np.int64),
    )

    # Remove raw predictions from metric json, keep them in npz.
    val_metrics_clean = dict(val_metrics)
    test_metrics_clean = dict(test_metrics)
    val_metrics_clean.pop("y_true", None)
    val_metrics_clean.pop("y_pred", None)
    test_metrics_clean.pop("y_true", None)
    test_metrics_clean.pop("y_pred", None)

    save_json(val_metrics_clean, result_dir / "val_metrics.json")
    save_json(test_metrics_clean, result_dir / "test_metrics.json")

    return FoldResult(
        fold_name=split.fold_name,
        best_epoch=int(fit_out["best_epoch"]),
        best_val_accuracy=float(fit_out["best_val_acc"]),
        val_metrics=val_metrics_clean,
        test_metrics=test_metrics_clean,
        best_checkpoint=str(fit_out["best_checkpoint"]),
        last_checkpoint=str(fit_out["last_checkpoint"]),
        history_file=str(history_file),
        predictions_file=str(pred_file),
    )
