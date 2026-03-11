"""Classification loss wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    """Cross entropy classification loss."""

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)
