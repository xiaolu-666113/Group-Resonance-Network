"""Prototype regularization loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeRegularizationLoss(nn.Module):
    """Encourage stable alignment between F and attended prototype embedding R."""

    def __init__(self, entropy_weight: float = 0.01, eps: float = 1e-8) -> None:
        super().__init__()
        self.entropy_weight = entropy_weight
        self.eps = eps

    def forward(
        self,
        f: torch.Tensor,
        r: torch.Tensor,
        attn: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prototype regularization term.

        - alignment: 1 - cosine(F, R)
        - entropy: lower entropy encourages sharper stable assignments
        """
        align = 1.0 - F.cosine_similarity(f, r, dim=-1)
        entropy = -torch.sum(attn * torch.log(attn + self.eps), dim=-1)
        if attn.shape[-1] > 1:
            entropy = entropy / torch.log(torch.tensor(float(attn.shape[-1]), device=attn.device))
        return align.mean() + self.entropy_weight * entropy.mean()
