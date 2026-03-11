"""Resonance-aware feature fusion."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResonanceAwareFusion(nn.Module):
    """Fuse F, R, G and interaction terms, then classify.

    Full fusion concat:
    [F, R, G, D_R, D_G, C_R, C_G]
    where D_* = F - *, C_* = F * *
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        use_prototypes: bool = True,
        use_resonance: bool = True,
    ) -> None:
        super().__init__()
        self.use_prototypes = use_prototypes
        self.use_resonance = use_resonance

        if use_prototypes and use_resonance:
            fusion_in = embedding_dim * 7
        elif use_prototypes or use_resonance:
            fusion_in = embedding_dim * 4
        else:
            fusion_in = embedding_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        f: torch.Tensor,
        r: torch.Tensor | None,
        g: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert f.ndim == 2, f"Expected F [B,d], got {f.shape}"

        if self.use_prototypes and self.use_resonance:
            assert r is not None and g is not None
            d_r = f - r
            d_g = f - g
            c_r = f * r
            c_g = f * g
            z = torch.cat([f, r, g, d_r, d_g, c_r, c_g], dim=-1)
        elif self.use_prototypes:
            assert r is not None
            d_r = f - r
            c_r = f * r
            z = torch.cat([f, r, d_r, c_r], dim=-1)
        elif self.use_resonance:
            assert g is not None
            d_g = f - g
            c_g = f * g
            z = torch.cat([f, g, d_g, c_g], dim=-1)
        else:
            z = f

        h = self.fusion_mlp(z)
        logits = self.classifier(h)
        return h, logits
