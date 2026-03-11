"""Learnable group prototype module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    """Learnable group prototypes p1...pM with attention-based aggregation."""

    def __init__(
        self,
        num_prototypes: int,
        embedding_dim: int,
        temperature: float = 0.07,
        similarity: str = "cosine",
    ) -> None:
        super().__init__()
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be > 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.similarity = similarity.lower()

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim) * 0.02)

    def _similarity(self, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if self.similarity == "cosine":
            f_norm = F.normalize(f, p=2, dim=-1)
            p_norm = F.normalize(p, p=2, dim=-1)
            return f_norm @ p_norm.transpose(0, 1)
        raise ValueError(f"Unsupported similarity: {self.similarity}")

    def forward(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prototype-induced embedding R.

        Args:
            f: [B, d]

        Returns:
            r: [B, d]
            attn: [B, M]
            sim: [B, M]
        """
        assert f.ndim == 2, f"Expected f [B,d], got {f.shape}"
        sim = self._similarity(f, self.prototypes) / self.temperature
        attn = torch.softmax(sim, dim=-1)
        r = attn @ self.prototypes
        return r, attn, sim
