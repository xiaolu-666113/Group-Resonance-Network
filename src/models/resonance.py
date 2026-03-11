"""Resonance tensor encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResonanceEncoder(nn.Module):
    """Encode multi-subject resonance tensor M_res into embedding G.

    Input:
    - M_res: [B, K, C, C, 2]

    Output:
    - G: [B, d]
    """

    def __init__(self, embedding_dim: int = 256, channels: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(2, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(channels * 2),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.out = nn.Linear(channels * 2, embedding_dim)

    def forward(self, m_res: torch.Tensor) -> torch.Tensor:
        assert m_res.ndim == 5, f"Expected M_res [B,K,C,C,2], got {m_res.shape}"
        x = m_res.permute(0, 4, 1, 2, 3).contiguous()  # [B,2,K,C,C]
        x = self.net(x).flatten(1)
        return self.out(x)
