"""Individual EEG encoder modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class IndividualEncoder(nn.Module):
    """Lightweight CNN + Transformer encoder.

    Input shape:
    - x: [B, T, C, Bf]

    Output shape:
    - F: [B, d]
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        conv_hidden_dim: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.temporal_conv = nn.Sequential(
            nn.LazyConv1d(conv_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_hidden_dim),
            nn.GELU(),
            nn.Conv1d(conv_hidden_dim, conv_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=conv_hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.out_norm = nn.LayerNorm(conv_hidden_dim)
        self.out_proj = nn.Linear(conv_hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode EEG sample into compact individual representation F."""
        assert x.ndim == 4, f"Expected x [B,T,C,Bf], got {x.shape}"
        b, t, c, bf = x.shape
        x = x.reshape(b, t, c * bf).transpose(1, 2)  # [B, C*Bf, T]
        x = self.temporal_conv(x)  # [B, H, T]
        x = x.transpose(1, 2)  # [B, T, H]
        x = self.transformer(x)
        x = self.out_norm(x)
        x = x.mean(dim=1)
        return self.out_proj(x)
