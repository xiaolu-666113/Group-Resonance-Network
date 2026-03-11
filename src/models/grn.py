"""Group Resonance Network model."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .encoders import IndividualEncoder
from .fusion import ResonanceAwareFusion
from .prototypes import PrototypeBank
from .resonance import ResonanceEncoder


class GRNModel(nn.Module):
    """Full GRN architecture for EEG emotion recognition."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        encoder_hidden_dim: int = 256,
        encoder_transformer_layers: int = 2,
        encoder_transformer_heads: int = 4,
        dropout: float = 0.2,
        num_prototypes: int = 8,
        prototype_temperature: float = 0.07,
        prototype_similarity: str = "cosine",
        resonance_channels: int = 32,
        fusion_hidden_dim: int = 512,
        use_prototypes: bool = True,
        use_resonance: bool = True,
    ) -> None:
        super().__init__()
        self.use_prototypes = use_prototypes
        self.use_resonance = use_resonance

        self.encoder = IndividualEncoder(
            embedding_dim=embedding_dim,
            conv_hidden_dim=encoder_hidden_dim,
            transformer_layers=encoder_transformer_layers,
            transformer_heads=encoder_transformer_heads,
            dropout=dropout,
        )

        self.prototype_bank = (
            PrototypeBank(
                num_prototypes=num_prototypes,
                embedding_dim=embedding_dim,
                temperature=prototype_temperature,
                similarity=prototype_similarity,
            )
            if use_prototypes
            else None
        )

        self.resonance_encoder = (
            ResonanceEncoder(embedding_dim=embedding_dim, channels=resonance_channels, dropout=dropout)
            if use_resonance
            else None
        )

        self.fusion = ResonanceAwareFusion(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            use_prototypes=use_prototypes,
            use_resonance=use_resonance,
        )

    def forward(self, x: torch.Tensor, resonance_tensor: torch.Tensor | None = None) -> Dict[str, Any]:
        """Forward pass.

        Args:
            x: [B, T, C, Bf]
            resonance_tensor: [B, K, C, C, 2] when resonance is enabled
        """
        f = self.encoder(x)  # [B, d]

        r = None
        attn = None
        proto_sim = None
        if self.use_prototypes:
            assert self.prototype_bank is not None
            r, attn, proto_sim = self.prototype_bank(f)

        g = None
        if self.use_resonance:
            if resonance_tensor is None:
                raise ValueError("resonance_tensor is required when use_resonance=True")
            assert self.resonance_encoder is not None
            g = self.resonance_encoder(resonance_tensor)

        h, logits = self.fusion(f, r, g)

        return {
            "logits": logits,
            "F": f,
            "R": r,
            "G": g,
            "H": h,
            "proto_attn": attn,
            "proto_sim": proto_sim,
        }
