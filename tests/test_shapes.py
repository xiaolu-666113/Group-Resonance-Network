"""Shape tests for GRN model."""

from __future__ import annotations

import torch

from src.models.grn import GRNModel


def test_grn_full_shapes() -> None:
    model = GRNModel(num_classes=3, use_prototypes=True, use_resonance=True)
    x = torch.randn(2, 12, 32, 5)
    m_res = torch.randn(2, 3, 32, 32, 2)
    out = model(x, resonance_tensor=m_res)

    assert out["logits"].shape == (2, 3)
    assert out["F"].shape == (2, 256)
    assert out["R"].shape == (2, 256)
    assert out["G"].shape == (2, 256)
    assert out["proto_attn"].shape[0] == 2


def test_grn_individual_only_shapes() -> None:
    model = GRNModel(num_classes=2, use_prototypes=False, use_resonance=False)
    x = torch.randn(4, 10, 16, 5)
    out = model(x)

    assert out["logits"].shape == (4, 2)
    assert out["R"] is None
    assert out["G"] is None
