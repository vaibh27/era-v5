"""Shared local limb-product predictor for schoolbook multiplication."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ProductCellOutput:
    low_logits: torch.Tensor
    high_logits: torch.Tensor


class ProductCell(nn.Module):
    """Map two learned limb latents to low and high limbs of their product."""

    def __init__(self, limb_dim: int, base: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2 * limb_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.low = nn.Linear(hidden_dim, base)
        self.high = nn.Linear(hidden_dim, base)

    def forward(self, a_latent: torch.Tensor, b_latent: torch.Tensor) -> ProductCellOutput:
        h = self.body(torch.cat((a_latent, b_latent), dim=-1))
        return ProductCellOutput(self.low(h), self.high(h))
