from __future__ import annotations

import torch
from torch import nn


class LimbEmbedding(nn.Module):
    """A shared learned latent vector for each digit/limb value in one base."""

    def __init__(self, base: int, dim: int) -> None:
        super().__init__()
        self.base = base
        self.dim = dim
        self.table = nn.Embedding(base, dim)
        nn.init.normal_(self.table.weight, std=0.1)

    def forward(self, limbs: torch.Tensor) -> torch.Tensor:
        return self.table(limbs)
