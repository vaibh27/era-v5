"""Schoolbook variable-length multiplication with a learned shared product cell."""

from __future__ import annotations

import torch
from torch import nn

from arithmetic.models.addition import RecurrentLatentAdder
from arithmetic.models.product_cell import ProductCell
from arithmetic.representations.limb_embedding import LimbEmbedding


class SchoolbookLatentMultiplier(nn.Module):
    """Predict local limb products, accumulate shifted partial products, normalize carries."""

    def __init__(self, base: int = 10, limb_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.base = base
        self.embedding = LimbEmbedding(base, limb_dim)
        self.product_cell = ProductCell(limb_dim, base, hidden_dim)

    def local_products(self, a: torch.Tensor, b: torch.Tensor):
        """Return low/high logits for every pair of limb positions."""
        ea, eb = self.embedding(a), self.embedding(b)
        batch, na, _ = ea.shape; nb = eb.shape[1]
        left = ea[:, :, None, :].expand(batch, na, nb, -1)
        right = eb[:, None, :, :].expand(batch, na, nb, -1)
        return self.product_cell(left, right)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return normalized predicted result limbs. Inputs are padded little-endian limbs."""
        products = self.local_products(a, b)
        low, high = products.low_logits.argmax(-1), products.high_logits.argmax(-1)
        batch, na, nb = low.shape
        totals = torch.zeros(batch, na + nb + 1, dtype=torch.long, device=a.device)
        for i in range(na):
            for j in range(nb):
                totals[:, i + j] += low[:, i, j]
                totals[:, i + j + 1] += high[:, i, j]
        result = torch.zeros_like(totals)
        carry = torch.zeros(batch, dtype=torch.long, device=a.device)
        for position in range(totals.shape[1]):
            value = totals[:, position] + carry
            result[:, position] = value % self.base
            carry = value // self.base
        return result


class CompositionalLatentMultiplier(nn.Module):
    """Schoolbook multiplication whose partial sums use a learned carry adder.

    This version deliberately avoids ``% base`` / ``// base`` normalization in
    the multiplication forward pass. The only algorithmic control flow is
    schoolbook shifting; all local product and carry transitions are predicted
    by shared learned latent cells.
    """

    def __init__(self, product_model: SchoolbookLatentMultiplier, adder: RecurrentLatentAdder) -> None:
        super().__init__()
        if product_model.base != adder.base:
            raise ValueError("product model and adder must use the same base")
        self.base = product_model.base
        self.product_model = product_model
        self.adder = adder

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        products = self.product_model.local_products(a, b)
        low = products.low_logits.argmax(-1)
        high = products.high_logits.argmax(-1)
        batch, na, nb = low.shape
        steps = na + nb + 1
        accumulated = torch.zeros(batch, steps, dtype=torch.long, device=a.device)
        for i in range(na):
            for j in range(nb):
                partial = torch.zeros_like(accumulated)
                partial[:, i + j] = low[:, i, j]
                partial[:, i + j + 1] = high[:, i, j]
                accumulated = self.adder(accumulated, partial).result_logits.argmax(dim=-1)
        return accumulated
