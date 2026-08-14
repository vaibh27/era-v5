from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from arithmetic.models.borrow_cell import BorrowCell
from arithmetic.representations.limb_embedding import LimbEmbedding


@dataclass
class SubtractionOutput:
    result_logits: torch.Tensor
    borrow_logits: torch.Tensor
    result_latents: torch.Tensor


class RecurrentLatentSubtractor(nn.Module):
    """Unsigned a-b for a>=b via a shared local borrow transition."""

    def __init__(self, base: int = 10, limb_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.base = base
        self.embedding = LimbEmbedding(base, limb_dim)
        self.cell = BorrowCell(limb_dim, base, hidden_dim)

    def forward(self, a_limbs: torch.Tensor, b_limbs: torch.Tensor, teacher_borrows: torch.Tensor | None = None) -> SubtractionOutput:
        if a_limbs.shape != b_limbs.shape or a_limbs.ndim != 2:
            raise ValueError("a_limbs and b_limbs must both have shape [batch, steps]")
        batch, steps = a_limbs.shape
        if teacher_borrows is not None and teacher_borrows.shape != (batch, steps):
            raise ValueError("teacher_borrows must have shape [batch, steps]")
        a_latents, b_latents = self.embedding(a_limbs), self.embedding(b_limbs)
        borrow = torch.zeros(batch, dtype=torch.long, device=a_limbs.device)
        result_logits, borrow_logits, result_latents = [], [], []
        for position in range(steps):
            output = self.cell(a_latents[:, position], b_latents[:, position], borrow)
            result_logits.append(output.result_logits)
            borrow_logits.append(output.carry_logits)
            result_latents.append(output.result_latent)
            borrow = teacher_borrows[:, position] if teacher_borrows is not None else output.carry_logits.argmax(dim=-1)
        return SubtractionOutput(torch.stack(result_logits, 1), torch.stack(borrow_logits, 1), torch.stack(result_latents, 1))
