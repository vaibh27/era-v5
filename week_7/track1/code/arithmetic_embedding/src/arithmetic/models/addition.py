from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from arithmetic.models.carry_cell import CarryCell
from arithmetic.representations.limb_embedding import LimbEmbedding


@dataclass
class AdditionOutput:
    result_logits: torch.Tensor  # [batch, steps, base]
    carry_logits: torch.Tensor  # [batch, steps, 2]
    result_latents: torch.Tensor  # [batch, steps, dim]


class RecurrentLatentAdder(nn.Module):
    """Variable-length addition by repeatedly applying one local carry cell."""

    def __init__(self, base: int = 1000, limb_dim: int = 32, hidden_dim: int = 128) -> None:
        super().__init__()
        self.base = base
        self.embedding = LimbEmbedding(base, limb_dim)
        self.cell = CarryCell(limb_dim, base, hidden_dim)

    def forward(
        self,
        a_limbs: torch.Tensor,
        b_limbs: torch.Tensor,
        teacher_carries: torch.Tensor | None = None,
    ) -> AdditionOutput:
        if a_limbs.shape != b_limbs.shape or a_limbs.ndim != 2:
            raise ValueError("a_limbs and b_limbs must both have shape [batch, steps]")
        batch, steps = a_limbs.shape
        if teacher_carries is not None and teacher_carries.shape != (batch, steps):
            raise ValueError("teacher_carries must have shape [batch, steps]")
        a_latents = self.embedding(a_limbs)
        b_latents = self.embedding(b_limbs)
        carry = torch.zeros(batch, dtype=torch.long, device=a_limbs.device)
        result_logits, carry_logits, result_latents = [], [], []
        for position in range(steps):
            output = self.cell(a_latents[:, position], b_latents[:, position], carry)
            result_logits.append(output.result_logits)
            carry_logits.append(output.carry_logits)
            result_latents.append(output.result_latent)
            # Autoregressive carry makes inference algorithmic; training can still
            # supervise every transition directly through carry_logits.
            carry = teacher_carries[:, position] if teacher_carries is not None else output.carry_logits.argmax(dim=-1)
        return AdditionOutput(
            result_logits=torch.stack(result_logits, dim=1),
            carry_logits=torch.stack(carry_logits, dim=1),
            result_latents=torch.stack(result_latents, dim=1),
        )
