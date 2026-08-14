from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CarryCellOutput:
    result_latent: torch.Tensor
    result_logits: torch.Tensor
    carry_logits: torch.Tensor


class CarryCell(nn.Module):
    """One shared local transition for addition at every limb position.

    Inputs are two limb latents and a categorical carry-in.  It predicts the
    result limb and carry-out; no parameter is position-specific.
    """

    def __init__(self, limb_dim: int, base: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.base = base
        self.carry_embedding = nn.Embedding(2, limb_dim)
        self.body = nn.Sequential(
            nn.Linear(3 * limb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.result_latent = nn.Linear(hidden_dim, limb_dim)
        self.result_classifier = nn.Linear(hidden_dim, base)
        self.carry_classifier = nn.Linear(hidden_dim, 2)

    def forward(self, a_latent: torch.Tensor, b_latent: torch.Tensor, carry_in: torch.Tensor) -> CarryCellOutput:
        carry_latent = self.carry_embedding(carry_in)
        hidden = self.body(torch.cat((a_latent, b_latent, carry_latent), dim=-1))
        return CarryCellOutput(
            result_latent=self.result_latent(hidden),
            result_logits=self.result_classifier(hidden),
            carry_logits=self.carry_classifier(hidden),
        )
