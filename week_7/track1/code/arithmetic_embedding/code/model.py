from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class FactorizedBilinear(nn.Module):
    """Low-rank bilinear map T(u,v) = P((Au) * (Bv))."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.left = nn.Linear(dim, rank, bias=False)
        self.right = nn.Linear(dim, rank, bias=False)
        self.out = nn.Linear(rank, dim, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.out(self.left(x) * self.right(y))


class AdditiveOperator(nn.Module):
    """Learned commutative operator. Symmetrized by construction."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = torch.cat([x, y], dim=-1)
        yx = torch.cat([y, x], dim=-1)
        return 0.5 * (self.net(xy) + self.net(yx))


@dataclass
class ModelOutput:
    add: torch.Tensor
    mul: torch.Tensor


class LearnedArithmeticEmbedding(nn.Module):
    """Minimal learnable arithmetic embedding experiment.

    Each integer in Z_M has a trainable vector. Two operators are trained
    to map operand embeddings to the embedding of the algebraic result.
    """

    def __init__(
        self,
        modulus: int,
        dim: int = 16,
        add_hidden: int = 64,
        bilinear_rank: int = 16,
    ) -> None:
        super().__init__()
        self.modulus = modulus
        self.dim = dim
        self.embedding = nn.Embedding(modulus, dim)
        self.add_op = AdditiveOperator(dim, add_hidden)
        self.mul_op = FactorizedBilinear(dim, bilinear_rank)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.5)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> ModelOutput:
        ex = self.embedding(x)
        ey = self.embedding(y)
        return ModelOutput(
            add=self.add_op(ex, ey),
            mul=self.mul_op(ex, ey),
        )

    def normalized_embeddings(self) -> torch.Tensor:
        return F.normalize(self.embedding.weight, dim=-1)


def contrastive_target_loss(
    predicted: torch.Tensor,
    embedding_matrix: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Classify predicted vectors against all learned number embeddings."""
    pred = F.normalize(predicted, dim=-1)
    emb = F.normalize(embedding_matrix, dim=-1)
    logits = pred @ emb.T / temperature
    return F.cross_entropy(logits, targets)


def separation_loss(embedding_matrix: torch.Tensor, margin: float = 0.10) -> torch.Tensor:
    """Penalize near-collisions among learned embeddings."""
    e = F.normalize(embedding_matrix, dim=-1)
    sim = e @ e.T
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    positive = F.relu(sim[mask] - (1.0 - margin))
    return positive.square().mean()
