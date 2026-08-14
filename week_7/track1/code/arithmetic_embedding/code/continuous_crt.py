"""Exact continuous-vector realization of the CRT arithmetic subspace.

Each residue r in Z_m is represented by its real one-hot vector e_r.  A fixed
bilinear tensor maps e_a, e_b to e_(a+b mod m) or e_(a*b mod m).  Concatenating
these channels gives a continuous vector compatible with a neural embedding,
while retaining exact finite-ring arithmetic on valid encoded states.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from crt import CRTSystem


class ContinuousCRTEmbedding(nn.Module):
    """Fixed real-vector embedding and exact bilinear operators for Z_M."""

    def __init__(self, moduli: Sequence[int]) -> None:
        super().__init__()
        self.system = CRTSystem(tuple(moduli))
        self.moduli = self.system.moduli
        self.dim = sum(self.moduli)
        self._offsets = tuple(range(0, self.dim))

        for index, modulus in enumerate(self.moduli):
            add_table = self._operation_table(modulus, multiply=False)
            mul_table = self._operation_table(modulus, multiply=True)
            self.register_buffer(f"add_table_{index}", add_table)
            self.register_buffer(f"mul_table_{index}", mul_table)

    @staticmethod
    def _operation_table(modulus: int, multiply: bool) -> torch.Tensor:
        """Return T[k, i, j] where T(e_i, e_j) = e_k."""
        indices = torch.arange(modulus)
        result = (indices[:, None] * indices[None, :] if multiply else indices[:, None] + indices[None, :]) % modulus
        table = torch.zeros((modulus, modulus, modulus), dtype=torch.float32)
        table[result, indices[:, None], indices[None, :]] = 1.0
        return table

    def encode(self, x: torch.Tensor | int | Sequence[int]) -> torch.Tensor:
        """Encode integers as concatenated one-hot residue channels."""
        values = torch.as_tensor(x, device=self.add_table_0.device, dtype=torch.long)
        blocks = [F.one_hot(torch.remainder(values, modulus), modulus).float() for modulus in self.moduli]
        return torch.cat(blocks, dim=-1)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """Decode valid (or nearest one-hot) arithmetic vectors to Z_M values."""
        if embedding.shape[-1] != self.dim:
            raise ValueError(f"Expected final dimension {self.dim}, got {embedding.shape[-1]}.")
        residues = []
        start = 0
        for modulus in self.moduli:
            residues.append(embedding[..., start : start + modulus].argmax(dim=-1))
            start += modulus

        value = torch.zeros_like(residues[0])
        M = self.system.capacity
        for residue, modulus in zip(residues, self.moduli):
            Mi = M // modulus
            value = value + residue * Mi * pow(Mi, -1, modulus)
        return torch.remainder(value, M)

    def _apply(self, x: torch.Tensor, y: torch.Tensor, operation: str) -> torch.Tensor:
        if x.shape != y.shape or x.shape[-1] != self.dim:
            raise ValueError(f"Operands must have the same shape ending in {self.dim}.")
        blocks = []
        start = 0
        for index, modulus in enumerate(self.moduli):
            table = getattr(self, f"{operation}_table_{index}")
            xb = x[..., start : start + modulus]
            yb = y[..., start : start + modulus]
            blocks.append(torch.einsum("...i,...j,kij->...k", xb, yb, table))
            start += modulus
        return torch.cat(blocks, dim=-1)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._apply(x, y, "add")

    def mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._apply(x, y, "mul")


class HybridArithmeticEmbedding(nn.Module):
    """Concatenate an exact arithmetic subspace with an independent semantic one.

    Arithmetic operators intentionally act only on ``math``.  A language model
    can train/use ``semantic`` for lexical information without changing the
    arithmetic guarantee supplied by the fixed CRT channels.
    """

    def __init__(self, moduli: Sequence[int], semantic_vocab_size: int, semantic_dim: int) -> None:
        super().__init__()
        self.math = ContinuousCRTEmbedding(moduli)
        self.semantic = nn.Embedding(semantic_vocab_size, semantic_dim)

    def forward(self, numbers: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.math.encode(numbers), self.semantic(token_ids)), dim=-1)
