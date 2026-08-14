from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Sequence

import torch


@dataclass(frozen=True)
class CRTSystem:
    moduli: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.moduli:
            raise ValueError("At least one modulus is required.")
        if any(m <= 1 for m in self.moduli):
            raise ValueError("All moduli must be > 1.")
        for i, a in enumerate(self.moduli):
            for b in self.moduli[i + 1 :]:
                if gcd(a, b) != 1:
                    raise ValueError(f"Moduli must be pairwise coprime: {a}, {b}.")

    @property
    def capacity(self) -> int:
        result = 1
        for m in self.moduli:
            result *= m
        return result

    def encode(self, x: int) -> tuple[int, ...]:
        return tuple(x % m for m in self.moduli)

    def decode(self, residues: Sequence[int]) -> int:
        if len(residues) != len(self.moduli):
            raise ValueError("Residue count must match modulus count.")
        x = 0
        M = self.capacity
        for residue, modulus in zip(residues, self.moduli):
            residue %= modulus
            Mi = M // modulus
            inv = pow(Mi, -1, modulus)
            x += residue * Mi * inv
        return x % M

    def add(self, a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
        if len(a) != len(self.moduli) or len(b) != len(self.moduli):
            raise ValueError("Residue counts must match modulus count.")
        return tuple((x + y) % m for x, y, m in zip(a, b, self.moduli))

    def mul(self, a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
        if len(a) != len(self.moduli) or len(b) != len(self.moduli):
            raise ValueError("Residue counts must match modulus count.")
        return tuple((x * y) % m for x, y, m in zip(a, b, self.moduli))

    def encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Return residue coordinates as float tensor of shape [batch, k]."""
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        cols = [torch.remainder(x, m) for m in self.moduli]
        return torch.stack(cols, dim=-1).float()


def verify_exhaustive(system: CRTSystem) -> None:
    """Exhaustively verify injectivity, addition and multiplication on Z_M."""
    M = system.capacity
    codes = [system.encode(x) for x in range(M)]
    assert len(set(codes)) == M, "CRT encoding is not injective."

    for x in range(M):
        for y in range(M):
            add_lhs = system.encode((x + y) % M)
            add_rhs = system.add(system.encode(x), system.encode(y))
            assert add_lhs == add_rhs

            mul_lhs = system.encode((x * y) % M)
            mul_rhs = system.mul(system.encode(x), system.encode(y))
            assert mul_lhs == mul_rhs
