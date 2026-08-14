"""Little-endian limb representation used by every variable-length experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _validate_base(base: int) -> None:
    if base < 2:
        raise ValueError("base must be at least 2")


def normalize_limbs(limbs: Iterable[int], base: int) -> list[int]:
    """Normalize non-negative little-endian limbs and remove leading zero limbs."""
    _validate_base(base)
    values = list(limbs)
    if not values:
        return [0]
    carry = 0
    result: list[int] = []
    for value in values:
        total = int(value) + carry
        if total < 0:
            raise ValueError("normalize_limbs only supports non-negative values")
        result.append(total % base)
        carry = total // base
    while carry:
        result.append(carry % base)
        carry //= base
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def int_to_limbs(x: int, base: int) -> list[int]:
    """Represent a non-negative Python integer as canonical little-endian limbs."""
    _validate_base(base)
    if x < 0:
        raise ValueError("int_to_limbs expects a non-negative integer; use SignedBigInt for signs")
    if x == 0:
        return [0]
    limbs: list[int] = []
    while x:
        limbs.append(x % base)
        x //= base
    return limbs


def limbs_to_int(limbs: Iterable[int], base: int) -> int:
    """Decode canonical or padded little-endian limbs."""
    _validate_base(base)
    value = 0
    place = 1
    for limb in limbs:
        if not 0 <= int(limb) < base:
            raise ValueError(f"limb {limb} is outside [0, {base})")
        value += int(limb) * place
        place *= base
    return value


@dataclass(frozen=True)
class SignedBigInt:
    """Sign-separated representation reserved for the subtraction stage."""

    sign: int
    limbs: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.sign not in (-1, 0, 1):
            raise ValueError("sign must be -1, 0, or 1")
