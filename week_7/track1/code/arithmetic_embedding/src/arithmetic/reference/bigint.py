"""Python-integer oracle. This module must never be used by neural forward passes."""

from __future__ import annotations


def add(a: int, b: int) -> int:
    return a + b


def sub(a: int, b: int) -> int:
    return a - b


def mul(a: int, b: int) -> int:
    return a * b


def divmod_exact(a: int, b: int) -> tuple[int, int]:
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return divmod(a, b)
