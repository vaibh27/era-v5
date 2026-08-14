from __future__ import annotations

import random

import torch

from arithmetic.reference.limbs import int_to_limbs


def _random_number(rng: random.Random, base: int, limb_count: int) -> int:
    limbs = [rng.randrange(base) for _ in range(limb_count)]
    if limb_count > 1:
        limbs[-1] = rng.randrange(1, base)
    return sum(limb * base**index for index, limb in enumerate(limbs))


def addition_batch(base: int, min_limbs: int, max_limbs: int, batch_size: int, rng: random.Random, device: torch.device) -> dict[str, torch.Tensor]:
    """Create padded limb sequences plus supervised result and carry transitions."""
    steps = max_limbs + 1  # final zero-input step emits a possible terminal carry
    a_rows, b_rows, result_rows, carry_rows = [], [], [], []
    for _ in range(batch_size):
        a = _random_number(rng, base, rng.randint(min_limbs, max_limbs))
        b = _random_number(rng, base, rng.randint(min_limbs, max_limbs))
        a_limbs = int_to_limbs(a, base) + [0] * steps
        b_limbs = int_to_limbs(b, base) + [0] * steps
        result_limbs = int_to_limbs(a + b, base) + [0] * steps
        carry = 0
        carries = []
        for position in range(steps):
            total = a_limbs[position] + b_limbs[position] + carry
            carry = total // base
            carries.append(carry)
        a_rows.append(a_limbs[:steps])
        b_rows.append(b_limbs[:steps])
        result_rows.append(result_limbs[:steps])
        carry_rows.append(carries)
    return {
        "a": torch.tensor(a_rows, dtype=torch.long, device=device),
        "b": torch.tensor(b_rows, dtype=torch.long, device=device),
        "result": torch.tensor(result_rows, dtype=torch.long, device=device),
        "carry": torch.tensor(carry_rows, dtype=torch.long, device=device),
    }


def local_addition_batch(base: int, batch_size: int, rng: random.Random, device: torch.device) -> dict[str, torch.Tensor]:
    """Uniform supervision for the reusable one-limb addition transition."""
    a = torch.tensor([rng.randrange(base) for _ in range(batch_size)], device=device)
    b = torch.tensor([rng.randrange(base) for _ in range(batch_size)], device=device)
    carry_in = torch.tensor([rng.randrange(2) for _ in range(batch_size)], device=device)
    total = a + b + carry_in
    return {
        "a": a.long(),
        "b": b.long(),
        "carry_in": carry_in.long(),
        "result": torch.remainder(total, base).long(),
        "carry": torch.div(total, base, rounding_mode="floor").long(),
    }


def local_subtraction_batch(base: int, batch_size: int, rng: random.Random, device: torch.device) -> dict[str, torch.Tensor]:
    """Uniform local transitions for result=(a-b-borrow_in) mod base."""
    a = torch.tensor([rng.randrange(base) for _ in range(batch_size)], device=device)
    b = torch.tensor([rng.randrange(base) for _ in range(batch_size)], device=device)
    borrow_in = torch.tensor([rng.randrange(2) for _ in range(batch_size)], device=device)
    total = a - b - borrow_in
    return {
        "a": a.long(), "b": b.long(), "borrow_in": borrow_in.long(),
        "result": torch.remainder(total, base).long(), "borrow": (total < 0).long(),
    }


def subtraction_batch(base: int, min_limbs: int, max_limbs: int, batch_size: int, rng: random.Random, device: torch.device) -> dict[str, torch.Tensor]:
    """Padded unsigned subtraction examples with every borrow transition labeled."""
    steps = max_limbs + 1
    a_rows, b_rows, result_rows, borrow_rows = [], [], [], []
    for _ in range(batch_size):
        a = _random_number(rng, base, rng.randint(min_limbs, max_limbs))
        b = _random_number(rng, base, rng.randint(min_limbs, max_limbs))
        if b > a:
            a, b = b, a
        a_limbs = int_to_limbs(a, base) + [0] * steps
        b_limbs = int_to_limbs(b, base) + [0] * steps
        result_limbs = int_to_limbs(a - b, base) + [0] * steps
        borrow, borrows = 0, []
        for position in range(steps):
            total = a_limbs[position] - b_limbs[position] - borrow
            borrow = int(total < 0)
            borrows.append(borrow)
        a_rows.append(a_limbs[:steps]); b_rows.append(b_limbs[:steps])
        result_rows.append(result_limbs[:steps]); borrow_rows.append(borrows)
    return {"a": torch.tensor(a_rows, dtype=torch.long, device=device), "b": torch.tensor(b_rows, dtype=torch.long, device=device), "result": torch.tensor(result_rows, dtype=torch.long, device=device), "borrow": torch.tensor(borrow_rows, dtype=torch.long, device=device)}


def local_product_batch(base: int, batch_size: int, rng: random.Random, device: torch.device) -> dict[str, torch.Tensor]:
    a = torch.tensor([rng.randrange(base) for _ in range(batch_size)], device=device)
    b = torch.tensor([rng.randrange(base) for _ in range(batch_size)], device=device)
    product = a * b
    return {"a": a.long(), "b": b.long(), "low": (product % base).long(), "high": (product // base).long()}


def exhaustive_local_addition_batch(base: int, device: torch.device, hard_case_repeat: int = 0) -> dict[str, torch.Tensor]:
    """Every possible local addition transition, for small-base diagnostics."""
    values = torch.arange(base, device=device)
    a, b, carry_in = torch.meshgrid(values, values, torch.arange(2, device=device), indexing="ij")
    a, b, carry_in = a.reshape(-1), b.reshape(-1), carry_in.reshape(-1)
    total = a + b + carry_in
    batch = {
        "a": a.long(),
        "b": b.long(),
        "carry_in": carry_in.long(),
        "result": torch.remainder(total, base).long(),
        "carry": torch.div(total, base, rounding_mode="floor").long(),
    }
    if hard_case_repeat:
        # The largest possible sum is a critical boundary: it must emit both a
        # non-zero carry and the largest residue. Repeat it only as a training
        # weight; the evaluator still scores every transition exactly once.
        edge = torch.full((hard_case_repeat,), base - 1, device=device, dtype=torch.long)
        carry_in = torch.ones(hard_case_repeat, device=device, dtype=torch.long)
        batch = {key: torch.cat((value, extra), dim=0) for key, (value, extra) in {
            "a": (batch["a"], edge),
            "b": (batch["b"], edge),
            "carry_in": (batch["carry_in"], carry_in),
            "result": (batch["result"], edge),
            "carry": (batch["carry"], carry_in),
        }.items()}
    return batch
