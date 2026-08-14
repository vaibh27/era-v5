"""Exhaustively evaluate the continuous-vector CRT arithmetic embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from continuous_crt import ContinuousCRTEmbedding
from evaluate_crt import parse_moduli


def evaluate(model: ContinuousCRTEmbedding) -> dict[str, int | float]:
    M = model.system.capacity
    numbers = torch.arange(M)
    embeddings = model.encode(numbers)
    round_trip = (model.decode(embeddings) == numbers).float().mean().item()

    add_correct = 0
    mul_correct = 0
    pair_count = 0
    # Vectorize one left operand against the complete right side.
    for x in range(M):
        ex = embeddings[x].expand(M, -1)
        add_targets = (x + numbers) % M
        mul_targets = (x * numbers) % M
        add_correct += int((model.decode(model.add(ex, embeddings)) == add_targets).sum().item())
        mul_correct += int((model.decode(model.mul(ex, embeddings)) == mul_targets).sum().item())
        pair_count += M

    return {
        "domain_size": M,
        "arithmetic_dim": model.dim,
        "pairs": pair_count,
        "round_trip_exact_accuracy": round_trip,
        "add_exact_accuracy": add_correct / pair_count,
        "mul_exact_accuracy": mul_correct / pair_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moduli", type=parse_moduli, default=(5, 7, 11))
    parser.add_argument("--out", type=Path, default=Path("artifacts/continuous_crt_exact.json"))
    args = parser.parse_args()

    model = ContinuousCRTEmbedding(args.moduli)
    result = {
        "method": "continuous_one_hot_crt_rns",
        "moduli": list(model.moduli),
        "capacity": model.system.capacity,
        "metrics": evaluate(model),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
