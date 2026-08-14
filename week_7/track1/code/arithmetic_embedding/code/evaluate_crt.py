"""Exhaustively evaluate the exact CRT arithmetic embedding.

This is the constructive baseline for the bounded-domain theorem in report.md.
Unlike train.py, it has no learned parameters: every latent arithmetic operation
is the component-wise ring operation prescribed by the CRT construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crt import CRTSystem, verify_exhaustive


def parse_moduli(value: str) -> tuple[int, ...]:
    try:
        moduli = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Moduli must be comma-separated integers.") from exc
    if not moduli:
        raise argparse.ArgumentTypeError("At least one modulus is required.")
    return moduli


def evaluate(system: CRTSystem) -> dict[str, int | float]:
    """Check every operand pair in Z_M and report exact latent accuracy."""
    M = system.capacity
    add_correct = 0
    mul_correct = 0
    round_trip_correct = 0

    for x in range(M):
        ex = system.encode(x)
        round_trip_correct += int(system.decode(ex) == x)
        for y in range(M):
            ey = system.encode(y)
            add_correct += int(system.add(ex, ey) == system.encode((x + y) % M))
            mul_correct += int(system.mul(ex, ey) == system.encode((x * y) % M))

    pairs = M * M
    return {
        "domain_size": M,
        "pairs": pairs,
        "round_trip_exact_accuracy": round_trip_correct / M,
        "add_exact_accuracy": add_correct / pairs,
        "mul_exact_accuracy": mul_correct / pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moduli", type=parse_moduli, default=(5, 7, 11))
    parser.add_argument("--out", type=Path, default=Path("artifacts/crt_exact.json"))
    args = parser.parse_args()

    system = CRTSystem(args.moduli)
    # The helper additionally checks injectivity and both homomorphism laws.
    verify_exhaustive(system)
    metrics = evaluate(system)
    result = {
        "method": "exact_crt_rns",
        "moduli": list(system.moduli),
        "capacity": system.capacity,
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
