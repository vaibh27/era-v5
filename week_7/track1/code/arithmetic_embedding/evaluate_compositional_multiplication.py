"""Evaluate multiplication composed from learned product and learned carry cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from arithmetic.models.addition import RecurrentLatentAdder
from arithmetic.models.multiplication import CompositionalLatentMultiplier, SchoolbookLatentMultiplier
from arithmetic.reference.limbs import limbs_to_int
from train_multiplication import padded_numbers


@torch.no_grad()
def evaluate(model, base, limbs, examples, batch_size, seed, device):
    model.eval(); rng = random.Random(seed); exact = 0
    for start in range(0, examples, batch_size):
        size = min(batch_size, examples - start)
        a = padded_numbers(base, limbs, size, rng, device); b = padded_numbers(base, limbs, size, rng, device)
        result = model(a, b).tolist()
        exact += sum(limbs_to_int(r, base) == limbs_to_int(x, base) * limbs_to_int(y, base) for r, x, y in zip(result, a.tolist(), b.tolist()))
    return {"examples": examples, "integer_exact_accuracy": exact / examples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=int, default=10); p.add_argument("--dim", type=int, default=16); p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--adder", type=Path, default=Path("artifacts/latent_addition_base10.pt")); p.add_argument("--product", type=Path, default=Path("artifacts/latent_multiplication_base10.pt"))
    p.add_argument("--examples", type=int, default=1000); p.add_argument("--batch-size", type=int, default=128); p.add_argument("--seed", type=int, default=7); p.add_argument("--out", type=Path, default=Path("artifacts/compositional_multiplication_base10.json"))
    args = p.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adder = RecurrentLatentAdder(args.base, args.dim, args.hidden_dim).to(device); adder.load_state_dict(torch.load(args.adder, map_location=device, weights_only=True))
    product = SchoolbookLatentMultiplier(args.base, args.dim, args.hidden_dim).to(device); product.load_state_dict(torch.load(args.product, map_location=device, weights_only=True))
    model = CompositionalLatentMultiplier(product, adder).to(device)
    result = {"method": "learned_product_plus_learned_recurrent_carry", "base": args.base, "metrics": {str(n): evaluate(model, args.base, n, args.examples, args.batch_size, args.seed + n, device) for n in (1, 2, 4, 8)}}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
