"""Train a local latent product cell and evaluate schoolbook multiplication."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from arithmetic.data.generators import local_product_batch
from arithmetic.models.multiplication import SchoolbookLatentMultiplier
from arithmetic.reference.limbs import int_to_limbs, limbs_to_int


def padded_numbers(base, limbs, batch_size, rng, device):
    rows = []
    for _ in range(batch_size):
        values = [rng.randrange(base) for _ in range(limbs)]
        if limbs > 1: values[-1] = rng.randrange(1, base)
        rows.append(values)
    return torch.tensor(rows, dtype=torch.long, device=device)


@torch.no_grad()
def evaluate(model, base, limbs, examples, batch_size, seed, device):
    model.eval(); rng = random.Random(seed); exact = 0
    for start in range(0, examples, batch_size):
        size = min(batch_size, examples - start); a = padded_numbers(base, limbs, size, rng, device); b = padded_numbers(base, limbs, size, rng, device)
        result = model(a, b).tolist()
        exact += sum(limbs_to_int(r, base) == limbs_to_int(x, base) * limbs_to_int(y, base) for r, x, y in zip(result, a.tolist(), b.tolist()))
    return {"examples": examples, "integer_exact_accuracy": exact / examples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=int, default=10); p.add_argument("--dim", type=int, default=16); p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--steps", type=int, default=5000); p.add_argument("--batch-size", type=int, default=256); p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--eval-examples", type=int, default=1000); p.add_argument("--seed", type=int, default=7); p.add_argument("--out", type=Path, default=Path("artifacts/latent_multiplication_base10.json"))
    args = p.parse_args(); torch.manual_seed(args.seed); rng = random.Random(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchoolbookLatentMultiplier(args.base, args.dim, args.hidden_dim).to(device); opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for step in range(1, args.steps + 1):
        target = local_product_batch(args.base, args.batch_size, rng, device); out = model.product_cell(model.embedding(target["a"]), model.embedding(target["b"]))
        loss = F.cross_entropy(out.low_logits, target["low"]) + F.cross_entropy(out.high_logits, target["high"])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step == 1 or step % 500 == 0 or step == args.steps:
            record = {"step": step, "loss": float(loss.item()), "one_limb": evaluate(model, args.base, 1, args.eval_examples, args.batch_size, args.seed + step, device)}; history.append(record); print(json.dumps(record))
    lengths = {str(n): evaluate(model, args.base, n, args.eval_examples, args.batch_size, args.seed + n, device) for n in (1, 2, 4, 8)}
    result = {"config": vars(args), "history": history, "length_generalization": lengths}; args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2, default=str) + "\n"); torch.save(model.state_dict(), args.out.with_suffix(".pt")); print(json.dumps({"length_generalization": lengths}, indent=2))


if __name__ == "__main__": main()
