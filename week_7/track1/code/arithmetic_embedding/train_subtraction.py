"""Train and evaluate a shared recurrent latent borrow cell for unsigned subtraction."""

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

from arithmetic.data.generators import local_subtraction_batch, subtraction_batch
from arithmetic.models.subtraction import RecurrentLatentSubtractor
from arithmetic.reference.limbs import limbs_to_int


@torch.no_grad()
def evaluate(model, base, min_limbs, max_limbs, examples, batch_size, seed, device):
    model.eval(); rng = random.Random(seed)
    exact = limbs = borrows = total = 0
    for start in range(0, examples, batch_size):
        target = subtraction_batch(base, min_limbs, max_limbs, min(batch_size, examples - start), rng, device)
        out = model(target["a"], target["b"])
        predicted = out.result_logits.argmax(-1); predicted_borrow = out.borrow_logits.argmax(-1)
        limbs += int((predicted == target["result"]).sum()); borrows += int((predicted_borrow == target["borrow"]).sum()); total += target["result"].numel()
        exact += sum(limbs_to_int(p, base) == limbs_to_int(t, base) for p, t in zip(predicted.tolist(), target["result"].tolist()))
    return {"examples": examples, "limb_exact_accuracy": limbs / total, "borrow_exact_accuracy": borrows / total, "integer_exact_accuracy": exact / examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=int, default=10); parser.add_argument("--dim", type=int, default=16); parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--min-limbs", type=int, default=1); parser.add_argument("--max-limbs", type=int, default=4)
    parser.add_argument("--local-pretrain-steps", type=int, default=5000); parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256); parser.add_argument("--lr", type=float, default=2e-3); parser.add_argument("--eval-examples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7); parser.add_argument("--out", type=Path, default=Path("artifacts/latent_subtraction_base10.json"))
    args = parser.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rng = random.Random(args.seed)
    model = RecurrentLatentSubtractor(args.base, args.dim, args.hidden_dim).to(device); opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(1, args.local_pretrain_steps + 1):
        target = local_subtraction_batch(args.base, args.batch_size, rng, device)
        out = model.cell(model.embedding(target["a"]), model.embedding(target["b"]), target["borrow_in"])
        loss = F.cross_entropy(out.result_logits, target["result"]) + F.cross_entropy(out.carry_logits, target["borrow"])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    history = []
    for step in range(1, args.steps + 1):
        target = subtraction_batch(args.base, args.min_limbs, args.max_limbs, args.batch_size, rng, device)
        out = model(target["a"], target["b"], teacher_borrows=target["borrow"])
        loss = F.cross_entropy(out.result_logits.flatten(0, 1), target["result"].flatten()) + F.cross_entropy(out.borrow_logits.flatten(0, 1), target["borrow"].flatten())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step == 1 or step % 200 == 0 or step == args.steps:
            record = {"step": step, "loss": float(loss.item()), "in_distribution": evaluate(model, args.base, args.min_limbs, args.max_limbs, args.eval_examples, args.batch_size, args.seed + step, device)}
            history.append(record); print(json.dumps(record))
    lengths = {str(n): evaluate(model, args.base, n, n, args.eval_examples, args.batch_size, args.seed + n, device) for n in (1, 2, 4, 8, 16)}
    result = {"config": vars(args), "history": history, "length_generalization": lengths}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    torch.save(model.state_dict(), args.out.with_suffix(".pt")); print(json.dumps({"length_generalization": lengths}, indent=2))


if __name__ == "__main__":
    main()
