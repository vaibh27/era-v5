"""Train a shared recurrent latent carry cell for variable-length addition."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from arithmetic.data.generators import addition_batch, exhaustive_local_addition_batch, local_addition_batch
from arithmetic.evaluation.metrics import addition_metrics, length_generalization
from arithmetic.models.addition import RecurrentLatentAdder
from arithmetic.training.losses import addition_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--min-limbs", type=int, default=1)
    parser.add_argument("--max-limbs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--local-pretrain-steps", type=int, default=0)
    parser.add_argument("--local-exhaustive", action="store_true", help="Use every local transition per pretraining step; intended for small bases.")
    parser.add_argument("--hard-case-repeat", type=int, default=0, help="Extra copies of the maximal-sum local transition during exhaustive pretraining.")
    parser.add_argument("--resume", type=Path, help="Load an existing RecurrentLatentAdder checkpoint before training.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--latent-weight", type=float, default=0.1)
    parser.add_argument("--eval-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("artifacts/latent_addition.json"))
    args = parser.parse_args()
    if args.min_limbs < 1 or args.max_limbs < args.min_limbs:
        parser.error("require 1 <= min-limbs <= max-limbs")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    model = RecurrentLatentAdder(args.base, args.dim, args.hidden_dim).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    best_state = None
    best_accuracy = -1.0
    best_local_accuracy = -1.0

    # Learn the exact reusable local transition before asking recurrence to
    # compose it across positions. This never exposes whole-number inputs.
    for step in range(1, args.local_pretrain_steps + 1):
        model.train()
        target = exhaustive_local_addition_batch(args.base, device, args.hard_case_repeat) if args.local_exhaustive else local_addition_batch(args.base, args.batch_size, rng, device)
        output = model.cell(model.embedding(target["a"]), model.embedding(target["b"]), target["carry_in"])
        limb = torch.nn.functional.cross_entropy(output.result_logits, target["result"])
        carry = torch.nn.functional.cross_entropy(output.carry_logits, target["carry"])
        target_latent = model.embedding(target["result"]).detach()
        latent = torch.nn.functional.mse_loss(
            torch.nn.functional.normalize(output.result_latent, dim=-1),
            torch.nn.functional.normalize(target_latent, dim=-1),
        )
        loss = limb + carry + args.latent_weight * latent
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == args.local_pretrain_steps:
            record = {"local_pretrain_step": step, "loss": float(loss.item()), "limb": float(limb.item()), "carry": float(carry.item())}
            if args.local_exhaustive:
                limb_accuracy = float((output.result_logits.argmax(dim=-1) == target["result"]).float().mean().item())
                carry_accuracy = float((output.carry_logits.argmax(dim=-1) == target["carry"]).float().mean().item())
                record["local_limb_exact_accuracy"] = limb_accuracy
                record["local_carry_exact_accuracy"] = carry_accuracy
                local_accuracy = min(limb_accuracy, carry_accuracy)
                if local_accuracy > best_local_accuracy:
                    best_local_accuracy = local_accuracy
                    best_state = copy.deepcopy(model.state_dict())
            print(json.dumps(record))

    if args.local_pretrain_steps:
        pretrain_metrics = addition_metrics(model, args.base, args.min_limbs, args.max_limbs, args.eval_examples, args.batch_size, args.seed + 5_000, device)
        best_accuracy = float(pretrain_metrics["integer_exact_accuracy"])
        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
        print(json.dumps({"after_local_pretraining": pretrain_metrics}))

    for step in range(1, args.steps + 1):
        model.train()
        target = addition_batch(args.base, args.min_limbs, args.max_limbs, args.batch_size, rng, device)
        output = model(target["a"], target["b"], teacher_carries=target["carry"])
        losses = addition_loss(model, output, target, args.latent_weight)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 200 == 0 or step == args.steps:
            in_distribution = addition_metrics(model, args.base, args.min_limbs, args.max_limbs, args.eval_examples, args.batch_size, args.seed + step, device)
            record = {"step": step, "loss": {key: float(value.item()) for key, value in losses.items()}, "in_distribution": in_distribution}
            history.append(record)
            if in_distribution["integer_exact_accuracy"] > best_accuracy:
                best_accuracy = float(in_distribution["integer_exact_accuracy"])
                best_state = copy.deepcopy(model.state_dict())
            print(json.dumps(record))

    if best_state is not None:
        model.load_state_dict(best_state)
    extrapolation = length_generalization(model, args.base, args.max_limbs, [1, 2, 4, 8, 16], args.eval_examples, args.batch_size, args.seed + 10_000, device)
    result = {"config": vars(args), "device": str(device), "best_in_distribution_integer_accuracy": best_accuracy, "best_local_transition_accuracy": best_local_accuracy if args.local_exhaustive else None, "history": history, "length_generalization": extrapolation}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), args.out.with_suffix(".pt"))
    print(json.dumps({"length_generalization": extrapolation}, indent=2))


if __name__ == "__main__":
    main()
