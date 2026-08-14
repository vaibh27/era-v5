from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import LearnedArithmeticEmbedding, contrastive_target_loss, separation_loss


def make_pairs(modulus: int, fraction: float, seed: int):
    rng = random.Random(seed)
    pairs = [(x, y) for x in range(modulus) for y in range(modulus)]
    rng.shuffle(pairs)
    n = int(len(pairs) * fraction)
    return pairs[:n], pairs[n:]


def batchify(pairs, batch_size: int, device: torch.device):
    x = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    y = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def evaluate(model, pairs, device, max_examples=None):
    model.eval()
    if max_examples is not None:
        pairs = pairs[:max_examples]
    total = 0
    exact_add = 0
    exact_mul = 0
    add_latent = 0.0
    mul_latent = 0.0
    with torch.no_grad():
        emb = model.normalized_embeddings()
        for start in range(0, len(pairs), 1024):
            chunk = pairs[start : start + 1024]
            if not chunk:
                continue
            x = torch.tensor([a for a, _ in chunk], dtype=torch.long, device=device)
            y = torch.tensor([b for _, b in chunk], dtype=torch.long, device=device)
            out = model(x, y)
            add_targets = (x + y) % model.modulus
            mul_targets = (x * y) % model.modulus

            add_logits = F.normalize(out.add, dim=-1) @ emb.T
            mul_logits = F.normalize(out.mul, dim=-1) @ emb.T
            add_pred = add_logits.argmax(dim=-1)
            mul_pred = mul_logits.argmax(dim=-1)

            exact_add += int((add_pred == add_targets).sum().item())
            exact_mul += int((mul_pred == mul_targets).sum().item())
            add_target_vec = model.embedding(add_targets)
            mul_target_vec = model.embedding(mul_targets)
            add_latent += float(F.mse_loss(out.add, add_target_vec, reduction="sum").item())
            mul_latent += float(F.mse_loss(out.mul, mul_target_vec, reduction="sum").item())
            total += len(chunk)

    return {
        "examples": total,
        "add_exact_accuracy": exact_add / max(total, 1),
        "mul_exact_accuracy": exact_mul / max(total, 1),
        "add_latent_mse": add_latent / max(total, 1),
        "mul_latent_mse": mul_latent / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulus", type=int, default=101)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--add-hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--lambda-mse", type=float, default=1.0)
    parser.add_argument("--lambda-contrastive", type=float, default=1.0)
    parser.add_argument("--lambda-separation", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("artifacts/run.json"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pairs, test_pairs = make_pairs(args.modulus, args.train_fraction, args.seed)
    loader = batchify(train_pairs, args.batch_size, device)

    model = LearnedArithmeticEmbedding(
        modulus=args.modulus,
        dim=args.dim,
        add_hidden=args.add_hidden,
        bilinear_rank=args.rank,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            out = model(x, y)
            add_targets = (x + y) % args.modulus
            mul_targets = (x * y) % args.modulus
            add_target_vec = model.embedding(add_targets)
            mul_target_vec = model.embedding(mul_targets)

            loss_mse = F.mse_loss(out.add, add_target_vec) + F.mse_loss(out.mul, mul_target_vec)
            loss_contrastive = contrastive_target_loss(out.add, model.embedding.weight, add_targets)
            loss_contrastive = loss_contrastive + contrastive_target_loss(out.mul, model.embedding.weight, mul_targets)
            loss_sep = separation_loss(model.embedding.weight)
            loss = (
                args.lambda_mse * loss_mse
                + args.lambda_contrastive * loss_contrastive
                + args.lambda_separation * loss_sep
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train_metrics = evaluate(model, train_pairs, device, max_examples=5000)
            test_metrics = evaluate(model, test_pairs, device, max_examples=5000)
            record = {"epoch": epoch, "train_loss": running / len(loader), "train": train_metrics, "test": test_metrics}
            history.append(record)
            print(json.dumps(record))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out.with_suffix(".pt"))
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "device": str(device), "history": history}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
