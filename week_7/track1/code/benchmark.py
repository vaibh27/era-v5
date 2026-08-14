r"""
Track 1 benchmark for KroneckerV2Embedding = [ Kronecker text || CRT math ].

Runs the plan in BENCHMARK-PLAN.md:
  B1  arithmetic exactness & magnitude extrapolation   (the headline)
  B2  text reversibility (tokenizer-free round-trip)
  B4  capacity scaling (linear dim -> exponential exact range)

KroneckerV2's arithmetic operator is FIXED (their verified CRT tensors, zero trainable
params), so it is exact at every magnitude below capacity.  The learned baseline holds a
trainable vector per integer in a small vocab and decodes by nearest neighbour, so it has
NO representation for numbers outside its trained range -> it collapses out of range.

Writes results/benchmark.json and prints a summary table.
    pip install -r code/arithmetic_embedding/requirements.txt
    python code/benchmark.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "arithmetic_embedding" / "code"))

from kronecker_v2_embedding import KroneckerV2Embedding  # noqa: E402
from model import LearnedArithmeticEmbedding  # noqa: E402  (their train.py model)

SEED = 7


def set_seed(s: int = SEED) -> None:
    random.seed(s)
    torch.manual_seed(s)


# ============================================================== helpers
def sample_pairs(lo: int, hi: int, n: int, rng: random.Random) -> list[tuple[int, int]]:
    """n random operand pairs with both operands in [lo, hi)."""
    return [(rng.randrange(lo, hi), rng.randrange(lo, hi)) for _ in range(n)]


# ============================================================== B1: arithmetic
def kronecker_v2_arith(pairs: list[tuple[int, int]], E: KroneckerV2Embedding) -> dict:
    """Fixed CRT operator, no training. Exact whenever the result < capacity."""
    add_ok = mul_ok = 0
    for a, b in pairs:
        fa, fb = E.embed_features(str(a)), E.embed_features(str(b))
        add_ok += int(E.number_of(E.add(fa, fb)) == a + b)
        mul_ok += int(E.number_of(E.mul(fa, fb)) == a * b)
    n = max(len(pairs), 1)
    return {"add": add_ok / n, "mul": mul_ok / n}


def train_learned(modulus: int, train_range: int, epochs: int, device) -> LearnedArithmeticEmbedding:
    """Train their LearnedArithmeticEmbedding on integer add & mul over operands < train_range.

    modulus is the model's fixed vocab / output ring; it only has vectors for 0..modulus-1,
    and only rows touched by training get any gradient.
    """
    set_seed()
    model = LearnedArithmeticEmbedding(modulus=modulus, dim=16, add_hidden=64, bilinear_rank=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    pairs = [(a, b) for a in range(train_range) for b in range(train_range)]
    x = torch.tensor([a for a, _ in pairs], device=device)
    y = torch.tensor([b for _, b in pairs], device=device)
    add_t = (x + y) % modulus
    mul_t = (x * y) % modulus
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(x, y)
        loss = (
            F.mse_loss(out.add, model.embedding(add_t))
            + F.mse_loss(out.mul, model.embedding(mul_t))
            + F.cross_entropy(F.normalize(out.add, dim=-1) @ F.normalize(model.embedding.weight, dim=-1).T / 0.07, add_t)
            + F.cross_entropy(F.normalize(out.mul, dim=-1) @ F.normalize(model.embedding.weight, dim=-1).T / 0.07, mul_t)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def learned_arith(pairs: list[tuple[int, int]], model: LearnedArithmeticEmbedding, device) -> dict:
    """Nearest-neighbour decode over the learned vocab. Operands outside vocab = automatic miss."""
    model.eval()
    add_ok = mul_ok = 0
    with torch.no_grad():
        emb = F.normalize(model.embedding.weight, dim=-1)
        for a, b in pairs:
            if not (0 <= a < model.modulus and 0 <= b < model.modulus):
                continue  # no embedding for this operand -> miss both
            out = model(torch.tensor([a], device=device), torch.tensor([b], device=device))
            add_pred = int((F.normalize(out.add, dim=-1) @ emb.T).argmax().item())
            mul_pred = int((F.normalize(out.mul, dim=-1) @ emb.T).argmax().item())
            add_ok += int(add_pred == (a + b) % model.modulus and a + b < model.modulus)
            mul_ok += int(mul_pred == (a * b) % model.modulus and a * b < model.modulus)
    n = max(len(pairs), 1)
    return {"add": add_ok / n, "mul": mul_ok / n}


def linear_axis(pairs: list[tuple[int, int]], train_range: int) -> dict:
    """Value-on-axis: least-squares fit result = w.[a,b] + c on the train range, then evaluate.

    Add is linear so this extrapolates exactly; mul is not, so it fails off the train range.
    """
    tr = [(a, b) for a in range(train_range) for b in range(train_range)]
    A = torch.tensor([[a, b, 1.0] for a, b in tr])
    w_add = torch.linalg.lstsq(A, torch.tensor([[a + b] for a, b in tr], dtype=torch.float32)).solution
    w_mul = torch.linalg.lstsq(A, torch.tensor([[a * b] for a, b in tr], dtype=torch.float32)).solution
    add_ok = mul_ok = 0
    for a, b in pairs:
        v = torch.tensor([a, b, 1.0])
        add_ok += int(round(float(v @ w_add)) == a + b)
        mul_ok += int(round(float(v @ w_mul)) == a * b)
    n = max(len(pairs), 1)
    return {"add": add_ok / n, "mul": mul_ok / n}


def run_b1(E: KroneckerV2Embedding, device) -> dict:
    rng = random.Random(SEED)
    cap = E.capacity
    train_range = 64          # operands the learned/linear baselines are trained on
    modulus = 4096            # learned vocab; covers integer results in range (63*63=3969, 63+63=126)
    n = 400

    # in-range: operands < train_range (add/mul results stay < modulus -> integer == modular)
    in_pairs = sample_pairs(0, train_range, n, rng)
    # out-of-range (extrapolation), kept < capacity so KroneckerV2 has no wrap:
    add_out = sample_pairs(100_000, 700_000, n, rng)          # sums < 1.4M < capacity
    mul_out = sample_pairs(200, 1200, n, rng)                 # products < 1.44M < capacity

    model = train_learned(modulus, train_range, epochs=400, device=device)

    kron_params = 0  # the CRT operator is fixed: zero trainable arithmetic params
    learned_params = sum(p.numel() for p in model.parameters())

    return {
        "config": {
            "capacity": cap, "train_range": train_range, "learned_modulus": modulus,
            "n_pairs": n, "add_out_range": [100_000, 700_000], "mul_out_range": [200, 1200],
        },
        "trainable_params": {"kronecker_v2_operator": kron_params, "learned": learned_params},
        "in_range": {
            "kronecker_v2": kronecker_v2_arith(in_pairs, E),
            "learned": learned_arith(in_pairs, model, device),
            "linear_axis": linear_axis(in_pairs, train_range),
        },
        "out_of_range": {
            "kronecker_v2": {
                "add": kronecker_v2_arith(add_out, E)["add"],
                "mul": kronecker_v2_arith(mul_out, E)["mul"],
            },
            "learned": {
                "add": learned_arith(add_out, model, device)["add"],
                "mul": learned_arith(mul_out, model, device)["mul"],
            },
            "linear_axis": {
                "add": linear_axis(add_out, train_range)["add"],
                "mul": linear_axis(mul_out, train_range)["mul"],
            },
        },
    }


# ============================================================== B2: text reversibility
def run_b2(E: KroneckerV2Embedding) -> dict:
    in_vocab = ["cat", "buy", "apples", "for", "rupees", "the", "number", "hello", "world"]
    novel = ["quixotic", "borborygmus", "Zürich", "naïve", "6µm", "x" * 40]  # incl. accents & a >32-byte word

    def kron_roundtrip(words: list[str]) -> float:
        ok = 0
        for w in words:
            f = E.embed_features(w)
            ok += int(E.text_of(f, len(w.encode("utf-8"))) == w)
        return ok / max(len(words), 1)

    # learned nn.Embedding: a fixed vocab; decode = nearest row. OOV words have no id -> cannot round-trip.
    vocab = {w: i for i, w in enumerate(in_vocab)}
    emb = torch.nn.Embedding(len(vocab), 32)

    def learned_roundtrip(words: list[str]) -> float:
        ok = 0
        for w in words:
            if w not in vocab:
                continue  # out of vocabulary -> no token, round-trip fails
            v = emb(torch.tensor(vocab[w]))
            pred = int((F.normalize(v, dim=-1) @ F.normalize(emb.weight, dim=-1).T).argmax().item())
            ok += int(pred == vocab[w])
        return ok / max(len(words), 1)

    return {
        "note": "novel[-1] is a 40-char word: KroneckerV2 truncates at 32 bytes (honest limit).",
        "kronecker_v2_text": {"in_vocab": kron_roundtrip(in_vocab), "novel_oov": kron_roundtrip(novel)},
        "learned_embedding": {"in_vocab": learned_roundtrip(in_vocab), "novel_oov": learned_roundtrip(novel)},
    }


# ============================================================== B4: capacity scaling
def run_b4() -> dict:
    from crt import CRTSystem

    primes = (5, 7, 11, 13, 17, 19, 23, 29)
    rng = random.Random(SEED)
    rows = []
    for k in range(1, len(primes) + 1):
        E = KroneckerV2Embedding(moduli=primes[:k])
        cap = E.capacity
        # spot-check exact add/mul on random in-range pairs (kept below capacity)
        ok = True
        for _ in range(50):
            a, b = rng.randrange(cap), rng.randrange(cap)
            if a + b < cap:
                ok &= E.number_of(E.add(E.embed_features(str(a)), E.embed_features(str(b)))) == a + b
            if a * b < cap:
                ok &= E.number_of(E.mul(E.embed_features(str(a)), E.embed_features(str(b)))) == a * b
        rows.append({
            "n_primes": k, "moduli": list(primes[:k]),
            "math_dim": E.mdim, "capacity": cap,
            "log10_capacity": round(__import__("math").log10(cap), 3),
            "exact_spotcheck": bool(ok),
        })
    return {"note": "math_dim grows linearly (sum of primes); capacity grows exponentially (product).", "rows": rows}


# ============================================================== table + main
def fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def print_table(res: dict) -> None:
    b1 = res["B1"]
    print("\n" + "=" * 66)
    print("B1  Arithmetic exactness & magnitude extrapolation")
    print("=" * 66)
    print(f"{'model':<16}{'add in':>9}{'mul in':>9}{'add out':>10}{'mul out':>10}")
    for name in ("kronecker_v2", "learned", "linear_axis"):
        ir, orr = b1["in_range"][name], b1["out_of_range"][name]
        print(f"{name:<16}{fmt_pct(ir['add']):>9}{fmt_pct(ir['mul']):>9}"
              f"{fmt_pct(orr['add']):>10}{fmt_pct(orr['mul']):>10}")
    tp = b1["trainable_params"]
    print(f"trainable operator params: kronecker_v2={tp['kronecker_v2_operator']}  learned={tp['learned']}")

    b2 = res["B2"]
    print("\n" + "=" * 66)
    print("B2  Text reversibility (char-level round-trip)")
    print("=" * 66)
    print(f"{'model':<20}{'in-vocab':>12}{'novel/OOV':>12}")
    print(f"{'kronecker_v2 text':<20}{fmt_pct(b2['kronecker_v2_text']['in_vocab']):>12}"
          f"{fmt_pct(b2['kronecker_v2_text']['novel_oov']):>12}")
    print(f"{'learned embedding':<20}{fmt_pct(b2['learned_embedding']['in_vocab']):>12}"
          f"{fmt_pct(b2['learned_embedding']['novel_oov']):>12}")

    b4 = res["B4"]
    print("\n" + "=" * 66)
    print("B4  Capacity scaling (linear dim -> exponential exact range)")
    print("=" * 66)
    print(f"{'#primes':>8}{'math_dim':>10}{'capacity':>14}{'log10':>9}{'exact':>8}")
    for r in b4["rows"]:
        print(f"{r['n_primes']:>8}{r['math_dim']:>10}{r['capacity']:>14}"
              f"{r['log10_capacity']:>9}{str(r['exact_spotcheck']):>8}")
    print()


def main() -> None:
    set_seed()
    device = torch.device("cpu")
    E = KroneckerV2Embedding()
    print(f"KroneckerV2: dim={E.dim} (text {E.tdim} + math {E.mdim}), exact range 0..{E.capacity - 1}")

    results = {
        "model": "KroneckerV2Embedding [ Kronecker text || CRT math ]",
        "capacity": E.capacity,
        "seed": SEED,
        "B1": run_b1(E, device),
        "B2": run_b2(E),
        "B4": run_b4(),
    }

    out = HERE.parent / "results" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print_table(results)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
