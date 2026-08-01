#!/usr/bin/env python3
"""Nano-RegMix proxy — implements wiki/60-proxy-experiments.md §3.

Protocol (RegMix-in-miniature): train ~20 tiny decoders on Dirichlet-sampled
5-lane mixtures at a fixed token budget, measure per-lane held-out loss, fit a
regressor composite<-mixture, predict the best mixture, confirm with fresh runs.
Honest scope: proof-of-method; rank signals only (see wiki page §3-§4).

Subcommands: prep | tokenizer | tokenize | train | sweep | fit
"""
import argparse, hashlib, json, math, os, random, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
LANES = ["web", "code", "math", "indic_native", "indic_translated"]

# H_A / H_B collapsed to 5 lanes (mapping documented in wiki/60 §3):
#   web absorbs science+india-first; code absorbs agentic; math absorbs reasoning;
#   Indic 22% splits ~8 native / ~14 translated (H_A) per the tier ledger in wiki/20.
PRIOR_HA = {"web": 0.48, "code": 0.18, "math": 0.12, "indic_native": 0.08, "indic_translated": 0.14}
PRIOR_HB = {"web": 0.55, "code": 0.18, "math": 0.14, "indic_native": 0.08, "indic_translated": 0.05}

# Composite C (wiki/60 §1) collapsed to 5 lanes and renormalized (agentic folded into code):
# Indic .30 (70/30 native/translated), code .25, math+reasoning .20, English-knowledge .15 -> /0.90
CWEIGHTS = {"indic_native": 0.2333, "indic_translated": 0.1000, "code": 0.2778,
            "math": 0.2222, "web": 0.1667}
COLLAPSE_GUARD = 0.90  # no lane's score may fall below 90% of its best across runs

CHAR_BUDGET = {  # raw chars collected per lane (aiming for >=40M BPE tokens each)
    "web": 260_000_000, "code": 240_000_000, "math": 240_000_000,
    "indic_native": 200_000_000, "indic_translated": 200_000_000,
}

# HF sources per lane; (dataset, kwargs, text_field) tried in order.
SOURCES = {
    "web": [("HuggingFaceFW/fineweb-edu", {"name": "sample-10BT"}, "text")],
    "code": [("bigcode/the-stack-smol", {}, "content"),
             ("codeparrot/codeparrot-clean-valid", {}, "content")],
    "math": [("open-web-math/open-web-math", {}, "text")],
    "indic_native": [("ai4bharat/sangraha", {"data_dir": "verified/hin"}, "text")],
    "indic_translated": [("ai4bharat/sangraha", {"data_dir": "synthetic/hin_Deva"}, "text"),
                         ("ai4bharat/sangraha", {"data_dir": "synthetic/hin"}, "text"),
                         ("ai4bharat/sangraha", {"data_dir": "unverified/hin"}, "text")],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- data prep
def cmd_prep(args):
    from datasets import load_dataset
    DATA.mkdir(exist_ok=True)
    for lane in LANES:
        out = DATA / f"{lane}.txt"
        if out.exists() and out.stat().st_size > 0.9 * CHAR_BUDGET[lane]:
            log(f"{lane}: already prepared ({out.stat().st_size/1e6:.0f} MB), skip")
            continue
        done = False
        for ds_name, kwargs, field in SOURCES[lane]:
            try:
                log(f"{lane}: streaming {ds_name} {kwargs} ...")
                ds = load_dataset(ds_name, split="train", streaming=True, **kwargs)
                n = 0
                with open(out, "w", encoding="utf-8") as f:
                    for ex in ds:
                        txt = ex.get(field) or ""
                        if len(txt) < 200:
                            continue
                        f.write(txt)
                        f.write("\n\n")
                        n += len(txt) + 2
                        if n >= CHAR_BUDGET[lane]:
                            break
                log(f"{lane}: wrote {n/1e6:.0f} MB from {ds_name}")
                done = True
                break
            except Exception as e:  # noqa: BLE001 — fall through to next source
                log(f"{lane}: {ds_name} failed ({type(e).__name__}: {e}); trying fallback")
        if not done:
            raise SystemExit(f"FATAL: no source worked for lane {lane}")


# ---------------------------------------------------------------- tokenizer
def cmd_tokenizer(args):
    from tokenizers import ByteLevelBPETokenizer
    sample = DATA / "tok_sample.txt"
    with open(sample, "w", encoding="utf-8") as f:
        for lane in LANES:  # balanced 8 MB per lane
            with open(DATA / f"{lane}.txt", encoding="utf-8") as src:
                f.write(src.read(8_000_000))
            f.write("\n\n")
    tok = ByteLevelBPETokenizer()
    tok.train(files=[str(sample)], vocab_size=16384, min_frequency=2)
    tok.save(str(DATA / "tokenizer.json"))
    log("tokenizer trained: 16384 byte-level BPE -> data/tokenizer.json")


def cmd_tokenize(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    for lane in LANES:
        out = DATA / f"{lane}.npy"
        if out.exists():
            log(f"{lane}: already tokenized, skip")
            continue
        ids = []
        with open(DATA / f"{lane}.txt", encoding="utf-8") as f:
            while True:
                chunk = f.read(8_000_000)
                if not chunk:
                    break
                ids.extend(tok.encode(chunk).ids)
        arr = np.array(ids, dtype=np.uint16)
        np.save(out, arr)
        log(f"{lane}: {len(arr)/1e6:.1f}M tokens -> {out.name}")


# ---------------------------------------------------------------- model
def build_model(vocab=16384, d=192, n_layer=6, n_head=6, ctx=512):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d, bias=False)
            self.proj = nn.Linear(d, d, bias=False)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
            q, k, v = (t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.proj(y)
            return x + self.mlp(self.ln2(x))

    class NanoGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(vocab, d)
            self.wpe = nn.Embedding(ctx, d)
            self.blocks = nn.ModuleList(Block() for _ in range(n_layer))
            self.lnf = nn.LayerNorm(d)
            self.head = nn.Linear(d, vocab, bias=False)
            self.head.weight = self.wte.weight  # tied
            # GPT-2-style init: without this, tied N(0,1) embeddings give
            # step-0 logits ~sqrt(d) too large (CE ~125 instead of ~ln(V)=9.7)
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
            for b in self.blocks:
                x = b(x)
            logits = self.head(self.lnf(x))
            if targets is None:
                return logits, None
            loss = F.cross_entropy(logits.view(-1, vocab), targets.reshape(-1))
            return logits, loss

    return NanoGPT()


class LaneSampler:
    """Samples (x, y) windows from per-lane token arrays according to a mixture."""

    def __init__(self, mixture, ctx, seed, split="train"):
        self.arrays, self.mix, self.ctx = {}, mixture, ctx
        self.rng = np.random.default_rng(seed)
        for lane in LANES:
            a = np.load(DATA / f"{lane}.npy", mmap_mode="r")
            cut = int(len(a) * 0.95)
            self.arrays[lane] = a[:cut] if split == "train" else a[cut:]
        self.lanes = LANES
        self.p = np.array([mixture[l] for l in LANES], dtype=np.float64)
        self.p /= self.p.sum()

    def batch(self, bs):
        import torch
        xs = np.empty((bs, self.ctx + 1), dtype=np.int64)
        lane_idx = self.rng.choice(len(self.lanes), size=bs, p=self.p)
        for i, li in enumerate(lane_idx):
            a = self.arrays[self.lanes[li]]
            start = int(self.rng.integers(0, len(a) - self.ctx - 2))
            xs[i] = a[start:start + self.ctx + 1]
        t = torch.from_numpy(xs)
        return t[:, :-1], t[:, 1:]


def eval_lanes(model, device, ctx, seed=1234, n_batches=20, bs=16):
    """Per-lane held-out CE loss on a fixed eval sample (same seed for every run)."""
    import torch
    model.eval()
    out = {}
    with torch.no_grad():
        for lane in LANES:
            s = LaneSampler({l: (1.0 if l == lane else 0.0) for l in LANES}, ctx,
                            seed=seed, split="eval")
            tot = 0.0
            for _ in range(n_batches):
                x, y = s.batch(bs)
                _, loss = model(x.to(device), y.to(device))
                tot += loss.item()
            out[lane] = tot / n_batches
    model.train()
    return out


def cmd_train(args):
    import torch
    RUNS.mkdir(exist_ok=True)
    mixture = json.loads(args.mixture) if args.mixture else PRIOR_HA
    assert abs(sum(mixture.values()) - 1.0) < 1e-3, "mixture must sum to 1"
    name = args.name
    result_path = RUNS / f"{name}.json"
    if result_path.exists():
        log(f"{name}: already done ({result_path.name}), skip")
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    seed = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % (2**31)
    torch.manual_seed(seed)
    # bs=16: peak memory is dominated by the (bs*ctx, 16384) logits tensor
    # (~0.5GB fp32) + autograd, not the 6M-param model — 16GB machines swap at bs=32.
    ctx, bs = 512, 16
    model = build_model(ctx=ctx).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    steps = max(1, args.tokens // (bs * ctx))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1,
                            betas=(0.9, 0.95))
    warmup = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(warmup, 1), 1.0)
        * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0)))))
    sampler = LaneSampler(mixture, ctx, seed=seed, split="train")
    log(f"{name}: {n_params/1e6:.1f}M params, {steps} steps ({args.tokens/1e6:.0f}M tok), "
        f"device={device}, mix={ {k: round(v,3) for k,v in mixture.items()} }")
    t0 = time.time()
    for step in range(steps):
        x, y = sampler.batch(bs)
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == steps - 1:
            log(f"{name}: step {step}/{steps} loss {loss.item():.3f} "
                f"({(step+1)*bs*ctx/max(time.time()-t0,1e-9)/1e3:.0f}K tok/s)")
    lane_losses = eval_lanes(model, device, ctx)
    row = {"name": name, "mixture": mixture, "lane_losses": lane_losses,
           "tokens": args.tokens, "params": n_params, "seconds": round(time.time() - t0, 1)}
    result_path.write_text(json.dumps(row) + "\n")  # per-run file: parallel-safe
    log(f"{name}: DONE lane_losses={ {k: round(v,3) for k,v in lane_losses.items()} }")


# ---------------------------------------------------------------- sweep & fit
def sample_mixtures(n, seed=7):
    rng = np.random.default_rng(seed)
    prior = np.array([PRIOR_HA[l] for l in LANES])
    out = []
    while len(out) < n:
        m = rng.dirichlet(prior * 2.5)  # diverse but prior-centered
        if m.min() < 0.02:  # keep every lane minimally present (floor analogue)
            continue
        out.append({l: round(float(v), 4) for l, v in zip(LANES, m)})
    return out


def cmd_sweep(args):
    runs = [("HA_prior", PRIOR_HA), ("HB_prior", PRIOR_HB)]
    runs += [(f"mix{i:02d}", m) for i, m in enumerate(sample_mixtures(args.n))]
    shard_i, shard_k = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
    for idx, (name, mix) in enumerate(runs):
        if idx % shard_k != shard_i:
            continue
        ns = argparse.Namespace(name=name, mixture=json.dumps(mix), tokens=args.tokens)
        cmd_train(ns)


def composite(rows):
    """score_lane = best_loss_lane / loss_lane (<=1, higher better); C = sum w*score."""
    best = {l: min(r["lane_losses"][l] for r in rows) for l in LANES}
    for r in rows:
        scores = {l: best[l] / r["lane_losses"][l] for l in LANES}
        r["scores"] = scores
        r["C"] = sum(CWEIGHTS[l] * scores[l] for l in LANES)
        r["collapsed"] = [l for l in LANES if scores[l] < COLLAPSE_GUARD]
    return rows


def cmd_fit(args):
    from sklearn.linear_model import Ridge
    allrows = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))
               if p.name != "fit_report.json" and not p.name.startswith("smoke")]
    # Design runs = the sweep (HA/HB/mixNN). Confirmations are held OUT of the
    # normalizer + regression so a degenerate confirm can't move the goalposts;
    # they are scored out-of-sample against the fixed design-run baseline.
    design = [r for r in allrows if not r["name"].startswith("confirm")]
    confirms = [r for r in allrows if r["name"].startswith("confirm")]
    # FIXED normalizer: best per-lane loss over design runs only.
    best_obs = {l: min(r["lane_losses"][l] for r in design) for l in LANES}

    def score(r):
        sc = {l: min(best_obs[l] / r["lane_losses"][l], 1.2) for l in LANES}
        r["C"] = sum(CWEIGHTS[l] * sc[l] for l in LANES)
        r["collapsed"] = [l for l in LANES if sc[l] < COLLAPSE_GUARD]
        return r
    for r in design + confirms:
        score(r)
    design.sort(key=lambda r: -r["C"])

    # per-lane loss models: loss_l ~ ridge(log mixture), fit on DESIGN runs only
    X = np.array([[math.log(max(r["mixture"][l], 1e-3)) for l in LANES] for r in design])
    models = {}
    for l in LANES:
        y = np.array([r["lane_losses"][l] for r in design])
        models[l] = Ridge(alpha=1e-2).fit(X, y)
    # Predict over a simplex grid, but CLAMP the search to the sampled hull
    # (each lane within its observed [min,max]) so the reported optimum is an
    # interpolation, not a degenerate extrapolated corner. We also record the
    # UNCONSTRAINED argmax separately and label it as extrapolation.
    lo = {l: min(r["mixture"][l] for r in design) for l in LANES}
    hi = {l: max(r["mixture"][l] for r in design) for l in LANES}
    rng = np.random.default_rng(0)
    grid = rng.dirichlet(np.ones(len(LANES)), size=400_000)
    grid = grid[grid.min(axis=1) >= 0.02]
    Xg = np.log(np.maximum(grid, 1e-3))
    pred_losses = {l: models[l].predict(Xg) for l in LANES}
    sc = {l: np.minimum(best_obs[l] / pred_losses[l], 1.2) for l in LANES}
    C = sum(CWEIGHTS[l] * sc[l] for l in LANES)
    guard_ok = np.all([sc[l] >= COLLAPSE_GUARD for l in LANES], axis=0)
    in_hull = np.all([(grid[:, j] >= lo[l]) & (grid[:, j] <= hi[l])
                      for j, l in enumerate(LANES)], axis=0)

    def argmax_mix(mask):
        cc = np.where(mask, C, -1.0)
        i = int(np.argmax(cc))
        return ({l: round(float(grid[i, j]), 4) for j, l in enumerate(LANES)},
                round(float(C[i]), 4))
    pred_hull, C_hull = argmax_mix(guard_ok & in_hull)          # trustworthy
    pred_extrap, C_extrap = argmax_mix(guard_ok)                # cautionary

    report = {
        "n_design": len(design), "n_confirm": len(confirms),
        "normalizer": "best per-lane loss over design runs (fixed)",
        "cweights": CWEIGHTS, "collapse_guard": COLLAPSE_GUARD,
        "ranked_design": [{"name": r["name"], "C": round(r["C"], 4),
                           "collapsed": r["collapsed"], "mixture": r["mixture"]}
                          for r in design],
        "HA": next(({"C": r["C"], "collapsed": r["collapsed"]}
                    for r in design if r["name"] == "HA_prior"), None),
        "HB": next(({"C": r["C"], "collapsed": r["collapsed"]}
                    for r in design if r["name"] == "HB_prior"), None),
        "confirmations_out_of_sample": [
            {"name": r["name"], "C": round(r["C"], 4), "collapsed": r["collapsed"],
             "mixture": r["mixture"]} for r in confirms],
        "predicted_best_in_hull": {"mixture": pred_hull, "C": C_hull},
        "predicted_best_unconstrained": {"mixture": pred_extrap, "C": C_extrap,
            "note": "extrapolated corner — NOT trusted; see confirm_predicted"},
    }
    out = RUNS / "fit_report.json"
    out.write_text(json.dumps(report, indent=2))
    log(f"HA C={report['HA']['C']:.4f} collapsed={report['HA']['collapsed']}  |  "
        f"HB C={report['HB']['C']:.4f} collapsed={report['HB']['collapsed']}")
    log(f"top design run: {design[0]['name']} C={design[0]['C']:.4f} "
        f"collapsed={design[0]['collapsed']}")
    log(f"predicted-best (in hull): C={C_hull} {pred_hull}")
    log(f"predicted-best (unconstrained, DISTRUSTED): C={C_extrap} {pred_extrap}")
    log(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    sub.add_parser("tokenizer")
    sub.add_parser("tokenize")
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--mixture", default=None, help="JSON lane->weight (default H_A)")
    t.add_argument("--tokens", type=int, default=30_000_000)
    s = sub.add_parser("sweep")
    s.add_argument("--n", type=int, default=18)
    s.add_argument("--tokens", type=int, default=30_000_000)
    s.add_argument("--shard", default=None, help="i/k — this worker runs indices i mod k")
    sub.add_parser("fit")
    args = p.parse_args()
    {"prep": cmd_prep, "tokenizer": cmd_tokenizer, "tokenize": cmd_tokenize,
     "train": cmd_train, "sweep": cmd_sweep, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
