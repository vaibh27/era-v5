#!/usr/bin/env python3
"""Nano-RegMix proxy — multilingual Indic mixture experiment (Mac/MPS).

Design:
  - 6 lanes: web, code, math, reasoning, Indic-native, Indic-translated.
  - Indic is multilingual {hin, ben, tam} for both native and translated; per-language
    loss is reported as a diagnostic so cross-script generalization is measured.
  - H_A / H_B priors run at 3 seeds each -> composite mean +/- range and a native-collapse
    frequency. H_A = 22% Indic, H_B = 13% Indic (see ../../README.md §9.1).
  - 24K byte-level BPE so Bengali/Tamil scripts get real coverage.
Honest scope: ~7.5M-param rank signal, proof-of-method, not 40B evidence.

Subcommands: prep | tokenizer | tokenize | train | sweep | fit
"""
import argparse, hashlib, json, math, os, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
LANES = ["web", "code", "math", "reasoning", "indic_native", "indic_translated"]
INDIC_LANGS = ["hin", "ben", "tam"]

# 6-lane priors, spec-faithful (../README.md §1 collapsed: web absorbs
# science+india-first, code absorbs agentic; math and reasoning are separate):
#   web 45+2+3=50, code 16+2=18, math 6, reasoning 4, indic 8 native / 14 translated.
PRIOR_HA = {"web": 0.50, "code": 0.18, "math": 0.06, "reasoning": 0.04,
            "indic_native": 0.08, "indic_translated": 0.14}
# H_B: Indic total 22->13; the 9pp goes to web. Native share held equal (8%) so the
# contrast is the translated bulk (14->5), exactly the H_A/H_B fork in README §11.
PRIOR_HB = {"web": 0.59, "code": 0.18, "math": 0.06, "reasoning": 0.04,
            "indic_native": 0.08, "indic_translated": 0.05}

# Composite C (README §9.1): Indic .30 (70/30 native/translated), code .25, math .12,
# reasoning .08, English-knowledge .15; agentic .10 dropped (folded, untested) -> /0.90.
CWEIGHTS = {"indic_native": 0.2333, "indic_translated": 0.1000, "code": 0.2778,
            "math": 0.1333, "reasoning": 0.0889, "web": 0.1667}
COLLAPSE_GUARD = 0.90  # no lane's score may fall below 90% of its best across runs

CHAR_BUDGET = {  # raw chars/lane; indic budgets are split evenly across INDIC_LANGS
    "web": 260_000_000, "code": 240_000_000, "math": 240_000_000,
    "reasoning": 200_000_000, "indic_native": 210_000_000, "indic_translated": 210_000_000,
}


def _txt(field):
    return lambda ex: ex.get(field) or ""


def _conv(ex):  # OpenThoughts-114k: text lives in a conversations[] list of {from,value}
    turns = ex.get("conversations") or []
    return "\n".join(t.get("value", "") for t in turns if isinstance(t, dict))


def _or1(ex):  # OpenR1-Math-220k fallback: problem + first long-CoT generation
    gens = ex.get("generations") or []
    sol = gens[0] if gens else (ex.get("solution") or "")
    return (ex.get("problem", "") + "\n" + (sol if isinstance(sol, str) else "")).strip()


# Simple lanes: (dataset, kwargs, extractor) tried in order until one fills the lane.
SIMPLE_SOURCES = {
    "web": [("HuggingFaceFW/fineweb-edu", {"name": "sample-10BT"}, _txt("text"))],
    "code": [("bigcode/the-stack-smol", {}, _txt("content")),
             ("codeparrot/codeparrot-clean-valid", {}, _txt("content"))],
    "math": [("open-web-math/open-web-math", {}, _txt("text"))],
    "reasoning": [("open-thoughts/OpenThoughts-114k", {}, _conv),
                  ("open-r1/OpenR1-Math-220k", {}, _or1)],
}
# Multilingual lanes: ALL langs consumed (not fallback), each -> its own shard file.
MULTI_SOURCES = {
    "indic_native": [(l, "ai4bharat/sangraha", {"data_dir": f"verified/{l}"}) for l in INDIC_LANGS],
    "indic_translated": [("hin", "ai4bharat/sangraha", {"data_dir": "synthetic/hin_Deva"}),
                         ("ben", "ai4bharat/sangraha", {"data_dir": "synthetic/ben_Beng"}),
                         ("tam", "ai4bharat/sangraha", {"data_dir": "synthetic/tam_Taml"})],
}


def shards_for(lane):
    """(shard_id, txt_filename) list — 1 for simple lanes, 3 for multilingual."""
    if lane in MULTI_SOURCES:
        return [(f"{lane}.{l}", f"{lane}.{l}.txt") for l, _, _ in MULTI_SOURCES[lane]]
    return [(lane, f"{lane}.txt")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- data prep
def _stream_to(path, ds_name, kwargs, extractor, budget):
    from datasets import load_dataset
    ds = load_dataset(ds_name, split="train", streaming=True, **kwargs)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in ds:
            txt = extractor(ex)
            if not txt or len(txt) < 200:
                continue
            f.write(txt); f.write("\n\n")
            n += len(txt) + 2
            if n >= budget:
                break
    return n


def cmd_prep(args):
    DATA.mkdir(parents=True, exist_ok=True)
    for lane in LANES:
        if lane in MULTI_SOURCES:
            per = CHAR_BUDGET[lane] // len(MULTI_SOURCES[lane])
            for lang, ds_name, kwargs in MULTI_SOURCES[lane]:
                out = DATA / f"{lane}.{lang}.txt"
                if out.exists() and out.stat().st_size > 0.9 * per:
                    log(f"{lane}.{lang}: cached ({out.stat().st_size/1e6:.0f} MB), skip"); continue
                log(f"{lane}.{lang}: streaming {ds_name} {kwargs} ...")
                n = _stream_to(out, ds_name, kwargs, _txt("text"), per)
                log(f"{lane}.{lang}: wrote {n/1e6:.0f} MB")
        else:
            out = DATA / f"{lane}.txt"
            if out.exists() and out.stat().st_size > 0.9 * CHAR_BUDGET[lane]:
                log(f"{lane}: cached ({out.stat().st_size/1e6:.0f} MB), skip"); continue
            done = False
            for ds_name, kwargs, extractor in SIMPLE_SOURCES[lane]:
                try:
                    log(f"{lane}: streaming {ds_name} {kwargs} ...")
                    n = _stream_to(out, ds_name, kwargs, extractor, CHAR_BUDGET[lane])
                    log(f"{lane}: wrote {n/1e6:.0f} MB from {ds_name}"); done = True; break
                except Exception as e:  # noqa: BLE001 — fall through to next source
                    log(f"{lane}: {ds_name} failed ({type(e).__name__}: {e}); fallback")
            if not done:
                raise SystemExit(f"FATAL: no source worked for lane {lane}")


# ---------------------------------------------------------------- tokenizer
def cmd_tokenizer(args):
    from tokenizers import ByteLevelBPETokenizer
    sample = DATA / "tok_sample.txt"
    with open(sample, "w", encoding="utf-8") as f:
        for lane in LANES:  # ~8 MB/lane, split across shards so every script is seen
            shards = shards_for(lane)
            per = 8_000_000 // len(shards)
            for _, fn in shards:
                with open(DATA / fn, encoding="utf-8") as src:
                    f.write(src.read(per))
                f.write("\n\n")
    tok = ByteLevelBPETokenizer()
    tok.train(files=[str(sample)], vocab_size=24576, min_frequency=2)
    tok.save(str(DATA / "tokenizer.json"))
    log("tokenizer trained: 24576 byte-level BPE -> data/tokenizer.json")


def cmd_tokenize(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    manifest = {}
    for lane in LANES:
        manifest[lane] = []
        for shard_id, fn in shards_for(lane):
            out = DATA / f"{shard_id}.npy"
            manifest[lane].append(out.name)
            if out.exists():
                log(f"{shard_id}: already tokenized, skip"); continue
            ids = []
            with open(DATA / fn, encoding="utf-8") as f:
                while True:
                    chunk = f.read(8_000_000)
                    if not chunk:
                        break
                    ids.extend(tok.encode(chunk).ids)
            np.save(out, np.array(ids, dtype=np.uint16))
            log(f"{shard_id}: {len(ids)/1e6:.1f}M tokens -> {out.name}")
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"manifest: { {k: len(v) for k,v in manifest.items()} }")


# ---------------------------------------------------------------- model
def build_model(vocab, d=192, n_layer=6, n_head=6, ctx=512):
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


_MANIFEST = None
def manifest():
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = json.loads((DATA / "manifest.json").read_text())
    return _MANIFEST


class LaneSampler:
    """Samples (x, y) windows per a mixture; each lane draws uniformly over its shards
    (so an indic lane weights hin/ben/tam equally)."""

    def __init__(self, mixture, ctx, seed, split="train", lanes=None):
        self.ctx = ctx
        self.rng = np.random.default_rng(seed)
        self.lanes = lanes or LANES
        self.shards = {}  # lane -> list of arrays (train or eval slice)
        for lane in self.lanes:
            arrs = []
            for name in manifest()[lane]:
                a = np.load(DATA / name, mmap_mode="r")
                cut = int(len(a) * 0.95)
                arrs.append(a[:cut] if split == "train" else a[cut:])
            self.shards[lane] = arrs
        self.p = np.array([mixture[l] for l in self.lanes], dtype=np.float64)
        self.p /= self.p.sum()

    def batch(self, bs):
        import torch
        xs = np.empty((bs, self.ctx + 1), dtype=np.int64)
        lane_idx = self.rng.choice(len(self.lanes), size=bs, p=self.p)
        for i, li in enumerate(lane_idx):
            arrs = self.shards[self.lanes[li]]
            a = arrs[self.rng.integers(0, len(arrs))]
            start = int(self.rng.integers(0, len(a) - self.ctx - 2))
            xs[i] = a[start:start + self.ctx + 1]
        t = torch.from_numpy(xs)
        return t[:, :-1], t[:, 1:]


def _lane_loss(model, device, ctx, lane, shard=None, seed=1234, n_batches=20, bs=16):
    import torch
    # eval a whole lane (all shards) or a single shard by masking the manifest
    if shard is not None:
        orig = manifest()[lane]
        manifest()[lane] = [shard]
    s = LaneSampler({lane: 1.0}, ctx, seed=seed, split="eval", lanes=[lane])
    tot = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = s.batch(bs)
            _, loss = model(x.to(device), y.to(device))
            tot += loss.item()
    if shard is not None:
        manifest()[lane] = orig
    return tot / n_batches


def eval_lanes(model, device, ctx):
    model.eval()
    lane_losses = {l: _lane_loss(model, device, ctx, l) for l in LANES}
    lang_diag = {}  # per-language native/translated loss (diagnostic, not in composite)
    for lane in MULTI_SOURCES:
        for name in manifest()[lane]:
            lang_diag[name.replace(".npy", "")] = round(
                _lane_loss(model, device, ctx, lane, shard=name), 4)
    model.train()
    return lane_losses, lang_diag


def cmd_train(args):
    import torch
    RUNS.mkdir(parents=True, exist_ok=True)
    mixture = json.loads(args.mixture) if args.mixture else PRIOR_HA
    assert abs(sum(mixture.values()) - 1.0) < 1e-3, "mixture must sum to 1"
    name = args.name
    result_path = RUNS / f"{name}.json"
    if result_path.exists():
        log(f"{name}: already done, skip"); return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    seed = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % (2**31)
    torch.manual_seed(seed)
    from tokenizers import Tokenizer
    vocab = Tokenizer.from_file(str(DATA / "tokenizer.json")).get_vocab_size()
    ctx, bs = 512, 16
    model = build_model(vocab, ctx=ctx).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    steps = max(1, args.tokens // (bs * ctx))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95))
    warmup = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(warmup, 1), 1.0)
        * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0)))))
    sampler = LaneSampler(mixture, ctx, seed=seed, split="train")
    log(f"{name}: {n_params/1e6:.1f}M params, vocab={vocab}, {steps} steps "
        f"({args.tokens/1e6:.0f}M tok), device={device}")
    t0 = time.time()
    for step in range(steps):
        x, y = sampler.batch(bs)
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0 or step == steps - 1:
            log(f"{name}: step {step}/{steps} loss {loss.item():.3f} "
                f"({(step+1)*bs*ctx/max(time.time()-t0,1e-9)/1e3:.0f}K tok/s)")
    lane_losses, lang_diag = eval_lanes(model, device, ctx)
    row = {"name": name, "mixture": mixture, "lane_losses": lane_losses,
           "lang_diag": lang_diag, "tokens": args.tokens, "params": n_params,
           "seconds": round(time.time() - t0, 1)}
    result_path.write_text(json.dumps(row) + "\n")
    log(f"{name}: DONE lane_losses={ {k: round(v,3) for k,v in lane_losses.items()} }")
    log(f"{name}: lang_diag={lang_diag}")


# ---------------------------------------------------------------- sweep & fit
def sample_mixtures(n, seed=7):
    rng = np.random.default_rng(seed)
    prior = np.array([PRIOR_HA[l] for l in LANES])
    out = []
    while len(out) < n:
        m = rng.dirichlet(prior * 2.5)
        if m.min() < 0.02:  # keep every lane minimally present (floor analogue)
            continue
        out.append({l: round(float(v), 4) for l, v in zip(LANES, m)})
    return out


def cmd_sweep(args):
    # multi-seed priors: 3 differently-named runs each -> distinct seeds via name hash
    runs = [(f"HA_s{i}", PRIOR_HA) for i in range(1, 4)]
    runs += [(f"HB_s{i}", PRIOR_HB) for i in range(1, 4)]
    runs += [(f"mix{i:02d}", m) for i, m in enumerate(sample_mixtures(args.n))]
    shard_i, shard_k = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
    for idx, (name, mix) in enumerate(runs):
        if idx % shard_k != shard_i:
            continue
        cmd_train(argparse.Namespace(name=name, mixture=json.dumps(mix), tokens=args.tokens))


def cmd_fit(args):
    from sklearn.linear_model import Ridge
    allrows = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))
               if p.name != "fit_report.json" and not p.name.startswith("smoke")]
    design = [r for r in allrows if not r["name"].startswith("confirm")]
    confirms = [r for r in allrows if r["name"].startswith("confirm")]
    best_obs = {l: min(r["lane_losses"][l] for r in design) for l in LANES}  # fixed normalizer

    def score(r):
        sc = {l: min(best_obs[l] / r["lane_losses"][l], 1.2) for l in LANES}
        r["C"] = sum(CWEIGHTS[l] * sc[l] for l in LANES)
        r["collapsed"] = [l for l in LANES if sc[l] < COLLAPSE_GUARD]
        return r
    for r in design + confirms:
        score(r)

    def agg(prefix):  # mean +/- range over the 3 seeds of a prior
        rs = [r for r in design if r["name"].startswith(prefix)]
        Cs = [r["C"] for r in rs]
        n_collapse = sum(1 for r in rs if "indic_native" in r["collapsed"])
        return {"n": len(rs), "C_mean": round(np.mean(Cs), 4), "C_min": round(min(Cs), 4),
                "C_max": round(max(Cs), 4), "native_collapse_seeds": f"{n_collapse}/{len(rs)}",
                "native_loss": [round(r["lane_losses"]["indic_native"], 4) for r in rs]}

    non_prior = [r for r in design if not (r["name"].startswith("HA_") or r["name"].startswith("HB_"))]
    non_prior.sort(key=lambda r: -r["C"])

    X = np.array([[math.log(max(r["mixture"][l], 1e-3)) for l in LANES] for r in design])
    models = {}
    for l in LANES:
        y = np.array([r["lane_losses"][l] for r in design])
        models[l] = Ridge(alpha=1e-2).fit(X, y)
    lo = {l: min(r["mixture"][l] for r in design) for l in LANES}
    hi = {l: max(r["mixture"][l] for r in design) for l in LANES}
    rng = np.random.default_rng(0)
    grid = rng.dirichlet(np.ones(len(LANES)), size=400_000)
    grid = grid[grid.min(axis=1) >= 0.02]
    Xg = np.log(np.maximum(grid, 1e-3))
    pred = {l: models[l].predict(Xg) for l in LANES}
    sc = {l: np.minimum(best_obs[l] / pred[l], 1.2) for l in LANES}
    C = sum(CWEIGHTS[l] * sc[l] for l in LANES)
    guard_ok = np.all([sc[l] >= COLLAPSE_GUARD for l in LANES], axis=0)
    in_hull = np.all([(grid[:, j] >= lo[l]) & (grid[:, j] <= hi[l])
                      for j, l in enumerate(LANES)], axis=0)

    def argmax_mix(mask):
        i = int(np.argmax(np.where(mask, C, -1.0)))
        return ({l: round(float(grid[i, j]), 4) for j, l in enumerate(LANES)}, round(float(C[i]), 4))
    pred_hull, C_hull = argmax_mix(guard_ok & in_hull)
    pred_extrap, C_extrap = argmax_mix(guard_ok)

    report = {
        "n_design": len(design), "n_confirm": len(confirms),
        "cweights": CWEIGHTS, "collapse_guard": COLLAPSE_GUARD,
        "HA": agg("HA_"), "HB": agg("HB_"),
        "ranked_non_prior": [{"name": r["name"], "C": round(r["C"], 4),
                              "collapsed": r["collapsed"], "mixture": r["mixture"]}
                             for r in non_prior],
        "confirmations_out_of_sample": [
            {"name": r["name"], "C": round(r["C"], 4), "collapsed": r["collapsed"],
             "mixture": r["mixture"]} for r in confirms],
        "predicted_best_in_hull": {"mixture": pred_hull, "C": C_hull},
        "predicted_best_unconstrained": {"mixture": pred_extrap, "C": C_extrap,
            "note": "extrapolated corner — NOT trusted"},
    }
    (RUNS / "fit_report.json").write_text(json.dumps(report, indent=2))
    log(f"HA {report['HA']}")
    log(f"HB {report['HB']}")
    if non_prior:
        log(f"top non-prior: {non_prior[0]['name']} C={non_prior[0]['C']:.4f} "
            f"collapsed={non_prior[0]['collapsed']}")
    log(f"predicted-best (in hull): C={C_hull} {pred_hull}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep"); sub.add_parser("tokenizer"); sub.add_parser("tokenize")
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--mixture", default=None)
    t.add_argument("--tokens", type=int, default=12_000_000)
    s = sub.add_parser("sweep")
    s.add_argument("--n", type=int, default=18)
    s.add_argument("--tokens", type=int, default=12_000_000)
    s.add_argument("--shard", default=None, help="i/k — this worker runs indices i mod k")
    sub.add_parser("fit")
    args = p.parse_args()
    {"prep": cmd_prep, "tokenizer": cmd_tokenizer, "tokenize": cmd_tokenize,
     "train": cmd_train, "sweep": cmd_sweep, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
