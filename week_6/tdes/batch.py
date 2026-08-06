"""Deterministic batch construction — the spine of resume, replay, and audit.

A batch is a pure function of (seed, step, lane_weights, eligible shards). Given those,
`build_batch` draws document spans, packs them, and returns the packed tensors plus:

  - batch_hash : sha256 over the packed input_ids (bit-exact batch identity)
  - sources    : provenance [{shard_id, lane, start, end}] — every token traced to a shard
  - util, n_loss_tokens : packing efficiency signals

Determinism rules honored here (so replay reproduces batches bit-for-bit):
  * RNG is counter-based on (seed, step) — never wall-clock or dict iteration order
  * lanes and shards are visited in sorted order
Only train-split shards are ever passed in (the firewall), so eval never enters a batch.
"""
import hashlib

import numpy as np

from . import pack, shard


def _rng(seed, step):
    return np.random.default_rng([int(seed), int(step)])


def _doc_pools(eligible):
    """lane -> list of (shard_id, lane, start, end, tokens) in deterministic order."""
    pools = {}
    for m in sorted(eligible, key=lambda x: x["shard_id"]):
        toks = shard.load_tokens(m["shard_id"])
        for _, s, e in shard.iter_docs(m, toks):
            pools.setdefault(m["lane"], []).append(
                (m["shard_id"], m["lane"], s, e, toks[s:e]))
    return pools


def batch_hash(packs):
    h = hashlib.sha256()
    for p in packs:
        h.update(p["input_ids"].astype(np.int64).tobytes())
    return h.hexdigest()


def build_batch(step, seed, eligible, lane_weights, seq_len=64, n_seqs=4,
                selector=None):
    """Build batch #step. `selector` is an optional OPUS selector; when None all drawn
    candidates are kept. Returns a batch dict.

    Two passes: (1) draw candidate chunks by token-aware lane targeting — a pure function
    of (seed, step); (2) score all candidates in one OPUS forward and apply decisions.
    """
    rng = _rng(seed, step)
    pools = _doc_pools(eligible)
    lanes = sorted(l for l in lane_weights if pools.get(l))
    w = np.array([lane_weights[l] for l in lanes], dtype=float)
    w = w / w.sum()

    target = n_seqs * seq_len
    # token-aware targeting: floors/weights are on TOKEN share, and doc lengths vary by
    # lane, so target weight*capacity tokens per lane and draw from the most-under lane.
    lane_target = {l: w[i] * target for i, l in enumerate(lanes)}
    drawn_tok = {l: 0 for l in lanes}
    cands = []
    total = 0
    guard = 0
    while total < target and guard < target * 8:
        guard += 1
        lane = max(lanes, key=lambda l: lane_target[l] - drawn_tok[l])
        pool = pools[lane]
        sid, ln, s, e, toks = pool[rng.integers(len(pool))]
        cand = {"shard_id": sid, "lane": ln, "start": int(s), "end": int(e),
                "n_tokens": int(len(toks))}
        cands.append((cand, toks))
        drawn_tok[lane] += len(toks)
        total += len(toks)

    # pass 2: batched OPUS scoring + decisions
    utils = selector.score_batch([t for _, t in cands]) if selector is not None else None
    spans, sources, decisions = [], [], []
    for i, (cand, toks) in enumerate(cands):
        keep = True if selector is None else selector.decide(cand, utils[i])
        decisions.append({**cand, "kept": bool(keep)})
        if keep:
            spans.append(toks)
            sources.append(cand)

    packs = pack.pack_spans(spans, seq_len, span_lanes=[s["lane"] for s in sources])[:n_seqs]
    bh = batch_hash(packs)
    n_loss = sum(int(p["loss_mask"].sum()) for p in packs)
    return {
        "step": step,
        "batch_id": f"b{step:06d}",
        "batch_hash": bh,
        "packs": packs,
        "sources": sources,
        "decisions": decisions,
        "n_seqs": len(packs),
        "seq_len": seq_len,
        "util": pack.utilization(packs, seq_len),
        "n_loss_tokens": n_loss,
    }
