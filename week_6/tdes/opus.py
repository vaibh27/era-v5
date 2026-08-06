"""OPUS-style data selection — a faithful nano miniature of arXiv 2602.05400.

OPUS scores a candidate by the *projected utility* of its update: does training on it
move parameters in a direction that also improves a benchmark-aligned proxy? We score in
the LM-head gradient block (dWhead = hf^T @ dlogits), which needs only a forward pass, and
take the cosine against a proxy direction. Decisions: accept / reject / defer / floor-override.

BENCH-PROXY: the proxy direction is built from *train-split* samples of the high-value
lanes (code + math + indic), i.e. retrieved from the corpus — never from eval — so the
evaluation firewall is untouched. The protected floor overrides OPUS: a lane in deficit
for the current window is force-accepted regardless of utility (the floor wins).
"""
import numpy as np

from . import pack
from .model import _ce_grad

PROXY_LANES = ("code", "math", "indic")


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _head_grad(model, spans, seq_len):
    """dWhead (LM-head gradient) flattened — the update-space signal, forward-only."""
    packs = pack.pack_spans(spans, seq_len)
    if not packs:
        return np.zeros(model.d * model.V)
    ii = np.stack([p["input_ids"] for p in packs])
    pos = np.stack([p["pos_ids"] for p in packs])
    seg = np.stack([p["seg_ids"] for p in packs])
    lab = np.stack([p["labels"] for p in packs])
    _, cache, _ = model.forward(ii, pos, seg, lab)
    dlogits = _ce_grad(cache["probs"], cache["labels"], cache["valid"], cache["n"])
    dWhead = np.einsum("btd,btv->dv", cache["hf"], dlogits)
    return dWhead.ravel()


def build_proxy_spans(eligible, max_docs=12):
    """Retrieve a small benchmark-aligned pool from train-split high-value lanes."""
    from . import shard
    spans = []
    for m in sorted(eligible, key=lambda x: x["shard_id"]):
        if m["lane"] not in PROXY_LANES:
            continue
        toks = shard.load_tokens(m["shard_id"])
        for _, s, e in shard.iter_docs(m, toks):
            spans.append(toks[s:e])
            if len(spans) >= max_docs:
                return spans
    return spans


class OpusSelector:
    # thresholds calibrated to the nano head-gradient cosine scale (utilities ~ ±0.01):
    # accept clearly-aligned, reject clearly-anti-aligned, defer the borderline middle.
    def __init__(self, model, eligible, seq_len=64, accept=0.002, reject=-0.002):
        self.model = model
        self.seq_len = seq_len
        self.accept = accept
        self.reject = reject
        self.proxy_spans = build_proxy_spans(eligible)
        self.u_ref = None
        self.deficits = {}
        self.step = -1
        self.records = []

    def set_context(self, step, deficits):
        """Called before each batch. Recomputes the proxy direction from the CURRENT model
        state (a pure function of model params) so decisions are reproducible on resume."""
        self.step = step
        self.deficits = dict(deficits)
        self.u_ref = _unit(_head_grad(self.model, self.proxy_spans, self.seq_len))

    def score_batch(self, toks_list):
        """Score many candidates in ONE forward pass (per-sample LM-head gradient cosine
        vs the proxy direction). Batching the vocab-softmax forward is the key speedup."""
        if not toks_list:
            return []
        packs = [pack.pack_spans([t], self.seq_len)[0] for t in toks_list]
        ii = np.stack([p["input_ids"] for p in packs])
        pos = np.stack([p["pos_ids"] for p in packs])
        seg = np.stack([p["seg_ids"] for p in packs])
        lab = np.stack([p["labels"] for p in packs])
        _, cache, _ = self.model.forward(ii, pos, seg, lab)
        dlog = _ce_grad(cache["probs"], cache["labels"], cache["valid"], cache["n"])
        hf = cache["hf"]
        utils = []
        for b in range(len(toks_list)):
            g = (hf[b].T @ dlog[b]).ravel()  # per-sample dWhead
            utils.append(float(_unit(g) @ self.u_ref))
        return utils

    def decide(self, cand, utility):
        util = utility
        lane = cand["lane"]
        if lane in self.deficits:
            decision, keep = "floor_override", True
        elif util >= self.accept:
            decision, keep = "accept", True
        elif util <= self.reject:
            decision, keep = "reject", False
        else:
            decision, keep = "defer", False
        self.records.append({**cand, "utility": round(util, 5),
                             "decision": decision, "step": self.step})
        return keep

    def decision_counts(self):
        from collections import Counter
        return dict(Counter(r["decision"] for r in self.records))
