"""NanoLM — a small but real decoder-only transformer, forward + backward in pure numpy.

Single-head attention, pre-norm, GELU MLP, untied LM head. It consumes the packing
outputs directly: positional embeddings are indexed by the RESET pos_ids, and attention
uses the block-diagonal causal mask from seg_ids — so packing correctness actually
matters to the loss. Gradients are hand-derived and validated by a finite-difference
gradient check (see tests), which is what makes the learning ledger and the OPUS utility
score (both gradient-based) trustworthy rather than decorative.

Everything is deterministic given the init seed and inputs.
"""
import numpy as np

IGNORE = -100
EPS = 1e-5


def gelu(x):
    # tanh approximation (stable, known derivative)
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def dgelu(x):
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    dinner = c * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * dinner


def _ln_forward(x, g, b):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + EPS)
    xhat = xc * inv
    return g * xhat + b, (xhat, inv, g)


def _ln_backward(dout, cache):
    xhat, inv, g = cache
    D = xhat.shape[-1]
    dg = (dout * xhat).reshape(-1, D).sum(0)
    db = dout.reshape(-1, D).sum(0)
    dxhat = dout * g
    dx = inv / D * (D * dxhat - dxhat.sum(-1, keepdims=True)
                    - xhat * (dxhat * xhat).sum(-1, keepdims=True))
    return dx, dg, db


def _softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


class NanoLM:
    def __init__(self, vocab, d_model=64, n_layer=2, max_pos=256, seed=0):
        self.V, self.d, self.L, self.P = vocab, d_model, n_layer, max_pos
        self.dff = 4 * d_model
        rng = np.random.default_rng(seed)
        s = 0.02
        p = {}
        p["wte"] = rng.normal(0, s, (vocab, d_model))
        p["wpe"] = rng.normal(0, s, (max_pos, d_model))
        for l in range(n_layer):
            for nm, shape in [("Wq", (d_model, d_model)), ("Wk", (d_model, d_model)),
                              ("Wv", (d_model, d_model)), ("Wo", (d_model, d_model)),
                              ("Wf1", (d_model, self.dff)), ("Wf2", (self.dff, d_model))]:
                p[f"l{l}.{nm}"] = rng.normal(0, s, shape)
            p[f"l{l}.bf1"] = np.zeros(self.dff)
            p[f"l{l}.bf2"] = np.zeros(d_model)
            p[f"l{l}.ln1g"] = np.ones(d_model); p[f"l{l}.ln1b"] = np.zeros(d_model)
            p[f"l{l}.ln2g"] = np.ones(d_model); p[f"l{l}.ln2b"] = np.zeros(d_model)
        p["lnfg"] = np.ones(d_model); p["lnfb"] = np.zeros(d_model)
        p["Whead"] = rng.normal(0, s, (d_model, vocab))
        p["bhead"] = np.zeros(vocab)
        self.p = p

    # ---- forward: returns (loss, cache, per_pos_loss) ----
    def forward(self, input_ids, pos_ids, seg_ids, labels):
        p, d = self.p, self.d
        B, T = input_ids.shape
        # block-diagonal causal mask [B,T,T]
        causal = np.tril(np.ones((T, T), bool))
        same = (seg_ids[:, :, None] == seg_ids[:, None, :]) & (seg_ids[:, :, None] >= 0)
        mask = causal[None] & same

        h = p["wte"][input_ids] + p["wpe"][pos_ids]  # [B,T,d]
        caches = []
        for l in range(self.L):
            ln1, ln1c = _ln_forward(h, p[f"l{l}.ln1g"], p[f"l{l}.ln1b"])
            q = ln1 @ p[f"l{l}.Wq"]; k = ln1 @ p[f"l{l}.Wk"]; v = ln1 @ p[f"l{l}.Wv"]
            scores = q @ k.transpose(0, 2, 1) / np.sqrt(d)  # [B,T,T]
            scores = np.where(mask, scores, -1e9)
            attn = _softmax(scores, -1)
            ctx = attn @ v                                  # [B,T,d]
            ao = ctx @ p[f"l{l}.Wo"]
            h = h + ao
            ln2, ln2c = _ln_forward(h, p[f"l{l}.ln2g"], p[f"l{l}.ln2b"])
            f1 = ln2 @ p[f"l{l}.Wf1"] + p[f"l{l}.bf1"]
            a1 = gelu(f1)
            f2 = a1 @ p[f"l{l}.Wf2"] + p[f"l{l}.bf2"]
            h = h + f2
            caches.append(dict(ln1=ln1, ln1_lncache=ln1c, q=q, k=k, v=v, attn=attn,
                               ctx=ctx, ln2v=ln2, ln2=ln2c, f1=f1, a1=a1))
        hf, lnf_cache = _ln_forward(h, p["lnfg"], p["lnfb"])
        logits = hf @ p["Whead"] + p["bhead"]               # [B,T,V]

        probs = _softmax(logits, -1)
        valid = labels != IGNORE
        safe = np.where(valid, labels, 0)
        logp = np.log(np.take_along_axis(probs, safe[..., None], -1)[..., 0] + 1e-12)
        per_pos = np.where(valid, -logp, 0.0)
        n = int(valid.sum())
        loss = per_pos.sum() / max(n, 1)

        cache = dict(input_ids=input_ids, pos_ids=pos_ids, mask=mask, caches=caches,
                     hf=hf, lnf_cache=lnf_cache, probs=probs, valid=valid, n=n,
                     labels=labels)
        return loss, cache, per_pos

    # ---- backward: returns grads dict (same keys as self.p) ----
    def backward(self, cache):
        p, d = self.p, self.d
        g = {k: np.zeros_like(v) for k, v in p.items()}
        probs, valid, n = cache["probs"], cache["valid"], cache["n"]
        B, T, V = probs.shape

        dlogits = _ce_grad(probs, cache["labels"], valid, n)

        dhf = dlogits @ p["Whead"].T
        g["Whead"] = np.einsum("btd,btv->dv", cache["hf"], dlogits)
        g["bhead"] = dlogits.reshape(-1, V).sum(0)

        dh, g["lnfg"], g["lnfb"] = _ln_backward(dhf, cache["lnf_cache"])

        for l in reversed(range(self.L)):
            c = cache["caches"][l]
            # MLP block
            df2 = dh
            g[f"l{l}.Wf2"] = np.einsum("btf,btd->fd", c["a1"], df2)
            g[f"l{l}.bf2"] = df2.reshape(-1, d).sum(0)
            da1 = df2 @ p[f"l{l}.Wf2"].T
            df1 = da1 * dgelu(c["f1"])
            g[f"l{l}.Wf1"] = np.einsum("btd,btf->df", c["ln2v"], df1)
            g[f"l{l}.bf1"] = df1.reshape(-1, self.dff).sum(0)
            dln2 = df1 @ p[f"l{l}.Wf1"].T
            dh2, g[f"l{l}.ln2g"], g[f"l{l}.ln2b"] = _ln_backward(dln2, c["ln2"])
            dh = dh + dh2  # residual

            # attention block
            dao = dh
            g[f"l{l}.Wo"] = np.einsum("btd,bte->de", c["ctx"], dao)
            dctx = dao @ p[f"l{l}.Wo"].T
            dattn = dctx @ c["v"].transpose(0, 2, 1)
            dv = c["attn"].transpose(0, 2, 1) @ dctx
            # softmax backward
            dscores = c["attn"] * (dattn - (dattn * c["attn"]).sum(-1, keepdims=True))
            dscores = dscores / np.sqrt(d)
            dq = dscores @ c["k"]
            dk = dscores.transpose(0, 2, 1) @ c["q"]
            g[f"l{l}.Wq"] = np.einsum("btd,bte->de", c["ln1"], dq)
            g[f"l{l}.Wk"] = np.einsum("btd,bte->de", c["ln1"], dk)
            g[f"l{l}.Wv"] = np.einsum("btd,bte->de", c["ln1"], dv)
            dln1 = dq @ p[f"l{l}.Wq"].T + dk @ p[f"l{l}.Wk"].T + dv @ p[f"l{l}.Wv"].T
            dh1, g[f"l{l}.ln1g"], g[f"l{l}.ln1b"] = _ln_backward(dln1, c["ln1_lncache"])
            dh = dh + dh1  # residual

        # embeddings
        np.add.at(g["wte"], cache["input_ids"], dh)
        np.add.at(g["wpe"], cache["pos_ids"], dh)
        return g


def _ce_grad(probs, labels, valid, n):
    B, T, V = probs.shape
    d = probs.copy()
    safe = np.where(valid, labels, 0)
    oh = np.zeros_like(d)
    np.put_along_axis(oh, safe[..., None], 1.0, -1)
    d = d - oh
    d = d * valid[..., None]
    return d / max(n, 1)
