"""Train softmax vs bytegrid heads on the prefix-copy task; evaluate and plot.

Proves: (a) quality parity, (b) open-vocabulary via bytegrid (softmax=0), (c) flat output params,
(d) high valid-decode rate (outputs are well-formed words), and the codec round-trip identity.
"""
import json, time, os
import numpy as np
import torch

from kron import (MAXLEN, D, N_CLASSES, kappa, kappa_decode, token_to_bytes,
                  token_to_grid_target, grid_logits_to_token,
                  make_word_pools, Vocab, make_sequences)
from model import TinyGPT

torch.manual_seed(0)
np.random.seed(0)
OUT = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(OUT, "plots"); os.makedirs(PLOTS, exist_ok=True)

# ---------------- config ----------------
N_TRAIN_VOCAB = 2000
N_OOV = 500
N_TRAIN_SEQ = 16000
N_EVAL_SEQ = 1000
D_MODEL, N_LAYER, N_HEAD = 128, 3, 4
STEPS = 2000
BATCH = 128
LR = 2e-3

# ---------------- data ----------------
train_words, oov_words = make_word_pools(N_TRAIN_VOCAB, N_OOV, seed=1)
vocab = Vocab(train_words)
SEQ_LEN = 5           # <bos> W <sep> W <eos>
BLOCK = SEQ_LEN - 1   # inputs 0..3 predict targets 1..4

# cache kappa feats per token string
_kappa_cache = {}
def kf(tok):
    if tok not in _kappa_cache:
        _kappa_cache[tok] = kappa(tok)
    return _kappa_cache[tok]

def batch_tensors(seqs):
    B = len(seqs)
    feats = np.zeros((B, BLOCK, D), dtype=np.float32)
    sm = np.zeros((B, BLOCK), dtype=np.int64)
    grid = np.zeros((B, BLOCK, MAXLEN), dtype=np.int64)
    for i, s in enumerate(seqs):
        for t in range(BLOCK):
            feats[i, t] = kf(s[t])            # input token t
            tgt = s[t + 1]                     # next token
            sm[i, t] = vocab.index(tgt)
            grid[i, t] = token_to_grid_target(tgt)
    return (torch.from_numpy(feats), torch.from_numpy(sm), torch.from_numpy(grid))

train_seqs = make_sequences(train_words, N_TRAIN_SEQ, seed=2)
eval_iv = make_sequences(train_words, N_EVAL_SEQ, seed=3)   # in-vocab
eval_oov = make_sequences(oov_words, N_EVAL_SEQ, seed=4)    # open-vocab (novel words)

train_feats, train_sm, train_grid = batch_tensors(train_seqs)

# The decisive prediction: position index of the token right after <sep>.
# seq = [<bos>, W, <sep>, W, <eos>]; inputs are seq[0:4]; the input at t=2 is <sep>,
# and its target (sm/grid at t=2) is W. So copy-position = 2.
COPY_T = 2

def train(head):
    model = TinyGPT(head, len(vocab), BLOCK, D_MODEL, N_LAYER, N_HEAD)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n = train_feats.size(0)
    t0 = time.time()
    for step in range(STEPS):
        idx = torch.randint(0, n, (BATCH,))
        loss = model.loss(train_feats[idx], train_sm[idx], train_grid[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0 or step == STEPS - 1:
            print(f"  [{head}] step {step:4d}  loss {loss.item():.4f}")
    print(f"  [{head}] trained in {time.time()-t0:.1f}s")
    return model

@torch.no_grad()
def evaluate(model, seqs):
    """Return copy-accuracy (token after <sep>), full next-token accuracy, valid-decode rate.

    valid_decode = fraction of bytegrid outputs that decode to a well-formed word (parseable,
    not "<invalid>"); this is a validity check, NOT exact-match against the target (that is
    copy_acc). All valid words decoded correctly here still leaves the ~11% OOV copy slips.
    """
    feats, sm, grid = batch_tensors(seqs)
    out = model.forward(feats)
    copy_correct = tot_correct = tot = valid = valid_tot = 0
    B = len(seqs)
    if model.head_type == "softmax":
        pred = out.argmax(-1)  # [B,BLOCK]
        for i in range(B):
            for t in range(BLOCK):
                gold_tok = seqs[i][t + 1]
                pred_tok = vocab.itos[pred[i, t].item()]
                ok = (pred_tok == gold_tok)
                tot += 1; tot_correct += ok
                if t == COPY_T:
                    copy_correct += ok
    else:
        pred_grid = out  # [B,BLOCK,MAXLEN,257]
        for i in range(B):
            for t in range(BLOCK):
                gold_tok = seqs[i][t + 1]
                dec = grid_logits_to_token(pred_grid[i, t].numpy())
                ok = (dec == gold_tok)
                tot += 1; tot_correct += ok
                valid_tot += 1; valid += (dec != "<invalid>")
                if t == COPY_T:
                    copy_correct += ok
    return dict(copy_acc=copy_correct / B,
                next_acc=tot_correct / tot,
                valid_decode=(valid / valid_tot) if valid_tot else None)

# ---------------- codec round-trip check ----------------
def roundtrip_check(words):
    bad = 0
    for w in words:
        dec = kappa_decode(kappa(w))
        if dec != token_to_bytes(w):
            bad += 1
    return len(words), bad

print("== codec round-trip (decode(kappa(w))==w) ==")
n_rt, bad_rt = roundtrip_check(train_words + oov_words)
print(f"  {n_rt-bad_rt}/{n_rt} exact  (bad={bad_rt})")

print("\n== training softmax head ==")
m_sm = train("softmax")
print("== training bytegrid head ==")
m_bg = train("bytegrid")

results = {"config": dict(n_train_vocab=N_TRAIN_VOCAB, n_oov=N_OOV, steps=STEPS,
                          d_model=D_MODEL, n_layer=N_LAYER, maxlen=MAXLEN, vocab_size=len(vocab)),
           "roundtrip": dict(n=n_rt, bad=bad_rt),
           "head_params": dict(softmax=m_sm.head_param_count(),
                               bytegrid=m_bg.head_param_count())}

for name, model in [("softmax", m_sm), ("bytegrid", m_bg)]:
    results[name] = {"in_vocab": evaluate(model, eval_iv),
                     "open_vocab": evaluate(model, eval_oov)}

# qualitative examples: bytegrid decoding novel (OOV) words
@torch.no_grad()
def examples(model, seqs, k=12):
    feats, _, _ = batch_tensors(seqs[:k])
    out = model.forward(feats)
    rows = []
    for i in range(k):
        gold = seqs[i][COPY_T + 1]
        dec = grid_logits_to_token(out[i, COPY_T].numpy())
        rows.append((gold, dec, "ok" if dec == gold else "X"))
    return rows

results["examples_oov_bytegrid"] = examples(m_bg, eval_oov)
print("\n== bytegrid OOV copy examples (novel words never seen in training) ==")
for gold, dec, ok in results["examples_oov_bytegrid"]:
    print(f"  target={gold:8s} -> decoded={dec:8s} [{ok}]")

print("\n== RESULTS ==")
print(json.dumps({k: v for k, v in results.items() if k != "examples_oov_bytegrid"}, indent=2))
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ---------------- plots ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (1) copy accuracy: in-vocab vs open-vocab, both heads
fig, ax = plt.subplots(figsize=(6, 4))
labels = ["in-vocab", "open-vocab (OOV)"]
x = np.arange(2); w = 0.35
sm_vals = [results["softmax"]["in_vocab"]["copy_acc"], results["softmax"]["open_vocab"]["copy_acc"]]
bg_vals = [results["bytegrid"]["in_vocab"]["copy_acc"], results["bytegrid"]["open_vocab"]["copy_acc"]]
ax.bar(x - w/2, sm_vals, w, label="softmax head", color="#d1495b")
ax.bar(x + w/2, bg_vals, w, label="bytegrid head", color="#2e86ab")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("copy accuracy"); ax.set_ylim(0, 1.05)
ax.set_title("Copy accuracy: softmax cannot emit unseen words")
for xi, v in zip(x - w/2, sm_vals): ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
for xi, v in zip(x + w/2, bg_vals): ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "copy_accuracy.png"), dpi=130)

# (2) output-head params vs vocab size
Vs = np.array([10, 100, 1000, 10_000, 100_000, 1_000_000])
sm_params = D_MODEL * Vs + Vs           # weight + bias
bg_params = m_bg.head_param_count()     # constant
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(Vs, sm_params, "o-", color="#d1495b", label="softmax head (d_model·V)")
ax.axhline(bg_params, ls="--", color="#2e86ab", label=f"bytegrid head (flat = {bg_params:,})")
ax.set_xlabel("vocabulary size V"); ax.set_ylabel("output-head parameters")
ax.set_title("Output head cost: softmax grows with V, bytegrid is flat")
ax.legend(); ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "head_params_vs_vocab.png"), dpi=130)

print(f"\nsaved plots to {PLOTS}")
print(f"head params: softmax(V={len(vocab)})={m_sm.head_param_count():,}  "
      f"bytegrid={m_bg.head_param_count():,}  "
      f"softmax(V=1e6)={D_MODEL*1_000_000+1_000_000:,}")
