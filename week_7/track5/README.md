# Track 5 — Reversible codec: drop the output head

**Problem (assignment #5):** Kronecker is forward-deterministic (same word → same embedding). Make
it *reversible* (embedding → word), and you can throw away the final softmax head — enabling a **1M+
vocabulary at no extra cost**.

**What was tried here:** a tiny transformer LM replacing the standard `V×d_model` **softmax head**
with a **byte-grid head** that predicts a per-position byte distribution and decodes by
argmax-per-position (inverting κ). This head is **vocab-free** and **open-vocabulary**, with quality
on par with softmax.

## Why it works
`word --κ--> R^D (256×MAXLEN) --W_proj--> R^{d_model}`. The Kronecker codec κ is **losslessly
invertible** — each position occupies its own block, so `argmax` per block reads the byte back. Only
`W_proj` (→ d_model) loses information. So at the **output**, predict the structured byte-grid and
invert κ, instead of scoring every word in a vocabulary through a bottleneck. The output head becomes
`Linear(d_model, MAXLEN×257)` — size **independent of vocabulary**.

## Experiment (prefix-copy language)
Sequences `<bos> W <sep> W <eos>`; standard next-token training. The decisive prediction is the token
after `<sep>`, which must reproduce `W` — so the model must learn a **copy/induction circuit**.
- **Train:** fixed pool of 2000 random words. Softmax vocab = specials + these 2000.
- **In-vocab eval:** held-out sequences over the training words (parity check).
- **Open-vocab eval:** a **disjoint** pool of 500 novel words *never seen in training*. A fixed
  softmax head cannot emit these by construction; the byte-grid head can if it learned to copy.

Shared for fairness: deterministic Kronecker **input** embedding + one trainable `W_proj`, and an
identical tiny GPT backbone (`d_model=128, n_layer=3, n_head=4`). The **only** difference is the head.
Run: `python3 run.py` (~80s CPU; writes `results.json` and `plots/`).

## Results

| Metric | softmax head | bytegrid head |
|---|---|---|
| Codec round-trip `decode(κ(w))==w` | — | **2500/2500 exact** |
| In-vocab copy accuracy | 1.000 | 1.000 |
| **Open-vocab (novel words) copy accuracy** | **0.000** | **0.886** |
| Full next-token accuracy (in-vocab) | 0.750 | 0.750 |
| Valid-decode rate (outputs are well-formed words, not exact-match) | n/a | 1.000 |
| Output-head params @ V=2004 | 258,516 | 265,224 |
| **Output-head params @ V=1,000,000** | **129,000,000** | **265,224 (flat)** |

Byte-grid decoding of novel words it was never trained on:
```
target=mwpnk  -> decoded=mwpnk  [ok]
target=icvti  -> decoded=icvti  [ok]
target=krbbz  -> decoded=krbbz  [ok]
```

Running `run.py` writes two figures to `plots/`: `copy_accuracy.png` (copy accuracy, in-vocab vs
open-vocab, both heads) and `head_params_vs_vocab.png` (output-head params vs vocab size).

## Reading of the result
- **(a) Parity:** in-vocab copy and next-token accuracy are identical to softmax.
- **(b) Open vocabulary:** softmax scores 0 on unseen words *by construction*; the byte-grid head
  reaches 0.886 — it emits correctly-spelled words it never saw, proving reversible decoding
  generalizes beyond any vocabulary.
- **(c) Flat cost:** the byte-grid head is constant in `V`; softmax is `d_model·V` → 129M params at
  1M vocab vs. 0.27M. This is the "1M vocab for free" the problem asks for.
- **(d) Exactness:** 100% of decoded outputs are valid words, and κ round-trips exactly.

## Honest limitations
- **32-byte (here 8) cap:** exact reversibility holds only within the position cap; longer words need
  Track 3 / Track 4 (unbounded length) for true open vocab. Cap set to `MAXLEN=8` for this small demo.
- **Independent-per-position decode:** the byte-grid head assumes byte independence given the hidden
  state; the residual ~11% OOV errors are single-byte slips, not invalid strings (valid-decode=1.0).
- **Copy task** is a controlled probe that isolates open-vocab decoding; it is not a full LM. It does
  cleanly separate "can emit unseen words" (byte-grid) from "cannot" (softmax).

## Files
`PLAN.md` · `kron.py` (codec + data) · `model.py` (GPT + swappable head) · `run.py` (train/eval/plot)
· `results.json` · `plots/`.
