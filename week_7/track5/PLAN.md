# Track 5 — Reversible codec / drop the output head

## Claim
A tiny transformer LM can replace the standard `V×d_model` **softmax output head** with a
**structured byte-grid head** that predicts, for each of `MAXLEN` output byte-positions, a
distribution over 256 byte values (+1 PAD). Decoding = argmax-per-position → bytes → word
(this *inverts* κ's per-position blocks). Consequences demonstrated:

- **(a) Quality parity** — competitive next-token accuracy vs. a softmax head.
- **(b) Open vocabulary** — it emits words *never seen in training*, which a fixed softmax head
  structurally cannot.
- **(c) Flat output params** — head size is independent of vocab; softmax grows linearly → 1M vocab.
- **(d) High exact-decode rate** — predicted grids invert to valid, correct words.

## Why it works (recap)
`word --κ--> R^D (256×MAXLEN) --W_proj--> R^{d_model}`. κ is **losslessly invertible** (argmax per
position block); only `W_proj` loses info. So at the *output*, predict the structured
byte-grid and invert κ, instead of scoring a vocab softmax through a bottleneck.

## Design
- **Tokens = words.** Each word is one position in the transformer sequence.
- **Input embedding (shared by both models, for fairness):** deterministic Kronecker codec
  `κ(bytes)` (D = 256×MAXLEN), then a single trainable `W_proj: D→d_model`. Fixed codec + one matrix,
  vocab-independent on the input side too.
- **Backbone (shared):** tiny GPT — `n_layer=2, d_model=64, n_head=2`, causal self-attention.
- **Heads (the only difference):**
  - `softmax`: `Linear(d_model, V)`; loss = CE over vocab; **cannot** emit out-of-vocab words.
  - `bytegrid`: `Linear(d_model, MAXLEN×257)`; reshape to `[MAXLEN, 257]`; loss = mean over positions
    of CE against the target word's padded byte sequence (class 256 = PAD/end). Decode = argmax per
    position, stop at PAD → bytes → word. Size independent of V.

## Task (makes open-vocab crisp): prefix-copy language
Each sequence: `<bos> W <sep> W <eos>` (5 word-positions). Standard next-token prediction; the
decisive prediction is the token right after `<sep>`, which must reproduce `W`. To succeed the model
must learn a **copy/induction circuit** (carry W's identity from position 1 to the output). Because
the input embedding is byte-structured, copying bytes is natural.
- **Train vocab:** fixed pool of `V` random lowercase words (len 3–6). Softmax vocab = specials + these.
- **In-vocab eval:** sequences using training-pool words (held-out sequences) → parity check.
- **Open-vocab (OOV) eval:** sequences using a **disjoint** pool of novel words never seen in
  training. Softmax → 0 by construction (word not in vocab); bytegrid → high iff it learned to copy
  (generalization, not memorization).

## Metrics
1. In-vocab copy accuracy (both) — quality parity.
2. OOV copy accuracy (both) — the headline (softmax ≡ 0).
3. Full-sequence next-token accuracy / perplexity (both).
4. Exact-decode rate (bytegrid) — fraction of decoded words that exactly match target.
5. Output-head param count vs vocab size {10…1e6} — softmax linear vs bytegrid flat (analytic plot).
6. Codec round-trip check: `decode(κ(w)) == w` for all words (should be exact by construction).

## Files
- `kron.py` — Kronecker encode/decode, tokenizer, data generation.
- `model.py` — tiny transformer + swappable head.
- `run.py` — train softmax & bytegrid, evaluate, save metrics JSON + plots.
- `plots/` — PNGs. `README.md` — claim, method, numbers, proof.

## Success criterion
bytegrid: OOV copy accuracy ≫ softmax (≈0), in-vocab parity, high exact-decode, flat param curve.
