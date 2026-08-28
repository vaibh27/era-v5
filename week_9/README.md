# Week 9 — Tiny GPT: Loss Harness Correctness

Notebook: [`tiny_gpt_submission.ipynb`](./tiny_gpt_submission.ipynb) (run top-to-bottom, includes all cell outputs / training logs)

Model: 4-layer, 4-head, 256-dim decoder-only transformer trained from scratch on `roneneldan/TinyStories`, GPT-2 BPE tokenizer (`vocab_size = 50257`), `seq_len = 512`.

All numbers below are taken directly from the notebook's actual cell outputs (not paraphrased/rounded from memory), so they reproduce from the file as-is.

## Part 1 — the seven numbers

**1. Shapes** (batch=4, seq_len=512, n_embd=256, vocab_size=50257)

| tensor | shape | meaning |
|---|---|---|
| `tokens` | (4, 512) | (batch, seq_len) input ids |
| `targets` | (4, 512) | (batch, seq_len) next-token ids, shifted by 1 |
| `hidden` | (4, 512, 256) | (batch, seq_len, n_embd) per-token representation |
| `logits` | (4, 512, 50257) | (batch, seq_len, vocab_size) unnormalized scores |
| `logits_flat` | (2048, 50257) | flattened (batch*seq_len, vocab_size) for cross_entropy |
| `targets_flat` | (2048,) | flattened target ids, row-aligned with `logits_flat` |

**2. Shift verified on strings, not ids** — e.g. input `'Once'` → target `' upon'`, `' upon'` → `' a'`, `' a'` → `' time'`, confirmed token-by-token by decoding both sides (see notebook cell "step 2").

**3. Padding mask — contributing token count changes**
Token positions before masking: **4096** → after masking: **1316** (8 sequences × 512 = 4096 total positions; only 1316 are real, non-pad targets).

**4. Packed two documents, boundary masked**
- Before training: unmasked loss `10.8892` vs masked (boundary excluded) `10.8890` — gap ≈ 0.0002, noise, since the model is freshly initialized and has no notion yet that the doc1→doc2 jump is nonsensical.
- After training (same two documents, same boundary): unmasked `7.6878` vs masked `7.6756` — gap `+0.0122`, ~60x bigger. The boundary transition itself scores `13.9226`, well above the trained average, confirming the model has learned local coherence well enough to be "surprised" by the meaningless cross-document jump, and masking it out removes that artifact from the loss.

**5. Perplexity gate (untrained model)**
Loss `10.8859` vs expected `ln(vocab_size) = 10.8249` (diff `0.0610`) → perplexity `53,420.3` vs `vocab_size = 50,257`. Close enough to pass.
Note: this did **not** pass on the first attempt — default PyTorch init produced too much logit spread. Root-caused and fixed with an explicit `N(0, 0.02)` init on `Linear`/`Embedding` weights (standard nanoGPT choice), which brought the gate in line.

**6. Tied vs untied output head params**
Untied: **29,019,136** params. Tied: **16,153,344** params. Difference: **12,865,792** = `n_embd * vocab_size` (256 × 50257), exactly one full output-projection matrix — the entire difference is that one matrix.

**7. Peak memory — ordinary vs hand-written chunked cross-entropy**
Ordinary (materializes full `(batch*seq_len, vocab_size)` logits): **1646.82 MB**.
Chunked (own implementation, processes 64 rows at a time): **12.93 MB**.
Ratio: **127.35x** — matches the predicted `(16*512)/64 = 128`.

## Part 2 — second head predicting t+2

Added a second output head `head_2` reusing the same shared transformer trunk. `head_2` consumes `hidden[:, :-1, :]` (position `k`'s hidden state, context through token `k`) against targets `raw[:, 2:]` (token at `k+2`), so it predicts two tokens ahead of the same context used by the first head. Both losses are summed and backpropagated in a single `.backward()` through the shared trunk.

After 2000 training steps (`train_seq_len=128`, `batch_size=4`):

| | loss_1 (predict t+1) | loss_2 (predict t+2) | sum |
|---|---|---|---|
| final step | 5.6490 | 5.7224 | 11.3715 |
| mean, last 10 steps | 5.6689 | 5.7183 | — |

`loss_2` sits consistently above `loss_1` throughout training. This is expected and is not a capacity gap: predicting `t+1` from context through `t` is the easiest next-token case, while predicting `t+2` implicitly requires marginalizing over the unseen `t+1`, so more continuations are plausible and the true target distribution has strictly higher entropy. That raises the floor loss a perfect model could achieve for `t+2`, independent of how well the model is trained — both heads share the same trunk and get the same number of steps, so the gap is attributable to task difficulty, not an unfair training head-start.

## Notes / known limitations
- No random seed is set, so exact numbers vary run to run (same order of magnitude, same conclusions each time).
- Packed-document experiment (step 4) masks only the boundary *target*; there's no separator token or attention-side isolation, so `doc2` can still attend into `doc1`'s context — intentionally out of scope for this step.
- The chunked cross-entropy function only demonstrates memory savings for loss evaluation (`torch.no_grad()`); a training-time memory-efficient version would additionally need to chunk the backward pass.
