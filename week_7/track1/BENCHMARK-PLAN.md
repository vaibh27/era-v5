# Track 1 — Benchmark plan (resume here after compaction)

Model under test: `code/kronecker_v2_embedding.py` — `KroneckerV2Embedding` = `[ Kronecker byte×position
text ‖ arithmetic_embedding.ContinuousCRTEmbedding math ]`. torch. `pip install -r
code/arithmetic_embedding/requirements.txt`.

## What is benchmarked (build `code/benchmark.py`, write `results/benchmark.json` + a table)

**B1 — Arithmetic exactness & magnitude extrapolation (headline).**
- Task: given two number tokens, produce `a+b` and `a*b`; check exact-match after decode.
- Splits: train-range (small, e.g. operands < 100) vs held-out EXTRAPOLATION range (large, up to CRT capacity).
- Compare: KroneckerV2 (fixed CRT operator, no training) vs learned `nn.Embedding` + MLP operator (train.py-style) vs linear "value-on-axis".
- Expect: KroneckerV2 ≈ 100% at all magnitudes (operator is fixed); learned collapses out-of-range (~21% monolithic baseline already measured).

**B2 — Text reversibility.**
- Task: word round-trip `decode(embed(w)) == w` over a word list (lengths 1..32, incl. long words).
- Compare: KroneckerV2 text side (reversible) vs learned `nn.Embedding` (not reversible).
- Metric: exact round-trip %. Expect KroneckerV2 = 1.0 within 32 bytes.

**B3 — Text + math coexistence (the report's Experiment 8 — the untested one).**
- Task: a tiny transformer on a mixed stream (words + numbers), two heads: (a) a text/next-token proxy, (b) arithmetic answer.
- Compare: input embedding = KroneckerV2 vs text-only Kronecker vs learned embedding.
- Metric: does adding the math register KEEP text performance while GAINING arithmetic? (Δtext ≈ 0, arithmetic ↑.)

**B4 — Capacity scaling (cheap, mostly done).**
- Exact range vs #primes / dimension (linear dim → exponential range). Reuse `arithmetic_embedding` evaluate.

## Baselines
learned `nn.Embedding` (+ MLP/bilinear operator), plain Kronecker (text only), optional xVal-style value scaling.

## Metrics
exact-match % (add, mul), round-trip %, extrapolation gap (in-range vs out-of-range), trainable-param count.

## Order of work
1. `code/benchmark.py` with B1 + B2 (pure, fast, no training for KroneckerV2; small train for the learned baseline). — **DONE**
2. B4 (reuse existing exact-CRT eval). — **DONE**
3. B3 last (needs a tiny transformer + training) — only if time. — **DEFERRED** (not built; user chose to stop after B1/B2/B4).
4. Write `results/benchmark.json` + a summary table into this folder's README. — **DONE**

## Status (2026-08-14)
`code/benchmark.py` implements B1/B2/B4, writes `results/benchmark.json`, prints + README-embeds the tables.
Results: KroneckerV2 add/mul = 100% in- AND out-of-range (0 trainable operator params); learned baseline
30% in-range → 0% out-of-range; linear-axis 100% add / 0% mul. B2 text round-trip 100% in-vocab, 83% novel
(one 40-char word truncates at 32 bytes). B4 shows linear dim → exponential capacity, exact at every size.
Only B3 (mixed-stream tiny transformer, Experiment 8) remains unbuilt.

## Known expectations (from work already done)
- CRT add/mul exact = 1.0 over all pairs; monolithic learned = 21%; learned limb length-gen (base-10) = 100%.
- Wrap-around past M is the honest failure; magnitude-blindness + error-correction are documented roadmap (`docs/MATHS-THAT-HELPS.md`).
