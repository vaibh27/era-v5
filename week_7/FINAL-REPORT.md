# Kronecker Embedding V2 — Final Report

One report for the whole submission. This is a record of what was **tried** and **measured** — not a claim that any of these problems is
solved. Part I is a **per-track write-up** (problem, approach, what was measured, honest limits). Part II is
the **consolidated list of every attempt that failed or was closed as a dead-end**, across all tracks.
Every number below was reproduced from the runnable code in this folder (see each track's `README.md` for
the exact command).

| # | Problem | What was tried / measured |
|---|---------|--------------------------|
| 1 | Store math structure + words in one embedding | fixed operator, **100% exact** `+`/`×` at all magnitudes < M; learned baseline collapses |
| 2 | Multimodal (text + image + audio) | `content ⊗ position` generalized to patches/frames; design only, not built |
| 3 | Dynamic length (kill the 32-cap) | **0% collisions vs V1's 100%**; exact to length ≈ k, tunable |
| 4 | Fourier alternative (char = wave, word = sum) | anagrams distinct; **0% collisions vs V1's 100%** |
| 5 | Reversible head (drop softmax, 1M vocab free) | open-vocab copy **0.886 vs 0.000**; head **265K flat vs 129M** @1M vocab |

---

# Part I — Per-track write-ups

## Track 1 — Math-structured embedding (`track1/`)

**Problem.** Make one embedding carry both language and numeric *value*, so a fixed operator does exact
arithmetic on the vector (`9+9→18`, `9*9→81`) and decodes back — the "use the 32 spaces for words, append
a math register" idea.

**Approach.** `E(token) = [ Kronecker text (byte×position, 8192 dims) ‖ CRT math register (72 dims) ]`.
Every token gets the deterministic, reversible byte×position code. Number tokens additionally fill the
**CRT/RNS math register** (residues mod `(5,7,11,13,17,19)`, one-hot per channel). Because CRT is a ring
homomorphism, `+`/`×` are **fixed bilinear tensors** applied channel-wise — **zero trained parameters**.
An `is_number` gate routes tokens; one trainable `W_proj → d_model` makes it a drop-in transformer input
embedding. The math register is `arithmetic_embedding`'s verified `ContinuousCRTEmbedding`, reused
verbatim.

**When measured (`code/benchmark.py`).**
- **B1 arithmetic:** KroneckerV2 = **100% exact** add & mul, in-range *and* out-of-range, with **0**
  trainable operator params. Learned `nn.Embedding`+MLP: ~30% in-range → **0% out-of-range**. Linear
  value-on-axis: 100% add / 0% mul.
- **B2 text:** char-level round-trip **100%** in-vocab, 83% on novel words; learned embedding **0%** OOV.
- **B4 capacity:** math dim grows linearly (∑ primes), exact range exponentially (∏ primes) — up to
  `M ≈ 1.08e9` at 124 dims.

**Limits (honest).** Wraps silently past `M = 1,616,615` (e.g. `2000×1000 → 383385`); magnitude-blind
(cos(99,100)=0); only non-negative integer tokens route to math (`-3`, `nine`, decimals do not); text
truncates at 32 bytes; the text+math coexistence test (report's Experiment 8 / benchmark B3) is
**not run**. The mechanism that *would* break the wrap ceiling — recurrent limb arithmetic — exists in
`code/arithmetic_embedding/src/` but was **not wired into** the unified model.

## Track 2 — Multimodal Kronecker (`track2/`) — design only

**Problem.** Extend Kronecker to images and audio, not just text.

**Approach (documented, not built).** Generalize `κ = (1/√N) Σ content(xᵢ) ⊗ position(i)` from
bytes-in-a-word to **atoms-in-a-signal**: image patch = VQ codeword ⊗ `row ⊗ col` (2-D position); audio
frame = spectral codeword ⊗ `time`. Per-modality `W_proj^m` lands all three in one shared `d_model`
space. `track2/README.md` gives the preprocessing, the fixed-vs-learned-codebook trade-off, and a
proof protocol (MNIST/CIFAR patches + FSDD audio + digit words → classification, locality probe,
cross-modal retrieval).

**Not built** — this problem was left as a design only.

## Track 3 — Dynamic length (`track3/`)

**Problem.** V1 forces 32 positions and **crops** anything longer. Make the codec length-dynamic.

**Approach.** Replace the one-hot position (whose dimensionality *is* the cap) with a smooth **Fourier
positional basis** `φ(p) ∈ R^k`: `κ(b) = (1/√L) Σ_p onehot₂₅₆(bₚ) ⊗ φ(p)`, dimension `D = 256·k`
independent of length, any `p` representable, exact linear (pinv) decode for `L ≤ k`.

**When measured (`experiments.py`).** Adversarial strings sharing a 32-byte prefix: **V1 collides
100%** (L2 dist 0.0), **Fourier 0%** (L2 dist 0.73). At equal dimension (`k=32`, D=8192) Fourier matches
V1's exact capacity and stays ≥ V1 at every length. Real >32-byte words
(`pneumonoultramicroscopic…` 45, a 65-byte German compound) round-trip exactly where V1 crops.

**Limits.** No free lunch on lossless dimension — exact decode of `N` positions still needs `k ≳ N`
(`D ≈ 256N`); the win is no-cap / graceful degradation / no-collisions / a tunable knob, **not**
compression. Decoder is given `L` (blind length recovery not solved). Downstream LM comparison not run.

## Track 4 — Fourier alternative (`track4/`)

**Problem.** A *real* Fourier form of Kronecker: represent each character as a wave and **add** them.

**Approach.** Replace Kronecker's **product** with a **superposition** (holographic / VSA binding):
`X(word) = (1/√L) Σ_p χ_{bₚ} ⊙ ρ_p`, χ = per-character unit wave, ρ = per-position rotation. `D` is a
free knob independent of alphabet and length. Decode = rotate back and correlate.

**When measured (`experiments.py`).** Anagrams (`listen/silent/enlist`, `cat/act/tac`) decode
correctly and stay far apart — **position phase carries order**. Distinct strings sharing a 32-byte
prefix: **V1 100% collision, Fourier 0%**. Round-trip exact for short/medium words; larger `D` = more
characters before crosstalk (D=2048 exact to length 64).

**Limits.** Reversible only **up to crosstalk**: signal per char ~ `D`, crosstalk ~ `√(D·L)`, so exact
decode needs `D ≫ L` — graceful, not a cliff, but not unconditionally exact.

## Track 5 — Reversible head (`track5/`)

**Problem.** Kronecker is invertible, so the `V×d_model` **softmax head** can be dropped → 1M+ vocab for
free.

**Approach.** Predict a **byte-grid** (per-position byte distribution) and invert κ by argmax-per-block,
instead of scoring a vocabulary. Head = `Linear(d_model, MAXLEN×257)`, size **independent of `V`**. Same
tiny GPT backbone and Kronecker input embedding for both heads; only the head differs.

**When measured (`run.py` → `results.json`).** On a prefix-copy task: **in-vocab copy 1.000 for
both** (parity). On **500 novel words never seen in training**: softmax **0.000** (impossible by
construction), byte-grid **0.886**, all decoded outputs valid words (valid-decode 1.0), κ round-trip
**2500/2500**. Output-head params at `V=1,000,000`: softmax **129M** vs byte-grid **265K (flat)**.

**Limits.** Exact reversibility only within the position cap (MAXLEN=8 in this demo; needs Track 3/4 for
unbounded open-vocab). Per-position independence assumption → the residual ~11% OOV errors are
single-byte slips (never invalid strings). Copy task is a controlled probe, not a full LM.

---

# Part II — Failed attempts & dead-ends (all finalized tracks)

The submission is stronger for what it *ruled out*. These are real, recorded negative results — the
sources are `track1-math-register/SYNTHESIS.md`, the per-track "honest limits", and this session's
empirical probes.

## Track 1 (math) — the bulk of the failures

1. **Monolithic learned operator — the headline failure.** Ask a plain `nn.Embedding`+MLP to *learn*
   `emb(a)∘emb(b)=emb(a∘b)` from examples: it never discovers the exact structure. Measured this
   session: aev2's learned model **~1–3%** (40 epochs, Z₁₀₁); benchmark learned baseline **30% in-range →
   0% out-of-range**; prior runs plateau near the documented **~21%**. This is *why* the operator is
   handed to the model, not learned.

2. **Ghost / Witt vectors over ℚ.** Coordinatewise `+` and `×` in principle, but the coordinates are
   **doubly-exponentially ill-conditioned** → numerically dead on arrival as a codec.

3. **Theta functions / abelian-variety geometry.** Geometrizes only the **addition carry**; `×` is *not*
   a morphism of an abelian variety → closed as a multiplication codec (but yielded rigorous
   carry-as-cocycle math).

4. **Naïve `[value ‖ log]` dual channel.** Desyncs after a single op — the value half says `18`, the log
   half says `log 81`; keeping them consistent needs an expensive log-sum-exp resync every step.

5. **Log / complex-log "cone" register.** Extrapolates magnitude and ordering gracefully, but only to
   **K significant figures** — never the exact digits of a large product. The exact complement of CRT.

6. **Large digit-alphabet limb multiplier.** The recurrent-limb approach *does* break the range wall
   (exact to 128-digit add, 64-digit mult at base 10/32) — but its exactness is bounded by the **base**,
   not length. At **base 256** (Kronecker's byte alphabet) the product cell reaches only **~4.6%** on its
   table (this session; docs quote ~22% for a larger cell) → any multi-digit multiply collapses to ~0. It
   is a memorized table + a hand-written schoolbook loop, not learned reasoning.

7. **CRT register wrap.** Past `M` the register silently returns wrong answers (`2000×1000 → 383385`),
   with no overflow flag. The honest ceiling of the finalized Track 1.

8. **Novelty audit — not first.** The "×-as-rotation via discrete log" mechanism is **already published**
   (arXiv 2606.17399, verified). It was independently rebuilt here, not invented.

9. **The strong hypothesis is impossible.** "A finite continuous embedding that *is* unrestricted ℤ/ℝ
   arithmetic" is ruled out by rigidity/no-go results (and undecidability for full arithmetic truth). The
   hypothesis was narrowed to a bounded CRT subspace — a negative result that reshaped the whole track.

10. **Not attempted (open, not failed):** Fork 1's learnability probe — can a transformer *produce* the
    codec features of novel large products and decode them? — never ran (session limit). The text+math
    coexistence test (Experiment 8 / benchmark B3) was likewise deferred.

## Track 3 (dynamic length)

11. **No compression win.** The hoped-for "fewer dimensions for lossless encoding" did **not**
    materialize — exact decode of `N` positions still needs `k ≳ N` (`D ≈ 256N`), the same order as V1.
    The wins are no-cap / graceful / no-collision / tunable, not smaller.

12. **Blind length recovery — unsolved.** The decoder is given `L`; recovering length from the embedding
    alone is left as future work.

## Track 4 (Fourier chars)

13. **Naïve amplitude-sum collides.** Summing character-waves *without* the per-position phase makes
    anagrams identical. Fixed by making position a rotation (phase carries order) — but the naïve version
    is a genuine dead-end.

14. **Not unconditionally exact.** Superposition trades exactness for compactness; decoding is exact only
    while `D ≫ L`. Beyond that it degrades (gracefully) — not a lossless codec.

## Track 5 (reversible head)

15. **Softmax cannot emit novel words — by construction.** The fixed-vocabulary baseline scores **0.000**
    on unseen words. This is the failure that *motivates* the byte-grid head, not a bug in it.

16. **Per-position independence leaks.** The byte-grid head assumes byte independence given the hidden
    state → ~11% of OOV words have single-byte slips. And exact reversibility holds only within the
    position cap (needs Track 3/4 to go unbounded).

## Track 2 (multimodal)

17. **Never built.** The only problem left as design-only — an attempt not made.

---

## Bottom line

Tracks 1, 3, 4, 5 each have a **working prototype with runnable, reproduced evidence** while naming
exactly where they break — these are explorations, not finished solutions. The recurring wall across the
math work is a single trade-off — recorded
as *"one wall, four faces"*: making an operator magnitude-uniform costs you **either** magnitude ordering
**or** exact digits **or** numerical conditioning. Kronecker V2's math register deliberately takes the
"exact digits, bounded range, magnitude-blind" corner; the other tracks kill V1's 32-cap (3, 4) and its
vocabulary bottleneck (5). Track 2 remains a design.
