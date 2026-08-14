# Problem 1 — Math-structured Kronecker embedding (text **and** math in one model)

> *"What if embeddings can store mathematical structure as well. Say `9` … somehow it has stored the
> meaning of 9 (in absolute math terms), such that when we actually do `9 + 9`, the mathematical-meaning
> part of the embedding is itself `18`! When we do `9*9` it becomes `81`! **How much can we push? Can we
> describe whole mathematics and all mathematical operations using this?** Of course we need some space
> for alphabets/words as well; for that we can use the 32 existing spaces, and add this new concept into
> new ones (that are appended)."* — the assignment, Problem 1

---

## TL;DR

A **single embedding for words and numbers**:

```
E(token) = [  Kronecker text  (byte × position)   ‖   CRT math register  ]
              the "32 existing spaces" for words       appended math dims
```

A **fixed operator** (zero trained parameters) does **exact** `+` and `×` directly on the math dims —
`9+9→18`, `9*9→81`, `12*11→132` — and the result **decodes back** to the number. Words leave the math
dims at zero, so language is untouched. One trainable `W_proj → d_model` makes the whole thing a drop-in
transformer input embedding.

**Measured (`code/benchmark.py`):** 100% exact `+`/`×` at *every* magnitude below capacity —
in-range and out-of-range — versus a learned baseline that collapses to 0% out of range. Text round-trips
100%. Exact range grows *exponentially* with dimension.

---

## 1. The problem, restated

The prompt has two halves:

1. **A concrete mechanism:** an embedding where the *value* of a number lives in the vector, so that a
   fixed operation on the vectors *is* the arithmetic — `emb(9) ∘ emb(9) = emb(18)`. Words keep their own
   space (the 32 slots); the math is appended.
2. **A research question:** *how far can this be pushed — can it describe all of mathematics?*

This README answers both: the mechanism (Sections 2–4), the proof it works (Section 5), and an honest
answer to "how far can this be pushed" including where it hits a wall (Section 6).

---

## 2. Architecture

With the default moduli `(5, 7, 11, 13, 17, 19)`:

| part | what it stores | dimension |
|---|---|---|
| **Kronecker text** | the spelling of the token (byte × position) | 256 × 32 = **8192** |
| **CRT math register** | the numeric *value* (residues) | 5+7+11+13+17+19 = **72** |
| **total** | | **8264** |
| **exact integer range** | `M = 5·7·11·13·17·19` | **0 … 1,616,614** |

A gate `is_number(tok)` decides whether the math register is filled. Words → math dims are zero.

### 2.1 The text half — reversible, tokenizer-free

`KroneckerText.encode` UTF-8-encodes the token and builds a `256 × 32` grid: a spike of height `1/√L` in
row `byte[p]` for each position `p`. It is **deterministic** (no vocabulary, any string encodes) and
**reversible**: `decode` takes `argmax` down each position column to read the byte back. This is
Kronecker V1 — the "32 existing spaces for words."

### 2.2 The math half — the CRT register (the real machinery)

Three steps make arithmetic a *fixed* operation:

**(a) Represent a number by its remainders (CRT).** Store `n` as
`(n mod 5, n mod 7, n mod 11, n mod 13, n mod 17, n mod 19)`. The Chinese Remainder Theorem guarantees
this fingerprint is **unique** for every `n < M`, so it is a lossless code for `0 … M−1`.

**(b) Make it a continuous vector.** Each residue `r mod m` becomes a **one-hot** block of length `m`;
concatenate the six blocks → a 72-dim vector.

**(c) Arithmetic is a fixed bilinear tensor, because CRT is a ring homomorphism.** Remainders add and
multiply independently per channel:

```
(x + y) mod m = ((x mod m) + (y mod m)) mod m          (same for  ×)
```

On one-hot vectors this is a precomputed tensor `T[k,i,j] = 1  ⇔  (i ∘ j) mod m = k`; `add` and `mul`
are one `einsum` each against these tables — **no trained parameters**. Decoding argmaxes each residue
block and reconstructs the integer with the CRT inverse.

**Worked example** (moduli `5,7,11`, so `M=385`):

```
E(9) = (4, 2, 9)
9 + 9 :  (4+4 mod5, 2+2 mod7, 9+9 mod11) = (3, 4, 7) = E(18)   ✓
9 × 9 :  (4·4 mod5, 2·2 mod7, 9·9 mod11) = (1, 4, 4) = E(81)   ✓
```

### 2.3 One model, one projection

`embed_features(tok)` computes the text code always, and fills the math register only for non-negative
integers `< M`. `forward(tokens)` stacks the fixed 8264-dim features and applies a single trainable
`W_proj → d_model` — a normal transformer input embedding, Kronecker-V1 style. Arithmetic (`add`, `mul`)
and decoding (`number_of`, `text_of`) operate on the fixed features.

The math register is `arithmetic_embedding`'s verified `ContinuousCRTEmbedding`, reused **verbatim**
(byte-identical `crt.py` / `continuous_crt.py`). This model is that project's `HybridArithmeticEmbedding`
with its *learned* semantic side replaced by the deterministic Kronecker text code — built on the
existing verified code, not a reimplementation.

---

## 3. Run it

```bash
pip install -r code/arithmetic_embedding/requirements.txt   # torch
python3 code/kronecker_v2_embedding.py                       # the demo
```

Verified output:

```
Kronecker V2 embedding: dim = 8264  (text 8192 + math 72), exact math range 0..1616614
words   : apple / a / cat   -> text decodes correctly, math dims empty
numbers : 9 / 42 / 255      -> BOTH a text embedding AND a math value
math    : 9+9 -> 18   9*9 -> 81   12*11 -> 132     (fixed operator = CRT tensors)
mixed   : "buy 9 apples for 42 rupees" -> each token embedded uniformly; gate routes words vs numbers
proj    : forward(['buy','9','apples']) -> [3, 128] via one trainable W_proj
```

---

## 4. What was measured (`code/benchmark.py` → `results/benchmark.json`)

The KroneckerV2 operator is **fixed** (0 trainable params), so it is exact at every magnitude below `M`.
The learned baseline holds a trainable vector per integer in a small vocab, so it has **no representation
out of range**.

### B1 — arithmetic exactness & magnitude extrapolation

Operands trained on `[0, 64)`; out-of-range = sums up to 1.4M / products up to 1.44M (both `< M`, so
KroneckerV2 does not wrap). Exact-match %, seed 7:

| model | add in-range | mul in-range | add **out-of-range** | mul **out-of-range** |
|---|---|---|---|---|
| **KroneckerV2** (fixed CRT, **0** params) | **100%** | **100%** | **100%** | **100%** |
| learned (`nn.Embedding`+MLP, 69k params) | 30% | 30% | **0%** | **0%** |
| linear value-on-axis | 100% | 0% | 100% | 0% |

The learned operator collapses out of range; linear handles `add` (it *is* linear) but never `mul`. Only
the fixed CRT register is exact for both at all magnitudes.

### B2 — text reversibility (char-level round-trip)

| model | in-vocab | novel / OOV |
|---|---|---|
| **KroneckerV2 text** | **100%** | **83%** (one miss = a 40-char word, honest 32-byte truncation) |
| learned `nn.Embedding` | 100% | **0%** (not tokenizer-free) |

### B4 — capacity scaling (the answer to "how far can this be pushed?")

`math_dim` grows **linearly** (∑ primes) while exact range grows **exponentially** (∏ primes):

| #primes | math_dim | exact range | exact spot-check |
|---:|---:|---:|:---:|
| 1 | 5 | 5 | ✓ |
| 3 | 23 | 385 | ✓ |
| 6 | 72 | 1,616,615 | ✓ |
| 8 | 124 | ~1.08 × 10⁹ | ✓ |

---

## 5. How far can this be pushed? — the ceiling

**Range:** as far as you like, cheaply — dimension is linear, range is exponential (B4). Want billions?
Add a few primes.

**But not "all of mathematics."** There is a single wall, and it has four faces. Making an operator
*magnitude-uniform* (so it extrapolates) forces you to give up one of three things:

| representation | operator | what you lose |
|---|---|---|
| **CRT / residues (this model)** | fixed, coordinatewise +,× | **magnitude ordering** (residues can't compare sizes) |
| log / complex-log | ×→translation | **exact digits** (only significant figures) |
| p-adic / Witt digits | fixed carry polynomial | magnitude ordering (≈ digit embeddings) |
| ghost / Witt over ℚ | coordinatewise +,× | **numerical conditioning** (doubly-exponential — unusable) |

This model deliberately sits in the **"exact digits, bounded range, magnitude-blind"** corner. Two hard
consequences follow, and both are real limits (Section 7): it **wraps** past `M`, and it cannot tell you
whether one number is bigger than another.

**"Whole mathematics" is provably out of reach.** Exact integer arithmetic is tied to undecidable
problems (Hilbert's 10th, Gödel). You can *encode* arithmetic syntax exactly, but no fixed-dimensional
latent computation can decide all arithmetic truth. The defensible, and still strong, claim is the one
demonstrated here: **a dedicated arithmetic subspace realizes exact `+`/`×` over a bounded domain via fixed
operators, while a separate subspace carries language.** Full theory: [`docs/report.md`](docs/report.md).

---

## 6. The interesting open question

The operator here is **handed** to the model (the fixed CRT tensors), which is *why* it is exact. When
you instead ask a plain network to **learn** the structure from examples (`emb(a)×emb(b)=emb(a·b)`), it
does not discover it: the learned baselines land at **~1–30%** (Section 4; see also the failed-attempts
record). Grokking results show networks *can* eventually learn exactly this clock/CRT structure under the
right conditions — so the open question the paper can attack is **whether a model can reliably *discover*
this structure rather than be handed it.** Handed = exact today; discovered = the research prize.

---

## 7. Honest limits

- **Wraps silently past `M`.** `2000×1000 = 2,000,000 → 383,385` (no overflow flag). Exact iff the
  *result* `< M`; the boundary for `a×b` is the hyperbola `a·b < M`, not any per-operand cap.
- **Magnitude-blind.** `cos(emb(99), emb(100)) = cos(emb(99), emb(6)) = 0`. No ordering, no plausibility
  check.
- **Only non-negative integer tokens route to math.** `-3`, `nine`, `3.5`, and numbers `≥ M` fall through
  to a zero math register. "nine" ≠ "9" (natural-language number words are not yet unified).
- **Text truncates at 32 bytes.** A 40-char word decodes to 32 (the B2 miss).
- **Coexistence not yet measured.** That adding the math register keeps language performance intact
  (report's Experiment 8) is architecturally true but **not benchmarked** (would be benchmark "B3").

The mathematics that *would* push past these (redundant-RNS overflow detection & error correction, the
RNS diagonal/core function for magnitude) is written up as a roadmap in
[`docs/MATHS-THAT-HELPS.md`](docs/MATHS-THAT-HELPS.md); plain-language limits in
[`docs/LIMITS-PLAIN.md`](docs/LIMITS-PLAIN.md).

---

## 8. What didn't work

Recorded in full in [`../FINAL-REPORT.md`](../FINAL-REPORT.md) (Part II). The headline dead-ends for this
track: the **monolithic learned operator** (~1–30%, never discovers the structure); **Witt/ghost vectors**
(doubly-exponentially ill-conditioned); **theta / abelian-variety geometry** (geometrizes only the
addition carry — `×` is not a variety morphism); the **naïve `[value ‖ log]` dual channel** (desyncs after
one op); and the **base-256 limb multiplier** (product-cell only ~4.6% → multiply collapses). Also noted:
the `×`-as-rotation-via-discrete-log mechanism is **already published** (arXiv 2606.17399) — rebuilt here,
not invented.

---

## 9. Files

- `code/kronecker_v2_embedding.py` — **the model** (text ‖ math, gate, exact arithmetic, decode, W_proj).
- `code/benchmark.py` — B1/B2/B4; writes `results/benchmark.json`.
- `code/arithmetic_embedding/` — the verified CRT engine (`crt.py`, `continuous_crt.py`) plus the trained
  limb-arithmetic track and its tests/artifacts.
- `docs/report.md` — full theory (theorem, lemmas, proofs). `docs/LIMITS-PLAIN.md`,
  `docs/MATHS-THAT-HELPS.md` — limits & roadmap.
- `results/benchmark.json` — measured results.
