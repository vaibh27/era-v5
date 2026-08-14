# Track 3 — Dynamic-length Kronecker embeddings (killing the 32-cap)

**Problem (from the assignment):** Kronecker V1 forces **32 positions for every word** — even `"a"`
pays for 32 slots, and any word **longer than 32 bytes is cropped** (its tail is silently discarded).
Can the codec be **dynamic** — encode any length, no cropping?

**Answer:** Yes. Replace the *one-hot* position (whose dimensionality **is** the length cap) with a
smooth **Fourier positional basis** `φ(p) ∈ R^k`. The codec becomes
`κ(b) = (1/√L) · Σ_p onehot₂₅₆(bₚ) ⊗ φ(p)`, dimension `D = 256·k` **independent of length**, `p` any
integer (no cap), and it is **exactly invertible for L ≤ k** via a least-squares decode.

---

## The idea in one picture

V1 stores "byte at position" in a `256 × 32` grid — 32 hard columns. The Fourier codec swaps the 32
one-hot columns for `k` **sinusoidal** columns `φ(p)` (frequencies evenly spread in `(0, π]`). Now
position is a smooth
`k`-vector instead of a 1-of-32 slot, so:

- any position `p` is representable → **no crop**;
- decoding is a matched linear inverse: `X̂ = √L · M · pinv(Φ)`, `byte(p) = argmax_v X̂[v,p]`, which is
  **exact whenever `L ≤ k`** (the position features are full-rank) and **degrades gracefully** beyond;
- `k` is a **free knob** decoupled from the number 32: pay `256·k` dims, decode losslessly up to `≈k`.

See `PLAN.md` for the full construction and `codec.py` for both codecs (~120 lines, pure numpy).

## How to run

```bash
pip install numpy matplotlib
python3 experiments.py        # ~12 s on CPU; writes plots/ and prints tables
```

## What was measured

### 1. No hard cap — graceful degradation instead of a cliff  (`plots/exp1_accuracy_vs_length.png`)
V1 is perfect up to 32 then its per-character accuracy falls as `32/L` because bytes past 32 are
**gone**. The Fourier codec is **exact up to `k`**, then degrades smoothly. At **equal dimension**
(`k=32`, `D=8192`), Fourier matches V1's exact capacity and stays **≥ V1 at every length** — with the
tail still partially recoverable rather than discarded.

| length L | V1 (D=8192) | Fourier k=16 (D=4096) | Fourier k=32 (D=8192) | Fourier k=64 (D=16384) |
|---:|---:|---:|---:|---:|
| 16 | 1.000 | 1.000 | 1.000 | 1.000 |
| 32 | 1.000 | 0.548 | **1.000** | 1.000 |
| 33 | 0.970 | 0.538 | **1.000** | 1.000 |
| 48 | 0.667 | 0.379 | **0.722** | 1.000 |
| 64 | 0.500 | 0.305 | **0.541** | 1.000 |
| 128| 0.250 | 0.183 | **0.286** | 0.526 |
*(per-character reconstruction accuracy; Fourier k=32 has the same 8192 dims as V1)*

### 2. V1 catastrophically collides; Fourier does not  (`plots/exp2_collisions.png`)
Take **distinct, equal-length** strings that share a 32-byte prefix and differ only afterward. V1
maps them to the **identical** embedding (its suffix is cropped away):

```
V1      : collision rate 100.0%   (mean L2 distance 0.0000)
Fourier : collision rate   0.0%   (mean L2 distance 0.7321)
```
This is the core failure of the 32-cap made concrete: past byte 32, V1 literally cannot tell two words
apart. The dynamic codec keeps them fully distinct.

### 3. Lossless length scales linearly with the knob `k`  (`plots/exp3_dim_vs_maxlen.png`)
Max length decodable at ≥99% accuracy, vs V1's fixed wall of 32:

| k | D = 256·k | max exact length |
|---:|---:|---:|
| 16 | 4096 | 18 |
| 32 | 8192 | 34 |
| 64 | 16384 | 66 |
| 128 | 32768 | 131 |
| **V1** | **8192** | **32 (hard cap — crops beyond)** |

### 4. Concrete long words round-trip  (printed by `exp4`)
With `k=64`, real >32-byte words reconstruct exactly where V1 crops them:

```
len | V1 ok? | Fourier ok? | word
 45 |   NO   |    YES      | pneumonoultramicroscopicsilicovolcanoconiosis
 65 |   NO   |    YES      | Rindfleischetikettierungsueberwachungsaufgaben...  (German)
 34 |   NO   |    YES      | supercalifragilisticexpialidocious
  1 |   YES  |    YES      | a
V1 recovered e.g. 'pneumonoultramicroscopicsilicovo'  <- cropped at byte 32
```

## Honest scope & limitations

- **No free lunch on lossless dimension.** Exact decoding of `N` positions needs `k ≳ N`, i.e. `D ≈ 256·N`
  — the same order as V1 with a 32→N cap. The win is **not** fewer dims for lossless; it is: (a) **no
  hard cap / no cropping**, (b) **graceful** degradation over the budget instead of a cliff, (c) **no
  catastrophic collisions**, and (d) `k` as a **tunable** length/dimension knob rather than a fixed 32.
- **Decoder is given `L`** (as V1 effectively is via its ≤32 nonzero columns). Blind length recovery is
  left as future work.
- **Optional LM not run.** torch is available; a downstream char/word-LM comparison on a long-token
  corpus (perplexity at long-word positions) is the natural next step, deferred to keep this proof
  self-contained. Protocol: use each codec as the fixed input embedding (+ one learned `W_proj`), train
  a tiny transformer, compare loss on tokens of length >32.

## Files
- `PLAN.md` — construction, decode math, experiment list.
- `codec.py` — `V1Codec` and `FourierPosCodec` (encode + decode), numpy only.
- `experiments.py` — Exp 1–4, writes `plots/*.png`, prints tables.
- `plots/` — the three figures above.
