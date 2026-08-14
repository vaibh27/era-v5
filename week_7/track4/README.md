# Problem 4 — A real Fourier alternative to Kronecker

> *"What is a REAL Fourier alternative of Kronecker? Why can't I represent each character like a
> Fourier wave, and just add them to make a word?"*

## Approach (what was tried)
Kronecker V1 uses a **product** (byte ⊗ position → a one-hot grid of dim 256×32). The Fourier
alternative uses a **superposition**: each character is a **wave**, each position is a **rotation**
(phase shift), and a word is the **sum** of its character-waves —

```
X(word) = (1/√L) · Σ_p  χ_{b_p} ⊙ ρ_p        χ_c = a fixed unit-modulus wave (character c)
                                              ρ_p = a fixed rotation (position p)  ⊙ = elementwise
```

Dimension `D` is a **free knob, independent of alphabet size and word length** — no 256×32 blow-up,
**no 32-position cap**. Decoding is Fourier analysis: rotate back by `conj(ρ_p)` and correlate with
each character-wave (`character at p = argmax_c ⟨χ_c, X·conj(ρ_p)⟩`). This is the Holographic /
Vector-Symbolic binding construction, which is exactly "each character is a wave, add them up."

## How to run
```bash
pip install numpy matplotlib
python3 experiments.py        # ~seconds; writes plots/ and prints the tables below
```

## What was measured
1. **Round-trip while capacity lasts, then graceful — no hard cap** (`plots/exp1_roundtrip.png`).
   Exact recovery for short/medium words; larger `D` = more characters before crosstalk. At `D=2048`,
   exact to length 64; `D=512` degrades gracefully past ~40. V1's wall at 32 does not exist here.

   | length | D=512 | D=1024 | D=2048 | D=4096 |
   |---:|---:|---:|---:|---:|
   | 20 | 1.000 | 1.000 | 1.000 | 1.000 |
   | 32 | 0.996 | 1.000 | 1.000 | 1.000 |
   | 64 | 0.850 | 0.997 | 1.000 | 1.000 |

2. **Anagrams stay distinct.** `cat/act/tac`, `listen/silent/enlist`, `abc/bca/cab` all decode
   correctly and are far apart (min pairwise L2 ≈ 52–64). A naïve amplitude-sum would collide;
   the **position phase carries order**, so it doesn't.

3. **No collisions where V1 crops.** Distinct strings sharing a 32-byte prefix: **V1 collides 100%**
   (its tail is cropped), **Fourier-chars 0%**.

4. **Concrete long words round-trip** — `supercalifragilisticexpialidocious` (34), 
   `pneumonoultramicroscopicsilicovolcanoconiosis` (45) — where V1 crops at 32.

## Honest scope
- **Reversible up to crosstalk.** Superposition trades exactness for compactness: signal per character
  scales like `D`, crosstalk like `√(D·L)`, so exact decode needs `D` comfortably larger than the
  length. Bigger `D` → longer exact words; the failure is graceful, not a cliff.
- Relationship to Problem 3: both use a smooth basis to kill the 32-cap. Problem 3 keeps the Kronecker
  **product** with a Fourier *position*; Problem 4 replaces the product with **pure summation of
  character-waves** — the "add waves to make a word" the prompt asks for.

## Files
`codec.py` — `FourierCharCodec` (encode/decode) + `V1Codec` for contrast. `experiments.py` — Exp 1–4.
