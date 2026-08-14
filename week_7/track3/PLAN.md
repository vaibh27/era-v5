# Track 3 — Dynamic-length Kronecker codec · PLAN

**Claim to prove:** replacing V1's *one-hot* position (which hard-crops words > 32 bytes and wastes
budget on short words) with a smooth **Fourier positional basis** `φ(p) ∈ R^k` gives a deterministic
word codec that (a) encodes **arbitrary length with no cropping**, (b) stays **decodable/invertible**,
and (c) does so at dimensionality `D = 256·k` that is **independent of length** and can be **≤ V1's 8192**.

---

## The two codecs

Both map a byte string `b = (b_1,…,b_L)`, `b_p ∈ {0..255}`, to a vector, viewed as a `256 × (pos-dim)`
grid `M` (row = byte value, "column space" = position code).

### V1 baseline (one-hot position)
```
κ_V1(b) = (1/√L) Σ_{p=1..min(L,32)}  onehot256(b_p) ⊗ onehot32(p)
```
- Grid `M ∈ R^{256×32}`; `M[b_p, p] = 1/√L`. Dim `D = 256·32 = 8192`.
- **Hard cap 32:** positions > 32 are dropped (cropped) → information lost.
- **Decode:** for each column `p`, `byte = argmax_v M[v, p]`. Recovers ≤ 32 bytes only.

### Fourier-position codec (this work)
```
φ(p) ∈ R^k,  φ(p)_2i   = sin(ω_i p),  φ(p)_2i+1 = cos(ω_i p),  ω_i = base^(-2i/(k))   (i=0..k/2-1)
             φ(p) normalized to unit L2 norm  (raw ‖φ(p)‖² = k/2)
κ_F(b) = (1/√L) Σ_{p=1..L}  onehot256(b_p) ⊗ φ(p)
```
- Grid `M ∈ R^{256×k}`; row `v` = `(1/√L) Σ_{p: b_p=v} φ(p)`. Dim `D = 256·k`, **independent of L**.
- **No cap:** `p` is any positive integer.
- **Decode (correlation / matched filter):** for each position `p = 1..L`,
  `score(v,p) = ⟨M[v], φ(p)⟩`; `byte(p) = argmax_v score(v,p)`.
  Correct because `⟨φ(p),φ(p)⟩ = 1` and `⟨φ(p),φ(p')⟩` is small for `p ≠ p'` (Dirichlet-like kernel of
  the frequency set), so the diagonal term dominates the cross-talk. Decoder is given `L` (as V1
  effectively is via its ≤32 nonzero columns); recovering `L` blind is future work.

**Why it works / trade-off.** The `{φ(p)}` are near-orthonormal while the frequency spread keeps
positions distinguishable. Cross-talk accumulates with `L` (more summed terms), so decode degrades
**gracefully** with length instead of V1's hard cliff at 32. `k` sets resolution: bigger `k` ⇒ more
separable positions; `base` sets the longest period (unambiguous range).

## Metrics / experiments (`experiments.py`)

1. **Cropping / round-trip vs length (1..128):** per-character reconstruction accuracy and whole-word
   exact-match. Expect V1 = perfect ≤32 then ∝32/L; Fourier = high, graceful decay, never a hard cliff.
2. **Adversarial collisions:** strings sharing first 32 bytes, differing after (e.g. `a*32+"X"` vs
   `a*32+"Y"`). V1 → *identical* embedding (collision, cosine=1). Fourier → distinct.
3. **Random collision / separability:** min pairwise cosine distance over N random distinct strings of
   a given length, V1 vs Fourier.
4. **Dimensionality vs accuracy:** sweep `k ∈ {8,16,32,64}` (D = 2048..16384); plot decode accuracy vs
   length per `k`, and D table vs V1's 8192. Show `k=32` (equal D) already beats V1's cap.
5. **(Optional) tiny char-LM** per codec on a long-word corpus — only if torch installs; else future work.

**Success:** any length encodes (no crop); Fourier decode ≫ V1 for L>32 at equal/smaller D; adversarial
pairs distinct under Fourier, identical under V1.

## Files
- `codec.py` — both codecs (encode + decode), pure numpy.
- `experiments.py` — runs 1–4, writes PNGs to `plots/`, prints a results table.
- `README.md` — claim, method, numbers, plots, how they prove it.
