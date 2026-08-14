# Kronecker Embedding V2 — Final Submission

Explorations of the Kronecker Embedding V2 problems — what was **tried** and **measured**, not a claim
that any problem is solved. Each track states **which** problem it explores and **what was measured**
(runnable code + measured results), and names exactly where the approach breaks.

| # | Problem | Status | Run |
|---|---------|--------|-----|
| **1** | Store math structure (`9+9→18`, `9*9→81`) + words in one model | **Explored — prototype + evidence** | `python3 track1/code/kronecker_v2_embedding.py` |
| 2 | Multimodal (text + image + audio) | Design only | `track2/README.md` |
| **3** | Dynamic length (kill the 32-position cap) | **Explored — prototype + evidence** | `cd track3 && python3 experiments.py` |
| **4** | Fourier alternative (each char a wave) | **Explored — prototype + evidence** | `cd track4 && python3 experiments.py` |
| **5** | Reversible embeddings (drop the head, 1M vocab) | **Explored — prototype + evidence** | `cd track5 && python3 run.py` |

## Problem 1 — `track1/`
A **proper embedding model that handles both words and numbers**: `[ Kronecker text ‖ CRT math ]`.
Words get the deterministic byte×position code (the "32 existing spaces"); numbers additionally fill an
appended **CRT math register** where a fixed operator does exact `+`/`×` (`9+9→18`, `9*9→81`) and decodes
back. Trained-model evidence (`track1/code/arithmetic_embedding/`) shows the structure survives learning
(exact 1.0 vs. monolithic 21%). Details, limits, and the maths to push further: `track1/README.md`.

## Problem 3 — `track3/`
Replace V1's one-hot position (which *is* the 32-cap) with a **Fourier position basis**: dimension
independent of length, any position encodes (no crop), exact round-trip for `L≤k`. **0% collisions vs
V1's 100%** on shared-prefix strings; 45/65-byte words recovered.

## Problem 4 — `track4/`
Replace Kronecker's **product** with a **superposition**: each character is a wave, each position a
rotation, a word is the **sum**. No 32-cap; **anagrams stay distinct** (phase carries order); **0%
collisions vs V1's 100%**.

## Problem 5 — `track5/`
Replace the `V×d_model` softmax head with a **byte-grid head** (predict per-position bytes, invert κ):
**open-vocab copy 0.886 vs softmax 0.000**, κ round-trip 2500/2500, output head **265K flat vs 129M** at
1M vocab.

## Problem 2 — `track2/` (not implemented)
Design for extending `content⊗position` to image/audio patches (VQ codebook + 2-D/time position). Doc only.

## Requirements
- Problems 1, 3, 4: `numpy`, `matplotlib` (Problem 1's model is pure numpy).
- Problem 5 and Problem 1's trained-model evidence: `torch` (`pip install -r track1/code/arithmetic_embedding/requirements.txt`).
