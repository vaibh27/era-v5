# Problem 2 — Multimodal Kronecker (text + image + audio)

> _"What is the natural extension of Kronecker, such that it can represent images and audio as
> well!! Yes we'll need to do some preprocessing of image and audio patches, but how do we use
> this concept to represent all 3!"_

## Restatement

Generalize the codec `κ = Σ (content ⊗ position)` from bytes-in-a-word to **atoms-in-a-signal**,
so the *same* deterministic Kronecker construction produces embeddings for image patches and
audio frames as it does for text — then project each modality into one shared `d_model` space.

## The unifying abstraction

V1 text is a special case of a general template:

```
κ(x) = (1/√N) · Σ_{i}  content(x_i) ⊗ position(i)
```

| Modality | `content(x_i)` (one-hot / quantized) | `position(i)` | Notes |
|----------|--------------------------------------|---------------|-------|
| **Text**  | byte value (256) | token position (1-D, ≤32) | this is V1 |
| **Image** | patch codeword (VQ codebook, e.g. 512–8192) | **2-D** `row ⊗ col` | patchify like ViT, then quantize |
| **Audio** | spectral/quantized frame codeword | time index (1-D) | frame = STFT/mel window or a neural codec token |

So the "natural extension" is: **(a)** replace the byte alphabet with a modality-specific
**discrete codebook** for patches/frames, and **(b)** replace the 1-D position one-hot with the
natural index structure of the modality (2-D for images → an extra Kronecker factor).

```
Image patch at (r,c):  κ += codeword(patch) ⊗ row_r ⊗ col_c
Audio frame at t:      κ += codeword(frame) ⊗ time_t
```

Each modality gets its own fixed codec dimension `D_m` and its own projection `W_proj^m : R^{D_m} → R^{d_model}`, landing all three in one shared latent so cross-modal attention works.

## Preprocessing (the "some preprocessing" the assignment expects)

- **Image:** resize → split into P×P patches → quantize each patch to a codebook index.
  Codebook options: a fixed VQ/k-means codebook (keeps the *deterministic* spirit of Kronecker),
  or a small pretrained VQ-VAE. Deterministic k-means preserves "same input → same embedding".
- **Audio:** STFT / mel-spectrogram → frame windows → quantize each frame (or use a neural audio
  codec's discrete tokens). Time index → position factor.

## Research directions

1. **Fixed vs. learned codebook.** A deterministic k-means/VQ codebook keeps the defining
   Kronecker property (determinism, no per-item lookup table). Measure the quality cost vs. a
   learned VQ-VAE codebook.
2. **How many Kronecker factors?** Images invite `content ⊗ row ⊗ col`; test whether the 2-D
   factorization beats flattening patches to a 1-D sequence (does explicit 2-D locality help?).
3. **Shared vs. per-modality projection.** One `W_proj` for all vs. per-modality. Per-modality is
   the natural choice since `D_m` differ; test whether a shared codebook prefix aids alignment.
4. **Cross-modal locality.** Verify the locality property carries over: nearby patches / adjacent
   frames get similar κ, analogous to shared-bytes-same-position in text.

## How to prove it

- **Dataset:** a tiny paired/aligned set — e.g. MNIST or CIFAR patches (image), spoken-digit
  audio (FSDD) (audio), digit words (text). Small enough for minutes-long training.
- **Tasks / metrics:**
  1. **Classification** on each modality with Kronecker patch/frame embeddings vs. a learned
     patch/frame embedding baseline — measure accuracy and **trainable-parameter savings** (the
     whole point of Kronecker: the codec is fixed, only `W_proj` trains).
  2. **Locality probe** — patch-similarity vs. pixel-space similarity correlation.
  3. **(Stretch) cross-modal retrieval** — do image `7`, audio "seven", text `7` land near each
     other after a light contrastive objective? Evidence the shared space works.
- **Success criterion:** competitive accuracy with far fewer input-side trainable params, and a
  clean locality/retrieval signal.

## Open questions / risks

- Quantization is lossy — codebook size trades reconstruction vs. `D_m` blow-up (`content ⊗ row ⊗
  col` grows fast). Watch `D_m`.
- Determinism vs. quality: a fixed codebook is "purely Kronecker" but a learned one may be needed
  for real images. State the trade-off explicitly rather than picking silently.
- 2-D position Kronecker can explode dimensionality; a low-rank / Fourier position basis
  (see [Problem 4](../track4/README.md)) keeps it tractable.

## Related

Same "append/define a structured content⊗position subspace" pattern as
[Problem 1](../track1/README.md); a Fourier position basis
([Problem 4](../track4/README.md)) tames the 2-D dimensionality.
