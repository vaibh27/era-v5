# Week 10

A small GPT-style transformer (128 embd, 4 heads, 4 layers, about 13.7M params) trained on TinyStories. The notebook covers seven tasks about the internals of a training step, with results taken directly from the executed output cells.

## Files

- `week10_assignment.ipynb`: the notebook, all seven tasks below with executed outputs.
- `week10_internals.py`: model code (`TinyModel`, `OutputHead`) and data pipeline (`get_dataset`, `get_batch`) for TinyStories.

## Setup

```bash
pip install torch transformers datasets matplotlib numpy
```

The notebook loads `roneneldan/TinyStories` (20,000 rows) and the GPT-2 tokenizer, then caches the processed dataset to `/tmp/tinystories_processed`. Device selection (CUDA, Apple MPS, or CPU) is automatic. The numbers in this README are from Apple M1 Pro (MPS).

## Task 1: tensor shapes

Forward pass and loss computation, batch=8, seq_len=128, vocab=50257:

| tensor | shape | meaning |
|---|---|---|
| `tokens` | (8, 128) | [batch, position], raw input token ids |
| `targets` | (8, 128) | [batch, position], next-token labels, `targets[b,i] = tokens[b,i+1]` |
| `pad_mask` | (8, 128) | [batch, position], True where position is a real (non-pad) token |
| `hidden` | (8, 128, 128) | [batch, position, n_embd], model's internal representation per position |
| `logits` | (8, 128, 50257) | [batch, position, vocab_size], unnormalized score per candidate next token |
| `flat_logits` | (1024, 50257) | [batch*position, vocab_size], batch and position collapsed for `cross_entropy` |
| `flat_targets` | (1024,) | [batch*position], one target id per row of `flat_logits` |
| `loss_per_token` | (1024,) | [batch*position], per-token loss before masking |
| `loss` | scalar | pad-masked mean loss over real (non-pad) tokens only |

## Task 2: gradient by hand

Forward and backward pass computed in float64, eval mode (dropout disabled). `param.grad[0,0]` of the output head weight compared against a central-difference numerical gradient at four epsilon values:

```
analytic grad (backward()):  0.01611399178392415
eps=1e-3   numeric=0.0161139918   rel diff=1.3e-10
eps=1e-4   numeric=0.0161139918   rel diff=6.4e-10
eps=1e-5   numeric=0.0161139918   rel diff=2.8e-09
eps=1e-6   numeric=0.0161139928   rel diff=6.3e-08
```

Agreement to about 9-10 significant digits at eps=1e-4. Error increases as eps drops below 1e-3, consistent with rounding and cancellation error dominating over truncation error at that scale in float64.

## Task 3: broken gradient accumulation

Two models with identical initialization, trained for 100 steps on the same superbatches. Each superbatch has four micro-batches of different lengths (16, 32, 64, 128 tokens, 8 rows each).

- Buggy: loss averaged as `loss / k` across micro-batches. This weights each micro-batch equally (25% each) regardless of its token count.
- Correct: token-weighted accumulation, `loss_sum / total_tokens`. This weights each token equally (6.7%, 13.3%, 26.7%, 53.3% by micro-batch).

Both curves are plotted against the same token-weighted loss metric:

```
final buggy loss:   7.4347
final correct loss: 5.5958
gap at final step:  1.8389 nats
```

The equal-weighting scheme under-weights the longer, more token-dense micro-batches throughout training, which produces the gap above.

## Task 4: grad norm relative to loss

Grad norm (pre-clip) recorded at every step across 300 steps. A z-score threshold on the step-to-step change in grad norm flags steps where the grad norm jumps; the loss change in the steps before and after each flagged step is then compared.

Largest candidate: step 270 (z=4.82). Loss changed -0.022 over the 3 steps before the grad norm jump, and -0.196 over the 2 steps after. This is correlational evidence that the grad norm moved before the loss at this step, not proof of causation. Batch-to-batch variance can also produce grad norm spikes, and the lead time is not consistent across all flagged candidates.

## Task 5: MFU

FLOPs per token: `6*N_matmul + 12*L*H*Q*T` (embedding parameters excluded, since embedding lookup is a gather, not a matmul).

```
N_total=13,673,984  N_matmul=7,224,704
flops_per_token = 44,134,656
16,368 tokens/sec (batch=16, seq_len=128, M1 Pro/MPS)
achieved = 0.722 TFLOPS
peak (MPS, fp32 estimate) = 4.55 TFLOPS

MFU = 15.88%
```

Factors limiting MFU, in order of impact:
1. Model size: the weights fit in cache, so step time is limited by kernel launch and dispatch overhead rather than GPU math throughput.
2. No kernel fusion: LayerNorm, GELU, dropout, and residual adds each run as a separate kernel launch, which is a larger fraction of step time at this model size on MPS.
3. MPS has no tensor cores and a less optimized GEMM and attention path than CUDA.
4. Batch size (16 x 128 = 2048 tokens/step) is small relative to kernel launch overhead.
5. Full fp32 precision throughout, no mixed precision.

15.88% MFU reflects this model size and hardware combination. Reaching 40% would need a larger model, a CUDA GPU with tensor cores, and fused kernels.

## Task 6: 0.1 in fp32, bf16, fp8 E4M3

0.1 = 1.10011001100110011001100110011...(base 2) x 2^-4 (repeating pattern), rounded to each format's mantissa width. Bit patterns cross-checked against `struct.pack` and `torch.bfloat16` output:

| format | bits | hex | stored value | rel. error |
|---|---|---|---|---|
| fp32 (1 sign, 8 exp, 23 mantissa) | `00111101110011001100110011001101` | `0x3dcccccd` | 0.10000000149 | 0.000001% |
| bf16 (1 sign, 8 exp, 7 mantissa) | `0011110111001101` | `0x3dcd` | 0.10009765625 | 0.097656% |
| fp8 E4M3 (1 sign, 4 exp, 3 mantissa) | `00011101` | `0x1d` | 0.1015625 | 1.562500% |

Recommendation: bf16 for weights, activations, and gradients, with fp32 optimizer state (standard mixed-precision setup). Task 2 shows the gradient signal holds to about 1e-9 relative error, while bf16 gives about 1e-3 relative precision, above that noise floor and sufficient for stable convergence when optimizer state is kept in fp32. fp8 E4M3 gives about 1e-1 relative precision, below what general-purpose training needs without extra measures such as loss scaling. fp32 works but uses twice the memory traffic of bf16 for no convergence benefit here.
