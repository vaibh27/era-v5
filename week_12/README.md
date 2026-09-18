# Simulating ZeRO-1, ZeRO-2 and ZeRO-3 on 32 Virtual GPUs

A from-scratch (no DeepSpeed) implementation of ZeRO-1/2/3 sharding, run
across 32 real `torch.distributed` processes standing in for GPUs. The point
was to make the paper's `4 * n_params` redundancy claim, and the
communication cost of removing it, checkable against actual measured
numbers instead of just quoted.

**ZeRO doesn't shrink a model: it decides how many GPUs have to agree on
holding each piece of it, and how often they have to talk to reassemble the
whole thing.** ([temperature2](https://temperature2.com/p/2026-09-05-guide-what-is-zero-sharding/))

## What gets sharded at each stage

ZeRO doesn't change data parallelism itself: every GPU still
forwards/backwards a different slice of the batch. What changes is which
piece of training state a GPU holds between steps:

```text
                    STANDARD DATA PARALLELISM
        ┌─────────────────────────────────────────┐
GPU 0   │  P  (full)  +  G  (full)  +  O  (full)   │
GPU 1   │  P  (full)  +  G  (full)  +  O  (full)   │
  ...   │                    ...                   │
GPU 31  │  P  (full)  +  G  (full)  +  O  (full)   │
        └─────────────────────────────────────────┘

                          ZeRO-1
        ┌─────────────────────────────────────────┐
GPU 0   │  P  (full)  +  G  (full)  +  O_0 (1/32)  │
GPU 1   │  P  (full)  +  G  (full)  +  O_1 (1/32)  │
  ...   │                    ...                   │
GPU 31  │  P  (full)  +  G  (full)  +  O_31(1/32)  │
        └─────────────────────────────────────────┘

                          ZeRO-2
        ┌─────────────────────────────────────────┐
GPU 0   │  P  (full)  +  G_0 (1/32) + O_0 (1/32)   │
GPU 1   │  P  (full)  +  G_1 (1/32) + O_1 (1/32)   │
  ...   │                    ...                   │
GPU 31  │  P  (full)  +  G_31(1/32) + O_31(1/32)   │
        └─────────────────────────────────────────┘

                          ZeRO-3
        ┌─────────────────────────────────────────┐
GPU 0   │  P_0 (1/32) + G_0 (1/32) + O_0 (1/32)    │
GPU 1   │  P_1 (1/32) + G_1 (1/32) + O_1 (1/32)    │
  ...   │                    ...                   │
GPU 31  │  P_31(1/32) + G_31(1/32) + O_31(1/32)    │
        └─────────────────────────────────────────┘
        full params/grads exist only transiently during
        forward/backward, never at rest between steps
```

## The memory math

With Adam in fp32, every parameter carries four buffers of state: the
parameter itself, its gradient, and Adam's `m` (first moment) and `v`
(second moment). Each buffer is 4 bytes, so that's
`n_params + n_params + 2*n_params = 4*n_params` elements, 16 bytes per
parameter, sitting on every single GPU at baseline. Each ZeRO stage shrinks
a different one of those four terms:

```text
Baseline DP:   4n                    (data parallel, everything replicated)
ZeRO-1:        2n + 2n/N             (optimizer sharded, grads/params still full)
ZeRO-2:        n + 3n/N              (gradients sharded too)
ZeRO-3:        4n/N                  (everything sharded, 1/N of baseline)
```

where `n` = parameter count, `N` = number of GPUs. In stage 1, gradients are
still all-reduced exactly like plain DDP, but each GPU only runs the Adam
update on its own 1/N slice of every parameter, using its own slice of `m`
and `v`; the updated slices then get `all_gather`ed back into a full
parameter on every rank. Stage 2 changes how the gradient itself gets
synced: instead of an `all_reduce` that leaves a full averaged gradient on
every GPU, it does a `reduce_scatter`, so each GPU only ever ends up holding
its own 1/N slice of the averaged gradient. Neither of those touches what
the forward pass actually reads, which is why both are cheap. Stage 3 is
where that stops being true: the parameters themselves are only ever stored
as a shard at rest, so before every forward pass the full parameter has to
be rebuilt with `all_gather` from everyone's shard, used, and then dropped
again once its gradient has been captured and reduce-scattered back down.

## Stage by stage

*(Bridge to names you may have seen elsewhere: PyTorch FSDP1's
`sharding_strategy` maps onto these one-for-one: `NO_SHARD` is stage 0,
`SHARD_GRAD_OP` is stage 2, `FULL_SHARD` is stage 3. FSDP has no separate
name for stage 1 alone, because it always shards params and grads together;
ZeRO is the one that exposes optimizer-only sharding as its own stage.)*

### ZeRO-1: optimizer state partitioning

- Shards only optimizer state: Adam's `m` and `v` (in mixed-precision
  setups, also the fp32 master copy of the parameters). Params and
  gradients stay fully replicated on every GPU.
- Gradient sync is unchanged from plain DDP. The only new traffic is an
  `all_gather` after the optimizer step, to rebuild the full parameter from
  everyone's updated shard.
- Total communication volume comes out the same as one ordinary
  `all_reduce`, so there's effectively no communication overhead over
  baseline DP.
- Optimizer state is bookkeeping nothing else in forward/backward ever
  reads, so sharding it doesn't force any extra mid-step synchronization.
- Params and grads stay full-size regardless of GPU count, so total memory
  has a floor: it shrinks toward that floor as `N` grows, never below it.
- Here: each rank runs Adam on just its 1/N parameter slice using its own
  `m`/`v` shard, then `all_gather`s the updated slices back into a full
  tensor (`zero_optimizer.py`, stage 1).

### ZeRO-2: + gradient partitioning

- Adds gradient sharding on top of ZeRO-1: gradients are `reduce_scatter`ed
  instead of `all_reduce`d, so each GPU only ever holds its own 1/N slice
  of the averaged gradient.
- Total communication volume is unchanged from stage 1: reduce-scatter +
  all-gather is the same total traffic as one all-reduce, just split into
  two operations instead of one.
- Removes the second full-size buffer (gradients) from memory, on top of
  the optimizer-state savings already made in stage 1.
- A gradient shard can be freed as soon as its own reduce-scatter finishes,
  rather than waiting for the whole backward pass, which plays well with
  gradient accumulation done locally before syncing.
- Still has a floor: parameters stay fully replicated, so returns diminish
  as GPU count grows, the same shape of limitation as stage 1.
- Here: `sync_grads()` reduce-scatters each parameter tensor's gradient and
  keeps only the local shard; params are still gathered back to full after
  every step's update.

### ZeRO-3: + parameter partitioning

- Shards everything: parameters, gradients, and optimizer state. No GPU
  holds a complete parameter at rest, ever.
- Before a parameter is used, it's rebuilt with `all_gather` from every
  rank's shard; once its gradient is captured, that gradient is
  `reduce_scatter`ed straight back down to a shard and the full tensor is
  dropped again.
- The only stage where memory keeps scaling down linearly with `N`, with no
  floor, which is what makes training models bigger than any single GPU
  possible at all.
- Communication cost: per the ZeRO paper's own accounting, baseline
  DP/ZeRO-1/ZeRO-2 all move `2Ψ` of traffic per step (an `all_reduce` costs
  the same total traffic as a `reduce_scatter` + `all_gather` pair);
  ZeRO-3 moves `3Ψ`, from the extra `all_gather` needed to rebuild
  parameters before the forward pass. `3Ψ` against a `2Ψ` baseline is a
  1.5x increase, worth naming precisely, since "3x baseline" gets thrown
  around informally and overstates it. Real implementations lean on
  prefetching the next layer's shard while still computing the current one
  to hide that overhead behind compute.
- Needs predictable, sequential execution to prefetch well. Architectures
  with dynamic routing (MoE) or slow interconnects don't get that overlap
  for free, and can end up compute-idle waiting on the network.
- Here: `gather_full_params()` gathers the entire model once per step
  rather than layer by layer; see "Implementation notes" below for why.
  At-rest memory is exact; peak memory during the step is higher than a
  real per-layer implementation would give.

## The demo model

Small decoder-only transformer: token + positional embeddings, 4
pre-norm transformer blocks (`nn.MultiheadAttention` + GELU MLP), LM head.
**4,191,744 parameters**.

```text
Token ids → Embedding + Position → [Transformer Block × 4] → LN → Linear head
```

## The 32 virtual GPUs

Each "GPU" is a real OS process (`torch.multiprocessing.spawn`, `nprocs=32`),
joined into one `gloo` process group over loopback TCP. `all_reduce`,
`all_gather`, and `reduce_scatter` are real `torch.distributed` collectives
moving real data between 32 independent processes, not mocked accounting.
`OMP_NUM_THREADS=1` prevents the 32 ranks from fighting over 8 physical cores.

Simulating the *rules* of ZeRO (what each rank owns, what collectives move
what), not the *silicon*, is what makes the memory numbers meaningful.

## Results

32 virtual GPUs, 4,191,744 parameters, 2 steps per stage. Numbers below are
copied directly from the notebook's own run, not recomputed separately.
Memory is theoretical, computed directly from shard sizes; step time is the
mean wall-clock per step across all 32 ranks.

| Stage | Params (MB) | Grads (MB) | Optim (MB) | Total/GPU (MB) | Step time (s) |
|---|---|---|---|---|---|
| Baseline DP | 16.77 | 16.77 | 33.53 | **67.07** | 12.63 |
| ZeRO-1 | 16.77 | 16.77 | 1.05 | **34.58** | 16.83 |
| ZeRO-2 | 16.77 | 0.52 | 1.05 | **18.34** | 145.27 |
| ZeRO-3 | 0.52 | 0.52 | 1.05 | **2.10** | 154.17 |

Memory drops exactly as the formula predicts: ZeRO-3 lands at `4n/N`, an
exact **32x reduction** from baseline, matching `N` on the nose.

Step time doesn't move smoothly. Stage 1 is only modestly slower than
baseline (same `all_reduce` traffic as DDP, plus a small `all_gather`),
while stages 2/3 jump to roughly **9-12x** stage 1's time. That jump tracks
the number of collective calls, not the number of bytes moved: this
implementation issues one `reduce_scatter`/`all_gather` per parameter
tensor (~30 calls) instead of bucketing into a few large flat buffers, and
gloo/TCP charges fixed per-call latency regardless of how much data rides
along with it. Production ZeRO (DeepSpeed, FSDP) bucket parameters
specifically to amortize this; the naive version here leaves that cost
exposed instead of hiding it.

### Picking a stage in practice

Which stage to reach for is a memory-*pressure* question, not a model-*size*
one. DeepSpeed's own field guidance is illustrative: pair a 1.5B model on 8
GPUs with stage 1, a 10B model on 32 GPUs with stage 2, and reserve stage 3
for models that genuinely don't fit even fully sharded: "the decision is how
much memory pressure the job actually has, not how large the model sounds."
ZeRO-1 is close to free here (barely more communication than plain DDP) and
already removes the single largest memory hog (`2n` of Adam state); stage
2/3's extra communication is only worth paying once memory, not stage
number, is the actual constraint.

## Implementation notes

**ZeRO-1 leaves the model itself untouched.** Params and grads are fully
replicated through ZeRO-2; only ZeRO-3 drops a full parameter copy from
every GPU. Sharding Adam's `m`/`v` costs almost nothing because nothing
outside the optimizer step ever reads that state.

**The stage 2/3 slowdown tracks call count, not bytes moved.** A sharded
gradient and a full gradient of the same tensor carry the same total
elements. The difference is ~30 separate collective calls per step instead
of 1, each paying fixed gloo/TCP latency, which is exactly the cost
bucketing exists to amortize away in production systems.

**Why stage 3 gathers the whole model per step instead of per layer.** The
"real" FSDP approach is a `forward_pre_hook` on each submodule that gathers
just that submodule's parameters right before it runs. That doesn't work
for `nn.MultiheadAttention`: its `forward()` reads `self.out_proj.weight`
and `self.out_proj.bias` as plain attributes and passes them straight into
a functional call; it never calls `self.out_proj(...)`, so a hook attached
to that submodule's `__call__` never fires. (Checked directly against
`torch.nn.MultiheadAttention.forward`'s source: this isn't a guess.)
Gathering every parameter once per step sidesteps the issue: at-rest
memory, the number in the table above, is still exact, but peak memory
during the step is higher than a real per-layer implementation would give.

## Limitations

CPU/gloo, not GPU/NCCL: absolute times aren't representative, but relative
trends are. fp32 throughout (no mixed precision). Real ZeRO's
mixed-precision accounting is `16 * n_params` bytes (2B fp16 params + 2B
fp16 grads + 4B fp32 master-weight copy + 4B+4B fp32 Adam `m`/`v`), not the
`4 * n_params` fp32-only number used throughout this project; fp32 keeps the
arithmetic legible at the cost of not showing that particular real-world
detail. Stage 3 gathers all params at once, not per-layer. No bucketing, no
compute/comm overlap. Both are deliberately omitted to keep the cost
visible.

## Further reading

Related material, cross-checked against the claims made above:

- [DeepSpeed's ZeRO-Offload tutorial](https://www.deepspeed.ai/tutorials/zero-offload/):
  what's beyond stage 3. Once params are already fully sharded across
  every GPU and still don't fit, ZeRO-Offload/ZeRO-Infinity push optimizer
  state (and even params) out to CPU RAM or NVMe instead of GPU memory,
  trading further memory for bandwidth instead of for parallelism.
- [Anyscale: FSDP, PyTorch, and DeepSpeed for large-scale training](https://www.anyscale.com/blog/fsdp-pytorch-deepspeed-ray-large-scale-distributed-training):
  a clear side-by-side of DeepSpeed's config-driven stage numbers against
  PyTorch FSDP's `sharding_strategy` API for the same underlying idea.
- [Lorenzo Cesconetto's ZeRO writeup](https://lorenzocesconetto.github.io/posts/2025-10-25-ZeRO/):
  another from-first-principles derivation of the memory math, useful as
  an independent check against the formulas above.
- [temperature2: What is FSDP?](https://temperature2.com/p/2026-09-04-guide-what-is-fsdp/)
  and [What is ZeRO sharding?](https://temperature2.com/p/2026-09-05-guide-what-is-zero-sharding/):
  source of the `2Ψ` vs `3Ψ` communication-volume accounting used above,
  plus the "match the stage to memory pressure, not model size" heuristic.

## Setup

```bash
uv sync
uv run jupyter notebook ZeRO_Simulation.ipynb
```

## Files

Everything below is what `ZeRO_Simulation.ipynb` actually imports to run;
there's no separate script layer to keep in sync with it.

| File | Purpose |
|---|---|
| `model.py` | Demo transformer + synthetic data |
| `shard_utils.py` | flatten/pad/shard + all_gather/reduce_scatter |
| `zero_optimizer.py` | Stage 0/1/2/3 sharding logic, hand-rolled Adam |
| `train.py` | Spawns 32 processes, runs training loop |
