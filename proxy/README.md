# proxy/ — the nano-RegMix run (v1, SUPERSEDED)

> **Superseded by [`v2/`](v2/README.md)** — Hindi-only, single-seed, 5 lanes. Kept for
> provenance; the current run is `v2/` (multilingual hi/bn/ta, 3-seed priors, reasoning
> unfolded). Numbers here are frozen as-run; the ran token budget was **10M/run** (an
> earlier draft of this file said 30M).

Implements `../wiki/60-proxy-experiments.md` §3 (RegMix-in-miniature, Mac/MPS).
**Honest scope:** proof-of-method; rank signals only — not evidence for 40B numbers.

## Design (mirrors the wiki page)
- **5 collapsed lanes:** web-edu · code · math(+reasoning) · Indic-native (hi) · Indic-translated (hi)
- **Model:** ~6M-param decoder (d=192, 6 layers, 6 heads, ctx 512), 16K byte-level BPE trained on the proxy corpus
- **Runs:** H_A + H_B + 18 Dirichlet-sampled mixtures (prior-centered, min lane weight 2%) × 30M tokens each
- **Eval:** per-lane held-out CE loss (fixed seed → identical eval set for every run)
- **Composite C** (wiki/60 §1, renormalized to 5 lanes): native .233 / translated .100 / code .278 / math .222 / web .167, with the **90% lane-collapse guard**
- **Fit:** ridge per-lane loss ← log(mixture), predict C over a 200K-point simplex grid, argmax under the guard → confirmation run

## Data sources (streamed once, cached in `data/`)
| Lane | Source |
|---|---|
| web | HuggingFaceFW/fineweb-edu (sample-10BT) |
| code | bigcode/the-stack-smol (fallback: codeparrot-clean) |
| math | open-web-math/open-web-math |
| indic_native | ai4bharat/sangraha `verified/hin` |
| indic_translated | ai4bharat/sangraha `synthetic/hin_Deva` (fallbacks in code) |

## Run order
```bash
uv venv .venv && uv pip install --python .venv/bin/python torch numpy datasets tokenizers scikit-learn
.venv/bin/python nanoproxy.py prep        # stream + cache ~1GB text
.venv/bin/python nanoproxy.py tokenizer   # train 16K BPE
.venv/bin/python nanoproxy.py tokenize    # -> data/<lane>.npy
.venv/bin/python nanoproxy.py train --name smoke --tokens 2000000   # smoke test
# parallel sweep: 3 workers sharing the MPS GPU (run list is deterministic; shards don't overlap;
# each run writes its own runs/<name>.json so workers never contend)
.venv/bin/python nanoproxy.py sweep --n 18 --tokens 30000000 --shard 0/3 &
.venv/bin/python nanoproxy.py sweep --n 18 --tokens 30000000 --shard 1/3 &
.venv/bin/python nanoproxy.py sweep --n 18 --tokens 30000000 --shard 2/3 &
.venv/bin/python nanoproxy.py fit         # after all workers finish -> runs/fit_report.json
```

## Outputs
- `runs/results.jsonl` — one row per run: mixture, per-lane losses, tokens, wallclock
- `runs/fit_report.json` — top-5 observed, H_A vs H_B composite, predicted-best mixture
- Results are filed back into `../wiki/60-proxy-experiments.md` and the final README.
