# proxy/v2 — nano-RegMix, multilingual + multi-seed

Honest scope: **7.5M-param rank signal, proof-of-method — not evidence for any 40B number.**

- **6 lanes:** web · code · math · reasoning · Indic-native · Indic-translated
  (agentic/science/india-first fold into others — too noisy to be evidence at nano scale).
- **Multilingual Indic:** native + translated over **{Hindi, Bengali, Tamil}**
  (Indo-Aryan/Devanagari, Indo-Aryan/Bengali, Dravidian/Tamil); per-language loss reported
  as a diagnostic so cross-script generalization is measured.
- **Multi-seed priors:** H_A and H_B each run at **3 seeds** → composite mean ± range and a
  native-collapse frequency (H_A = 22% Indic, H_B = 13% Indic).
- **24K byte-level BPE** so Bengali/Tamil scripts get real coverage.

## Design
- **Model:** decoder d=192, 6 layers, 6 heads, ctx 512, tied embeddings, 24K BPE (~7.5M params)
- **Runs:** H_A×3 + H_B×3 + 18 Dirichlet (prior-centered, min lane 2%) + 2 confirmations,
  **10M tokens each** (v1-parity; ~7.5 min/run on MPS, run **serially** — 3 parallel
  workers OOM a 16 GB Mac).
- **Composite C** (README §9.1, renormalized to 6 lanes): native .233 / translated .100 /
  code .278 / math .133 / reasoning .089 / web .167, with the **90% lane-collapse guard**.
- **Eval:** per-lane held-out CE (fixed seed) + per-language native/translated diagnostic.
- **Fit:** ridge per-lane loss ← log(mixture); argmax C over a 400K-point simplex grid,
  clamped to the sampled hull (extrapolated corner recorded separately + distrusted).

## Data sources (streamed once → `data/`)
| Lane | Source |
|---|---|
| web | HuggingFaceFW/fineweb-edu (sample-10BT) — reused from v1 |
| code | bigcode/the-stack-smol — reused from v1 |
| math | open-web-math/open-web-math — reused from v1 |
| reasoning | open-thoughts/OpenThoughts-114k (`conversations` join; fallback OpenR1-Math) |
| indic_native | ai4bharat/sangraha `verified/{hin,ben,tam}` |
| indic_translated | ai4bharat/sangraha `synthetic/{hin_Deva,ben_Beng,tam_Taml}` |

## Run order
```bash
PY=../.venv/bin/python
$PY nanoproxy.py prep        # stream reasoning + 6 indic shards (web/code/math reused)
$PY nanoproxy.py tokenizer   # 24K BPE over a balanced 6-lane sample
$PY nanoproxy.py tokenize    # -> data/<shard>.npy + data/manifest.json
$PY nanoproxy.py train --name smoke --tokens 500000   # smoke test
$PY nanoproxy.py sweep --n 18 --tokens 10000000       # SERIAL — do not shard-parallel on 16 GB
$PY nanoproxy.py fit         # -> runs/fit_report.json
# clean Arm E (reasoning 0/4/8%; r4 == H_A, so only two extra runs):
$PY nanoproxy.py train --name arm_e_r0 --tokens 10000000 --mixture '{"web":0.54,"code":0.18,"math":0.06,"reasoning":0.0,"indic_native":0.08,"indic_translated":0.14}'
$PY nanoproxy.py train --name arm_e_r8 --tokens 10000000 --mixture '{"web":0.46,"code":0.18,"math":0.06,"reasoning":0.08,"indic_native":0.08,"indic_translated":0.14}'
```

## Outputs
- `runs/<name>.json` — per run: mixture, per-lane losses, `lang_diag`, tokens, wallclock
- `runs/fit_report.json` — H_A vs H_B (mean ± range, collapse frequency), per-lane ranking,
  in-hull vs distrusted-extrapolated optimum
- Findings are injected into `../../README.md` §9.2 via `INJECTION.md`.
