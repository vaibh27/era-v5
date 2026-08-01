# V5: Mixture & Curriculum Specification

**A ~40B, India-first, dense model trained from scratch via progressive growth (1B→3B→8B→20B→40B) on a 15T-token budget.** This document fixes *how much* of each kind of data the model sees (mixture) and *when* it sees it (curriculum), across pre-training, SFT, and RL. Every share is defended against a benchmark and against the real supply behind it, with the gap between the two stated plainly rather than hidden.

> Every share here is a **hypothesis** grounded in evidence, not a guess. The mixture below applies the direction from a **nano-scale proxy** (§9.2) — the largest experiment this project's compute (a single Mac) allows; the 1B/3B runs that would normally validate a mixture (§9.1) are out of reach here. That is a real limit, but a bounded one: what a proxy this small can report is a *direction* (which lane is over- or under-funded), not an absolute 40B loss — and mixture *ranking* is largely scale-invariant, which is the premise the RegMix and data-mixing-law results rest on ([RegMix](https://arxiv.org/abs/2407.01492)). So the direction below is expected to transfer; the exact shares are not claimed to. This README is the standalone spine; fuller derivations exist as a separate evidence trail.

---

## 0. Method

The mixture is **not** a survey of available data. It is a reverse projection of the capabilities the model must win, ranked, each tied to the benchmark that measures it:

| Rank | Capability | Benchmark it must win |
|---|---|---|
| 1 | **Indic** (identity) | MILU, IndicGenBench, IndicXTREME |
| 2 | **Code** | HumanEval+, MBPP+, LiveCodeBench |
| 3 | **Math + reasoning** | GSM8K, MATH, MathArena, MMLU-Pro |
| 4 | **Agentic / tool-use** | BFCL v4, τ-bench |
| 5 | **Long-context** | RULER, needle-in-haystack, long-doc QA |
| 6 | **English world-knowledge** | MMLU, ARC, HellaSwag |
| 7 | **India-first framing** | bespoke cultural-correctness eval |
| 8 | **Science** | GPQA, MMLU-STEM |

Six steps: rank capabilities → set a defended share per lane → convert share to tokens-needed → posit real *unique* supply → compute the gap (repetition epochs and/or synthesis %) → read off the starved slots.

**The anti-wishful-accounting rule:** every share that exceeds its plausible real unique supply must say so and name its filler: epochs or synthesis. A lane handed a large share with almost no real data behind it is a defect, not a decision. So, for example, ~55% of the Indic lane is called out as hard-gated translation rather than hidden inside the 22% headline.

---

## 1. The pre-training mixture

**H_A and H_B** name the two candidate Indic-lane sizes the spec deliberately leaves open. **H_A** ("Hypothesis A") is this table's headline bet: **22% Indic**. **H_B** is a leaner rival: **13% Indic**, saving scarce native tokens for the anneal (§5). They differ *only* in the Indic share (the rest goes to web). The nano-proxy (§9.2) tests these aggregate ratios; it supports 22% > 13% and flagged web as over-funded relative to code/reasoning — which this table now reflects (web trimmed into code/reasoning, §9.2) — but **only Arm B at 3B settles the 22-vs-13 fork.**

Shares are **scale-transferable ratios**, not token counts defended to the digit. RegMix's finding is that mixture *ranking* is largely rank-invariant across scale, so the ratios are set once and proven cheaply ([RegMix](https://arxiv.org/abs/2407.01492)). Shares are of **the target tokenizer's tokens**; at Indic fertility ~1.6–1.8 vs English ~1.2, a 22% Indic *token* share delivers only ~⅔ the *content* share (stated, not hidden).

| Lane | % | Tokens | Benchmark | Honest filler (real unique supply) |
|---|---|---|---|---|
| Web (general) | 39% | 5.85T | MMLU, ARC, HellaSwag | **~1 epoch** (5.85T of 6.3T+ unique) — the only lane with no repetition or synthesis |
| _↳ edu-classified subset_ | _⅓ of web_ | _~1.95T_ | MMLU, ARC | _a slice of the 39% above (not extra): the FineWeb-Edu-style high-quality portion that carries the knowledge load_ |
| **Indic** | 22% | 3.3T | MILU, IndicGenBench | **~95% translated/synthetic**; verified-native ≈64B (§3) |
| Code | 20% | 3.0T | HumanEval+, LiveCodeBench | **~3.3 epochs** of 900B unique (StarCoder2 itself ran 3.7–4.8) |
| Math | 6% | 0.9T | GSM8K, MATH, MathArena | **~2–2.5 epochs** of ~350–450B (MegaMath ∪ Nemotron-CC-Math) |
| **Reasoning traces** *(new)* | 6% | 0.9T | MMLU-Pro, BBH → GSM8K/MATH | **~93% distilled**; open long-CoT pool is single-digit-to-tens of B |
| Science | 2% | 0.3T | GPQA, MMLU-STEM | **~2–3 epochs** of peS2o 57B + synthesis on the remainder |
| India-first curated | 3% | 0.45T | bespoke cultural eval | **mostly synthetic** exam/civic + 100% of a tens-of-B real scrape |
| Agentic | 2% | 0.3T | BFCL v4, τ-bench | **~fully synthetic**, verifier-gated (real supply ~hundreds of M) |
| **Total** | **100%** | **15T** | | |

Long-context is deliberately **not a lane**: it is a late re-pack of already-counted tokens (§5), and a separate lane would double-count.

**What changed vs the week-3 prior, and why:** code 17→20 (R2's 15–25% band, upper-middle after the nano flagged code under-funded, §9.2); science 4→2 (real supply is only 57B; the old 4% was undeclared wishful accounting); India-first 5→3 (scrape ceiling is tens of B, not 0.75T); reasoning added at 6% (NVIDIA front-loading — pre-training needs a *broad* reasoning prior, unrecoverable if deferred — sized up after nano Arm E); web 44→39 (trimmed into code/reasoning per the nano, which flagged it over-funded).

---

## 2. The supply ledger

| Lane | Needed | Real unique supply | Epochs | Synthesis (named) | Verdict |
|---|---|---|---|---|---|
| Web | 5.85T | 6.3T+ (Nemotron-CC + DCLM + FineWeb-Edu) | ~0.9× | 0% new | **healthy** |
| **Indic** | 3.3T | ~90–150B native + 162.7B Sangraha-translated | native ≤4× | ~95% (IndicTrans2 + LLM) | **STARVED (native)** |
| Code | 3.0T | ~900B (Stack v2) | ~3.3× | 0% (StarCoder2 precedent) | repetition-lane |
| Math | 0.9T | ~350–450B | ~2.2× | already-synth inside MegaMath | repetition-lane |
| **Reasoning** | 0.9T | ~10–30B | >4× impossible | ~93% R1-distill | **STARVED** |
| **Science** | 0.3T | 57B (peS2o) | ~2–3× | ~45–60% textbook-gen | **STARVED** |
| **India-first** | 0.45T | tens of B (legal ~7B, Varta ~9B) | scrape 1× | ~90% exam/civic-gen | **STARVED** |
| **Agentic** | 0.3T | hundreds of M | negligible | ~99% APIGen-MT-gen | **STARVED** |

**Five starved slots** (Indic-native, reasoning, science, India-first, agentic) are where cleaning + generation effort is aimed (§8). Web is the only lane seen ~once (no repetition or synthesis). Code and Math are honest repetition lanes with published precedent, needing an epoch count, not a synthesis flag.

---

## 3. The Indic split (five tiers)

The 22% headline hides a **UniMax epoch cap**: native tokens repeat ≤~4× before overfitting, so the native floor caps out far below 3.3T and the rest is *openly* generated. The lane splits into five tiers, each with its gate (sums to 3.3T):

| Tier | Real seed | Effective | % of lane | Gate |
|---|---|---|---|---|
| **1 · verified-native** | 64.3B (Sangraha-V) | 0.26T | 7.9% | human-verified; ≤4 epochs total, ~3 held for the anneal (§6) |
| **2 · unverified-native** | 24.3B (+ ≤26B top-up, *uncertain*) | 0.20T | 6.1% | perplexity-filtered |
| **3 · translated** | 162.7B Sangraha + new IndicTrans2 | 1.80T | 54.5% | **hard-gated** COMET/LaBSE + round-trip; bad MT dropped, not trained |
| **4 · romanized** | ~1B real (L3Cube) | 0.26T | 7.9% | IndicXlit transliteration; script-validity + CMI gate |
| **5 · LLM-synthetic** | 0 (generated) | 0.78T | 23.6% | verifier/judge-gated, rejection-sampled |

Tiers 3–5 collapse to the assignment's canonical **"translated + synthetic"** (2.84T, ~86% of the lane) if only four buckets are wanted. **The honest headline: ~95% of the Indic lane is generated or repeated; only ~64B is authentically native**, which is *why* verified tokens are reserved for the anneal, not diluted into bulk.

**Per language, "22% Indic" is ten different realities** (ceiling = (V+U)×4 epochs): Hindi ~100B and Bengali ~65B **anchor** the native tiers; Tamil→Punjabi (~5–22B) are **majority-translated** the moment they claim a real share; **Kashmiri / Dogri / Santali (~0B native) get script coverage only, no capability promise.**

---

## 4. The three named slots

Each of the assignment's named slots is sized per stage and pointed at the datasets that fill it:

- **Agentic:** pretrain 2%/0.3T (format-prior seed: JSON/API-docs/schemas) + SFT ~22% + primary RLVR environment. Real supply is ~hundreds of M tokens (xLAM 60K, APIGen-MT 5K, ToolACE), so the slot is **generated**: APIGen-MT-style *blueprint → simulate → execute → verify* trajectories, execution-gated. Indian tools (UPI/IRCTC/DigiLocker/GST) don't exist publicly → generated with mock envs.
- **Reasoning:** pretrain 6%/0.9T (broad short+medium traces) + SFT ~15% (long-CoT cold-start) + anneal (gold). ~93% distilled (OpenThoughts, R1-distill ~800K). Front-loading asymmetry: **broad/diverse in pretrain, longest/highest-quality reserved for SFT+anneal.** Gate = answer verification (SymPy/tests).
- **Long-context:** **not a lane; a late re-pack stage** (S3–S4, anneal-aligned). Ramp 4K→32K→128K→256K, RoPE ABF, **40/60 rule** (40% at max length, 60% shorter). Packing sources ranked: code repos > books/science > Indian judgments (~7B, best real Indic long-doc) > stitched Indic. Honest: long-context is **mostly English/code**; Indic long-doc supply is thin, and the spec says so.

---

## 5. Protected floor + anneal reserve

**The floor** is the contract that stops the online (OPUS-style) selector (which compresses abundant lanes by training-utility) from starving a lane the offline fit deliberately funded. Enforced **per ~100B-token window**, not per batch, following the capability ranking:

| Floor | Minimum | From |
|---|---|---|
| Indic native tiers | **≥8%** of every window | S1 |
| Code + math (combined) | **≥12%** every window | S0 |
| Reasoning traces | **≥2%** every window | S2 |
| Agentic format-signal | **≥0.5%** every window | S2 |

Floors are **UniMax-consistent**: satisfied by capped-epoch native + gated translated, *never* by exceeding a per-language epoch cap.

**The nano-proxy makes this floor load-bearing, not decorative (§9.2).** A mixture with Indic at ~5% total (native only 3%, `mix02`) **collapsed both Indic lanes** (native score **0.896**, translated **0.875**, both under the 0.90 guard), while **8% native stayed safe (score 0.94–0.95)**. So there is a real collapse cliff just below the plan's floor. An unconstrained OPUS-style utility selector, which deprioritizes high-loss/slow-improving lanes, would drive Indic straight off that cliff; the **≥8% always-on native floor is precisely what stops it.** (The nano shows the *failure mode* that motivates the floor; that the floor+selector loop actually holds is still assumed, not tested.)

**The anneal reserve:** the final **~10% ≈ 1.5T tokens** (tail of S4), LR→0, checkpoint-averaged; the Llama-3 recipe that bought **+24% GSM8k / +6.4% MATH at 8B**. This is identity-setting, so the scarcest asset is spent here near-fresh:

| Reserve row | ~Tokens | Why |
|---|---|---|
| Verified-native Indic | ~0.19T | 64.3B held to ~1 epoch in bulk so ~3 run here on *real* text, not translationese |
| Edu-band-5 web (score 5) | ~0.40T | highest-density knowledge, upsampled last |
| Execution-verified code | ~0.25T | code that passed its tests, at LR→0 |
| FineMath-4+ / proofs | ~0.20T | top math band, where the +24% GSM8k is bought |
| NCERT / India-first + exam-synth | ~0.20T | ₹/IST/Indian-entity framing concentrated where it sticks |
| Gold long-reasoning (4–16K) | ~0.20T | longest/highest-quality CoT, reserved out of pretrain |

**Excluded from the anneal, deliberately:** unverified/translated Indic *bulk*. A model that ends on translated Indic learns English framing rendered in Devanagari as its default register.

---

## 6. The curriculum (per-stage weights)

Stage token budgets (stated assumption): S0 ~0.5T · S1 ~1.5T · S2 ~4T · S3 ~5T · S4 ~4T (incl. the 1.5T anneal) = 15T. Trajectory: **bilingual foundation (S0–S1) → capability core (S2) → Indic pivot (S3) → India-first anneal (S4)**, English code/math replayed throughout.

| Lane | S0 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| Web | 62 | 51 | 42.5 | 34 | 33 |
| Indic | 10 | 12 | 15 | 26 | 30 |
| Code | 18 | 22 | 24 | 20 | 16 |
| Math | 6 | 7 | 8 | 6 | 4 |
| Reasoning | 3 | 4 | 5 | 7 | 7 |
| Science | 1 | 2 | 3 | 2 | 1 |
| India-first | 0 | 1 | 1 | 3 | 6 |
| Agentic | 0 | 1 | 1.5 | 2 | 3 |

Token-weighted average recovers the §1 mixture within ~0.4pp on every lane; floors honored in every column; web recedes 62→33 as Indic climbs 10→30. Each rung boundary is a **go/no-go gate** (function-preservation + fundamentals probe). *Per-stage weights are provisional; the proxy tests only the aggregate ratios, not the stage schedule.*

---

## 7. Difficulty & reasoning-length bands

Difficulty is **measured**, per lane: web = FineWeb-Edu score 0–5; math = pass-rate; code = task-shape ladder; Indic = tier × KenLM-perplexity. **Reasoning-length bands: short <1K / medium 1–4K / long 4–16K** (OpenThoughts mean ≈11.6K); pretrain carries short+medium broadly, the long band is reserved for SFT/anneal. A worked example sits in every band, a sample rather than an adjective:

- **Math:** easy (GSM8K): *"12 notebooks at ₹35, sold at ₹50 each, total profit?"* → ₹180. mid (MATH): *"remainder of 2²⁰²⁶ mod 7"* → 2. hard (AIME): least *n* with a specified base-143 two-digit tail.
- **Code:** easy: docstring→`running_max`. mid: distinct-values-in-subarray under N,Q≤2×10⁵. hard: repo-level `pandas` groupby-rolling regression with a named failing test.
- **Indic:** verified-native (idiomatic Hindi), translated (a sentence with a *pointed-out* fronted-verb translationese artifact), synthetic exam-style (NCERT Pythagoras question in Devanagari).
- **Agentic:** single tool-call JSON / 2-turn dependent booking / error-recovery (`SEAT_UNAVAILABLE`→re-search→verify).

Band-to-schedule mapping: easy-broad-early → hard+long-late, with a Goldilocks exclusion routing near-zero-pass items to DART-MATH-style synthesis.

---

## 8. SFT and RL lanes

**SFT (~2.5–3M examples; shares are of examples, supply is sample-bound):** general 20% · **coding 22%** · **agentic 22%** · math/science reasoning 15% · long-context/format 9% · India-first + safety 12%. Buckets 1/4/5 are *also* localized; India-first is not confined to bucket 6. Agentic pools are tens-of-K samples → majority APIGen-MT-generated; reasoning uses OpenThoughts/R1-distill cold-start.

**RL:** DPO on preferences (UltraFeedback/HelpSteer2 + the **must-build** Indic preference set) → RLVR with *deterministic verifiers* (code: sandboxed tests; math: SymPy/Math-Verify; agentic: executable environments) → reasoning GRPO with a **language-consistency reward** so Indic CoT doesn't collapse to English. Grounded in the week-3 post-training plan.

---

## 9. Proxy validation

### 9.1 The 1B/3B proxy program

No share is trusted at 40B until a proxy has tested it. Five arms, each with a named confirm/refute rule, optimizing a **composite metric** (Indic .30 / code .25 / math+reasoning .20 / English .15 / agentic .10) under a **no-collapse guard** (no lane below 90% of its best) so a single-objective fit can't quietly trade Indic away:

| Arm | Question | Design | Confirm/refute |
|---|---|---|---|
| **A** | are the H_A ratios right at all? | 16×1B, Dirichlet mixtures, RegMix regression | predicted-best beats H_A by >1σ → adopt |
| **B** ⭐ | **22% Indic (H_A) vs ~13% + native-to-anneal (H_B)?** | 2×3B×100B, matched, both anneal + SFT probe | adopt H_B iff Indic gate within 1pt *and* English/reasoning gains ≥1.5pt. *Nano ran this multi-seed, trilingual (hi/bn/ta): H_A preferred but not sharply separated (§9.2); 3B settles it.* |
| **C** | code 20% right? | 3×1B, code 16/20/24% | pick the knee (expect flat 18–22) |
| **D** | translated-tier tolerance? | 3×1B, translated 40/55/70% of Indic | max translated share with <1pt native degradation |
| **E** | does reasoning at 6% earn its tokens? | 3×1B, reasoning 4/6/8% | confirm 6% post-SFT. *Nano ran 0/4/8%: composite rises monotonically, 4% still sub-guard → §1 adopted 6% (§9.2 R4).* |

Arm B resolves the fork the spec deliberately leaves open. Small→large mixture transfer is not an assumption here but a published, validated method: [Data Mixing Laws](https://arxiv.org/abs/2403.16952) and [Scaling Laws for Optimal Data Mixtures](https://arxiv.org/abs/2507.09404) both fit on sub-1B runs and predict the 8B optimum; RegMix validated 1M-param/1B-token proxies → 1B/25B. *Caveat:* the nano runs ~1.3 tok/param, far under RegMix's ~1000, so only large rank gaps are trusted, not absolute losses.

### 9.2 The nano-proxy run

A **7.5M-param decoder (24K BPE) × 10M tokens** on a MacBook (MPS): **6 lanes** (web · code · math · reasoning · Indic-native · Indic-translated) with **Indic split over {Hindi, Bengali, Tamil}**, so results are per-language rather than a Hindi average. **24 design runs** (H_A×3 seeds · H_B×3 seeds · 18 Dirichlet) + **Arm E** (reasoning 0/4/8%) + 2 out-of-sample confirmations. Composite + 90% collapse guard as §9.1; normalizer = best per-lane loss over design runs. It exercises the **offline** mixture layer only, not the §5 online selector; it is a **rank signal**, not a 40B number. Code + artifacts: [`proxy/v2/`](proxy/v2/README.md).

**Result 1 (Arm B): H_A (22% Indic) beats H_B (13%), but only modestly.**

| Hypothesis (3 seeds) | Composite C (mean, range) | native loss | native collapses |
|---|---|---|---|
| **H_A: Indic 22%** | **0.9333** (0.9302–0.9362) | **3.549** | 0/3 |
| H_B: Indic 13% | 0.9292 (0.9280–0.9312) | 3.596 | 0/3 |

Holding everything else fixed, more Indic scores higher, but the composite margin (0.004) sits **inside H_A's own 3-seed spread (0.006)**, so the robust signal is the **native-loss gap** (3.549 vs 3.596, no seed overlap), not the composite ranking. Neither lane collapses (native score 0.95 / 0.94, guard 0.90). So the nano *prefers* H_A but cannot *settle* 22-vs-13 by itself.

**Result 2: the result holds per-language, and native quality is bought with translated tokens.** H_A and H_B carry the *same* 8% native share and differ only in translated bulk (14% vs 5%), yet cutting it raises native loss in **all three languages, both script families:**

| Δ loss, translated 14→5% | Hindi | Bengali | Tamil |
|---|---|---|---|
| native | +0.054 | +0.048 | +0.047 |
| translated | +0.154 | +0.153 | +0.112 |

The lever on native quality is **total in-language exposure (native + translated)**, measured across Indo-Aryan and Dravidian. This is *why* the machine-made translated tier is worth funding.

**Result 3: the best mixtures keep Indic but slash web into code + reasoning, so H_A's weakness is an over-fat web lane, not its Indic share.** The regressor's predicted optimum (web .22 / code .37 / reasoning .19 / **Indic ~12%**) was **confirmed at C=0.9534, the highest of any run, no collapse** (Indic native score 0.93, translated 0.92); the best no-collapse observed run `mix15` (Indic 19%, code 32%) reproduced within **0.0003** on a fresh seed, so these optima are stable. But the driver is *not* less Indic (holding code/reasoning fixed, H_A's 22% still beats H_B's 13%, Result 1). The high-scoring mixtures all **cut web (H_A's 50% → ~19–26%) into code (18%→32–37%) and reasoning (4%→6–19%)**, while keeping Indic in the no-collapse band (**~10–33%, so 22% is safe**); push the other way and **60% Indic (`mix12`) collapses code/math/reasoning**. So the proxy's message about H_A is precise: **it over-funds web and under-funds code/reasoning** (consistent with Arm E) — which **§1 now corrects** (web 45→39, code 16→20, reasoning 4→6). *(Caveat: the composite rewards lanes whose loss is still improvable and is near-flat on lanes at their floor, so "cut web" may under-value web's downstream knowledge (MMLU/ARC) exactly as it under-values Indic identity. A 1B hypothesis, not a mandate.)*

**Result 4 (Arm E): the reasoning lane earns its tokens, and 4% looks a touch low.** With all else fixed, raising reasoning 0 → 4 → 8% lifts the reasoning-lane score **0.73 → 0.87 → 0.91** and the composite monotonically (**0.920 → 0.936** across the range). But at the plan's **4% the lane is still below the 0.90 guard** (0.87), clearing only at 8%. A signal to **consider 4→~6%**, tested properly at 1B.

**What the nano proves, and what it changed.** It ranks *aggregate mixture ratios* only — not the curriculum (§6), the anneal (§5), or any 40B number — and it runs on a proxy far smaller than the 1B/3B runs that would validate it at scale. What it earns is a signed direction, and **§1 now applies it: web trimmed 45→39%, code raised 16→20%, reasoning 4→6%, Indic held at 22%** (the floor the native-loss signal defends), with the §2 ledger re-run to match. Committing on nano-scale evidence is deliberate — and it is recorded that **the 22-vs-13 Indic fork is what a 1B/3B run (§9.1), Arm B especially, would settle: the validation this spec designs but cannot run on the compute available here.**

---

## 10. Starved slots (spec-only)

The five ledger-confirmed starved slots each get one honest fix; **running it is the team's parallel track, not this week's scope**, and the cumulative gating threshold is a course number not yet known here.

| Slot | Fix | Named pipeline | Success metric |
|---|---|---|---|
| Indic-native | **cleaning** (only real win) | IndicXlit romanized→native + targeted scrape + C8 MT-flag | +20–50B verified-native (×1.3–1.5, **never to lane scale**) |
| Agentic | generation | APIGen-MT verified-trajectory factory + Indian mocks | N verified multi-turn trajectories |
| Reasoning | generation | R1-distill + rejection sampling + DART-MATH synthesis | +XB verified CoT, difficulty-banded |
| Science | clean-then-gen | peS2o @2–3 epochs + paper-grounded textbook gen *(verifier-light, flagged)* | peS2o consumed + flagged synthetic |
| India-first | scrape + gen | GODL/§52 scrape + exam/civic QA gen | scrape ceiling harvested |

The key limit: **cleaning raises native Indic ~1.3–1.5×, never to lane scale**; the lane stays generation-heavy and the spec says so.

---

## 11. Caveats

- **~95% of the Indic lane is generated/repeated.** Stated, tiered, gated, not hidden. Whether 22%-of-mostly-generated beats a smaller high-density lane is **Arm B**; the nano's multi-seed trilingual signal *modestly* favors H_A (on native loss).
- **Floor values, anneal row sizes, per-stage weights, H_B's 13%, and the proxy decision thresholds are design proposals**, flagged provisional-until-proxy, not cited facts. Supply numbers, benchmark ties, and epoch precedents *are* sourced.
- **The composite-metric weights** (Indic .30 / code .25 / math+reasoning .20 / English .15 / agentic .10) are a renormalization of the capability ranking, not a sourced fact, and the nano-proxy verdict rides on them.
- **The nano-proxy is a 7.5M-param rank signal**, folds the 8 lanes to 6 (agentic/science/india-first folded into others), and measures Indic across 3 languages at ~1.3 tok/param. It is proof-of-method, not evidence for the 40B numbers.
- **Long-context is mostly English/code**; Indic long-doc depth is not something this data can honestly promise.
- **Kashmiri/Dogri/Santali get script coverage, not capability.** This spec claims no fluency it cannot fund.

---

### Map of the repository
- [`proxy/v2/`](proxy/v2/README.md): the runnable nano-proxy (multilingual hi/bn/ta, multi-seed) + results
- `research/week3`, `research/week4`: carried-over foundation (data/tokenizer strategy; cleaning pipeline)
- Fuller derivations and the dated decision trail live in a separate interlinked wiki (available on request).
