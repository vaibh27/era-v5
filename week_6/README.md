# TDES — Training Data Execution System (V5)

A small but complete training-data pipeline that **proves** what it consumed, why it
consumed it, what the model learned from it, and how the run can be reconstructed. Scale is
not the point; **correctness, reproducibility, and auditability** are. Pure numpy — no
torch, no external ML deps — so the whole demonstration runs from one command and every
number is reconstructible.

## Setup

Requires **Python 3.9+**. The only dependency is **numpy**:

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_demo.py
```

This regenerates `submission_artifacts/` end to end and exits 0 iff every requirement passes.
It is deliberately a **fresh, reproducible** run: re-running produces byte-identical
artifacts, which is what the grader checks.

For the **resumable operational flow** — the one you can kill and restart — use the pipeline:

```bash
python run_pipeline.py                 # runs to completion, checkpointing as it goes
#   ^C  /  kill   → flushes a just-in-time checkpoint, exits 130
python run_pipeline.py                 # resumes from that checkpoint, finishes the run
```

It writes to its own `pipeline_artifacts/` tree (never touching the graded output). Because
every batch is a pure function of `(seed, step)`, a killed-then-resumed run converges to the
**same** final state as an uninterrupted one. See *Scaling the checkpoint & recovery path*.

## The pipeline

```
documents → tokenized shards → manifests → mixture schedule → packing →
batches → training → consumption ledger → learning ledger → checkpoint →
crash → resume → replay → audit
```

## Architecture (`tdes/`)

| Module | Role |
|---|---|
| `tokenizer.py` | Frozen byte-level BPE (vendored `assets/tokenizer.json`); stable `tokenizer_hash` |
| `shard.py` | Immutable, content-addressed token shards (id = sha256 of tokens); docs chunked to ≤28 tokens |
| `manifest.py` | Re-derives every content/tokenizer hash from bytes on disk |
| `firewall.py` | Blocks eval shards from loss-bearing batches; Jaccard decontamination |
| `mixture.py` | Curriculum stages + per-window protected floors (indic ≥0.20, code+math ≥0.20) |
| `opus.py` | OPUS selector — utility = LM-head-gradient cosine vs a corpus-retrieved proxy (arXiv 2602.05400) |
| `pack.py` | Fixed-length packing: loss mask, **block-diagonal** attention, reset position-ids |
| `model.py` | Nano transformer, hand-written forward **and backward** (gradient-checked) |
| `batch.py` | Deterministic `(seed, step) → batch` with a content hash and full provenance |
| `ledger.py` | Append-only consumption + learning ledgers |
| `checkpoint.py` | Model + AdamW + ledger-offset checkpoints; **atomic** writes, **in-memory tier**, optional **async** |
| `ckpt_policy.py` | *When* to checkpoint: Young/Daly-sized interval or just-in-time, plus keep-last-N |
| `train.py` | The loop: batch → forward/backward → record → checkpoint; supports crash + resume + JIT stop-hook |
| `resume.py` / `replay.py` / `fork.py` | The three reconstruction proofs |
| `audit.py` | Independently re-verifies every requirement → `evidence.json`, `evidence.md`, `performance.json` |

Plus two entry points: `run_demo.py` (fresh reproducible demonstration) and `run_pipeline.py`
(resumable operational flow).

## Design decisions

- **Pure numpy, no torch.** A self-contained one-command run that the grader can re-execute,
  with full control over determinism (no framework RNG to fight). The model is genuinely
  tiny (2-layer, d=64) but real: hand-written backward, validated by a finite-difference
  **gradient check** — so the learning ledger and OPUS utility are trustworthy.
- **Determinism is foundational.** Batch draws use a counter-based RNG on `(seed, step)`;
  lanes/shards are visited in sorted order. Checkpoints store the ledger offset, so resume
  rebuilds batch `#offset` deterministically — no RNG state to save.
- **OPUS is real, not a stub.** It scores candidates in LM-head *update space* (a cosine
  against a benchmark-aligned proxy retrieved from the *train* split), and the protected
  floor overrides it. All four decision types (accept/reject/defer/floor-override) occur.
- **Evidence is derived.** The audit re-reads shards/ledgers/checkpoints and recomputes
  hashes, shares, and metrics. Nothing in `evidence.json` is written by hand.
- **Reuses prior weeks** (vendored so the run is self-contained): the week2 BPE tokenizer,
  real web+indic docs from the week4 corpus, and the week5 §40 floor/curriculum discipline.

## Corpus

`assets/corpus.jsonl` — 4 lanes: **web** and **indic** (real, from the week4 corpus) plus
**code** and **math** (deterministically generated). Near-dedup'd (no pair exceeds 0.7
token-Jaccard) and split train/eval. Rebuild with `python tdes/assets/build_corpus.py`.

## Output (`submission_artifacts/`)

```
run.log            ordered event log ([PASS] tokenizer_hash_verified, eval_shard_blocked, …)
evidence.json      machine-readable pass/fail + evidence pointers (all derived)
evidence.md        human-readable Requirement/Result/Evidence table
manifests/         one JSON per shard (hashes, lane, split, doc spans)
ledgers/           consumption.jsonl (batch→spans+hashes+OPUS), learning.jsonl (per-lane loss), fork.json
checkpoints/       model+optimizer+offset (npz + json)
shards/            immutable token .bin shards (kept for replay/audit)
performance.json   tokens/sec, useful loss-bearing tokens/sec, packing utilization
```

## The three proofs

- **Crash recovery** — a deliberate crash mid-run; resume from the last checkpoint produces
  *exactly* the next expected batch (hash-matched against an independent reference run) with
  no skipped or repeated step.
- **Replay** — an earlier interval is reconstructed purely from the ledger's provenance and
  the immutable shards; batch ids, token spans, and hashes match the original.
- **Fork** — a branch from an earlier checkpoint with a different seed diverges into a valid,
  firewall-clean stream while sharing the parent's history up to the branch point.

## Scaling the checkpoint & recovery path

Both entry points size their checkpoint interval by **Young/Daly** rather than a magic
constant. `run_demo.py` derives it from *fixed representative constants* (logged) so it stays
byte-reproducible; `run_pipeline.py` measures the constants **live**. This matters because
too-frequent checkpointing burns throughput while too-rare checkpointing loses compute on
every failure. The design is informed by the literature, and the ideas with an honest
single-process analog are actually implemented (no simulation); the rest are documented here
for when this moves to real GPU training.

**Implemented now** (`ckpt_policy.py` + `checkpoint.py`, used by `run_pipeline.py`):

- **Young/Daly interval** — instead of a magic constant, size the period by
  `T* ≈ √(2·C·MTBF)` (C = measured checkpoint cost, MTBF = assumed mean-time-between-failures).
  `run_pipeline.py` calibrates C and the step time live and logs the derivation.
  *(Young 1974; Daly 2006; survey [S0167739X24003777].)*
- **Just-in-time checkpointing** — `--mode jit`: don't checkpoint on a timer; flush only on a
  stop/failure signal (plus the final step). Near-zero steady-state overhead; a hard crash
  replays from the last committed offset. *(Gupta et al., "Just-In-Time Checkpointing",
  EuroSys 2024.)*
- **Atomic writes** — temp file + `os.replace`, so a `kill -9` mid-write never corrupts a
  checkpoint.
- **In-memory tier (Gemini-style)** — `save` also caches the checkpoint in RAM (as copies);
  `load` serves from RAM to accelerate *in-process* recovery. Honest limit: process-local, so
  it does **not** survive a real kill — that falls back to the durable disk copy. Gemini's
  cross-node RAM replication is what removes that limit at scale.
  *(Wang et al., "Gemini", SOSP 2023.)*
- **Async writes** — `save(async_=True)` writes on a background thread while the RAM tier keeps
  the checkpoint immediately readable. *(DataStates-LLM, arXiv 2406.10707; CheckFreq, FAST'21.)*
- **Keep-last-N retention** — bound checkpoint storage.

**Forward-looking (documented, not implemented — no meaningful analog at nano/CPU scale):**

| Technique | Paper | Why deferred |
|---|---|---|
| Differential checkpointing | Check-N-Run (NSDI'22, arXiv 2010.08679) | Every nano-model param changes each step → dense, no savings. Pays off for sparse embedding tables. |
| Quantized/compressed checkpoints | Check-N-Run; Inshrinkerator (arXiv 2306.11800) | float16/compression breaks the **bit-exact** resume proof; only safe as a separate size metric. |
| Convergence-aware placement | COCI (arXiv/ScienceDirect S0167739X24005612) | Denser checkpoints early (progress is worth more early) — needs a longer run to matter. |
| Tiered GPU→CPU→remote | TierCheck (arXiv 2605.17821); ByteCheckpoint (arXiv 2407.20143) | No GPU/remote tiers here; our RAM+disk is the two-tier subset. |
| Redundancy / peer recovery | Oobleck (SOSP'23, arXiv 2309.08125); Bamboo | Needs ≥2 workers to recover from a peer's live state. |
| Failure-rate modeling | Meta reliability study (arXiv 2410.21680); FlashRecovery (arXiv 2509.03047) | Supplies the real MTBF that feeds Young/Daly at cluster scale. |

## Tests

```
python tests/test_invariants.py     # or: pytest tests/
```

Covers tokenizer integrity, shard reproducibility, no-cross-document-attention, the model
gradient check, batch determinism + provenance, the firewall, mixture floors, the
resume/replay/fork proofs, and the checkpoint policy (atomic + in-memory + retention,
Young/Daly interval).

## References

OPUS data selection (arXiv 2602.05400) · sequence packing without cross-contamination
(2107.02027) · packing + FlashAttention (2407.09105) · RegMix (2407.01492) · UniMax
(2304.09151) · deterministic resumption pattern (MosaicML StreamingDataset).
