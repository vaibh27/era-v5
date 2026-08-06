"""Invariant tests for TDES. Runs with pytest, or standalone: `python tests/test_invariants.py`.

Covers the properties the assignment grades: tokenizer integrity, immutable/reproducible
shards, packing correctness (no cross-document attention), a gradient check on the model's
backward pass, deterministic batches + provenance, the evaluation firewall, mixture floors,
and the crash-resume / replay / fork proofs.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tdes import (audit, batch, checkpoint, firewall, fork, manifest, mixture,  # noqa: E402
                  opus, pack, paths, replay, resume, shard, train)
from tdes.ckpt_policy import CheckpointPolicy, youngdaly_interval_steps  # noqa: E402
from tdes.ledger import Ledger  # noqa: E402
from tdes.model import NanoLM  # noqa: E402
from tdes.tokenizer import Tokenizer  # noqa: E402


# ---- fast unit tests (no training) ----

def test_tokenizer_hash_stable_and_roundtrip():
    a, b = Tokenizer(), Tokenizer()
    assert a.tokenizer_hash == b.tokenizer_hash
    for s in ["the quick brown fox", "नमस्ते दुनिया", "def f(x): return x*2", ""]:
        assert a.decode(a.encode(s)) == s


def test_shards_reproducible_and_manifest_valid():
    ids1 = sorted(m["shard_id"] for m in shard.build_shards())
    ids2 = sorted(m["shard_id"] for m in shard.build_shards())
    assert ids1 == ids2
    ok, _ = manifest.validate_all()
    assert ok


def test_no_cross_document_attention():
    d1, d2, d3 = np.array([11, 12, 13, 14]), np.array([21, 22, 23]), np.array([31, 32, 33])
    p = pack.pack_spans([d1, d2, d3], seq_len=10)[0]
    assert p["pos_ids"].tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 1, 2]
    m = pack.attention_mask(p["seg_ids"])
    seg = p["seg_ids"]
    leaks = [(i, j) for i in range(10) for j in range(10) if m[i, j] and seg[i] != seg[j]]
    future = [(i, j) for i in range(10) for j in range(10) if m[i, j] and j > i]
    assert not leaks and not future


def test_gradient_check():
    m = NanoLM(vocab=20, d_model=8, n_layer=2, max_pos=16, seed=1)
    rng = np.random.default_rng(3)
    ii = np.array([[3, 5, 7, 2, 9, 4], [1, 6, 8, 5, 3, 2]])
    seg = np.array([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, -1]])
    pos = np.array([[0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 2, 0]])
    lab = np.array([[5, 7, -100, 9, 4, -100], [6, 8, -100, 3, 2, -100]])
    loss, cache, _ = m.forward(ii, pos, seg, lab)
    g = m.backward(cache)
    eps, worst = 1e-5, 0.0
    for name in ["wte", "Whead", "l0.Wq", "l0.Wf2", "l0.ln1g", "lnfg"]:
        flat, gflat = m.p[name].reshape(-1), g[name].reshape(-1)
        for i in rng.choice(len(flat), size=min(3, len(flat)), replace=False):
            old = flat[i]
            flat[i] = old + eps; lp = m.forward(ii, pos, seg, lab)[0]
            flat[i] = old - eps; lm = m.forward(ii, pos, seg, lab)[0]
            flat[i] = old
            num, ana = (lp - lm) / (2 * eps), gflat[i]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    assert worst < 1e-4, f"gradient check failed: {worst}"


def test_batch_deterministic_and_provenance():
    mans = shard.build_shards()
    eligible, _ = firewall.partition(mans)
    lw = {l: 1.0 for l in mixture.LANES}
    b1 = batch.build_batch(3, 42, eligible, lw)
    b2 = batch.build_batch(3, 42, eligible, lw)
    assert b1["batch_hash"] == b2["batch_hash"]
    recon = [shard.load_tokens(s["shard_id"])[s["start"]:s["end"]] for s in b1["sources"]]
    packs = pack.pack_spans(recon, 64, span_lanes=[s["lane"] for s in b1["sources"]])[:4]
    assert batch.batch_hash(packs) == b1["batch_hash"]


def test_firewall_partition_and_leak_detection():
    mans = shard.build_shards()
    eligible, blocked = firewall.partition(mans)
    assert blocked and all(m["split"] == "eval" for m in blocked)
    ok, _ = firewall.assert_no_eval_consumed([m["shard_id"] for m in eligible], mans)
    assert ok
    bad, leaked = firewall.assert_no_eval_consumed([blocked[0]["shard_id"]], mans)
    assert not bad and leaked == [blocked[0]["shard_id"]]


def test_decontamination():
    ok, rep = firewall.decontaminate(shard.build_shards())
    assert ok and rep["worst_jaccard"] < firewall.DUP_JACCARD


def test_checkpoint_atomic_ram_and_retention():
    import tempfile
    default = paths.ART
    d = tempfile.mkdtemp()
    try:
        paths.use_artifacts_dir(d)
        checkpoint.clear()
        m = NanoLM(vocab=30, d_model=8, n_layer=1, max_pos=16, seed=2)
        o = train.AdamW(m.p)
        checkpoint.save(8, 42, m, o, "S0", "h8", async_=True)   # async + in-memory tier
        assert checkpoint.loaded_from_ram(8)
        checkpoint.join_writes()
        assert (paths.CHECKPOINTS / "ckpt_000008.npz").exists()
        assert (paths.CHECKPOINTS / "ckpt_000008.json").exists()
        # AdamW updates params in place; the RAM-tier copy must be unaffected
        w0 = float(m.p["Whead"][0, 0]); m.p["Whead"] -= 123.0
        checkpoint.load(8, m, o)
        assert abs(float(m.p["Whead"][0, 0]) - w0) < 1e-9
        # retention keeps only the newest N (disk + RAM tier)
        checkpoint.save(16, 42, m, o, "S1", "h16")
        checkpoint.save(24, 42, m, o, "S2", "h24")
        checkpoint.keep_last_n(1)
        assert checkpoint.latest_offset() == 24
        assert not (paths.CHECKPOINTS / "ckpt_000008.json").exists()
        assert not checkpoint.loaded_from_ram(8)
    finally:
        paths.use_artifacts_dir(default)


def test_youngdaly_interval_and_policy():
    yd = youngdaly_interval_steps
    # interval grows with MTBF and checkpoint cost, and degenerate inputs clamp to lo=1
    assert yd(0.1, 0.25, 3600) > yd(0.1, 0.25, 60) > yd(0.1, 0.25, 1)
    assert yd(1.0, 0.25, 100) > yd(0.1, 0.25, 100)
    assert yd(0.1, 0.0, 100) == 1
    per = CheckpointPolicy("periodic", 8, total_steps=32)
    assert [o for o in range(1, 33) if per.should_checkpoint(o)] == [8, 16, 24, 32]
    jit = CheckpointPolicy("jit_only", 8, total_steps=32)
    assert [o for o in range(1, 33) if jit.should_checkpoint(o)] == [32]


# ---- integration tests (training) ----

def test_training_and_ledgers():
    train.run_training(seed=42)
    learn = Ledger().read_learning()
    cons = Ledger().read_consumption()
    assert len(cons) == len(learn) == mixture.total_steps()
    assert [r["step"] for r in cons] == list(range(len(cons)))
    assert np.mean([r["loss"] for r in learn[-4:]]) < np.mean([r["loss"] for r in learn[:4]])


def test_mixture_floors_respected():
    train.run_training(seed=42)
    ok, _, detail = audit._mixture_compliance()
    assert ok, detail["floor_breaches"]


def test_reproducible_batch_stream():
    train.run_training(seed=42)
    h1 = [r["batch_hash"] for r in Ledger().read_consumption()]
    train.run_training(seed=42)
    h2 = [r["batch_hash"] for r in Ledger().read_consumption()]
    assert h1 == h2


def test_resume_next_batch_matches():
    ok, rep = resume.run_resume_proof(seed=42, crash_at=20)
    assert ok and rep["next_matched"] and rep["stream_identical"] and rep["contiguous"]


def test_replay_hashes_match():
    train.run_training(seed=42)
    ok, rep = replay.run_replay_proof(a=4, b=16)
    assert ok and not rep["mismatches"]


def test_fork_diverges_and_valid():
    ok, rep = fork.run_fork_proof(fork_offset=8, parent_seed=42, fork_seed=7, k=6)
    assert ok and rep["diverged"] and rep["firewall_clean"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
