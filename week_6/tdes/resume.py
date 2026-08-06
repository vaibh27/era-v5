"""Crash recovery proof — resume produces EXACTLY the next expected batch (no skip/repeat).

Method:
  1. reference run [0..N) -> capture the batch-hash stream (the ground truth).
  2. crash run with the same seed, crashing mid-run after checkpoints were saved.
  3. resume from the latest checkpoint -> continue to N.
  4. prove: (a) the checkpoint's recorded next_batch_hash == reference[offset]
            (b) the first resumed batch == reference[offset]  (resume_next_batch_matched)
            (c) the full resumed stream == reference  (no skipped or repeated batch)
"""
import json

from . import checkpoint, train
from .ledger import Ledger


def run_resume_proof(seed=42, crash_at=20, log=None):
    # 1. reference run
    train.run_training(seed=seed)
    ref = [r["batch_hash"] for r in Ledger().read_consumption()]

    # 2. crash run (same seed) — raises Crash after saving checkpoints
    try:
        train.run_training(seed=seed, crash_at=crash_at)
    except train.Crash:
        pass

    offset = checkpoint.latest_offset()
    meta = json.loads((checkpoint.paths.CHECKPOINTS / f"ckpt_{offset:06d}.json").read_text())
    expected = meta["next_batch_hash"]

    # 3. resume
    train.run_training(seed=seed, resume=True)
    resumed = [r["batch_hash"] for r in Ledger().read_consumption()]
    steps = [r["step"] for r in Ledger().read_consumption()]

    # 4. proofs
    checkpoint_expected_ok = (expected == ref[offset])
    next_matched = (resumed[offset] == ref[offset])
    stream_identical = (resumed == ref)
    contiguous = (steps == list(range(len(ref))))

    ok = checkpoint_expected_ok and next_matched and stream_identical and contiguous
    if log:
        log(f"crash_simulated at_step={crash_at} resumed_from_offset={offset}")
        log(f"run_resumed from_offset={offset} expected_next_hash={expected[:16]}")
        log(f"[{'PASS' if next_matched else 'FAIL'}] resume_next_batch_matched "
            f"offset={offset} hash={ref[offset][:16]}")
        log(f"[{'PASS' if stream_identical and contiguous else 'FAIL'}] "
            f"resume_no_skip_or_repeat steps={len(resumed)}")
    return ok, {"offset": offset, "checkpoint_expected_ok": checkpoint_expected_ok,
                "next_matched": next_matched, "stream_identical": stream_identical,
                "contiguous": contiguous, "n_steps": len(ref),
                "expected_hash": expected, "reference_hash": ref[offset]}
