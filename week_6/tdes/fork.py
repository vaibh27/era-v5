"""Fork proof — branch a new run from an earlier checkpoint.

Load the parent's checkpoint at `fork_offset`, then continue with a DIFFERENT seed. The
fork inherits the parent's exact state at the branch point (shared history [0..fork_offset))
but produces a divergent, valid stream afterwards. Proves: divergence at the branch point,
and that the fork remains firewall-clean with real (finite) losses.
"""
import json

import numpy as np

from . import batch, checkpoint, mixture, opus, train
from .ledger import Ledger
from .model import NanoLM


def run_fork_proof(fork_offset=None, parent_seed=42, fork_seed=7, k=6, log=None):
    # parent reference run (establishes shared history + checkpoints)
    _, eligible = train.setup()
    train.run_training(seed=parent_seed, eligible=eligible)
    parent = [r["batch_hash"] for r in Ledger().read_consumption()]

    # branch from an actually-saved checkpoint. Which offsets exist depends on the checkpoint
    # interval, so snap a missing/None request to the earliest non-final checkpoint.
    avail = sorted(int(p.stem.split("_")[1])
                   for p in checkpoint.paths.CHECKPOINTS.glob("ckpt_*.json"))
    if fork_offset not in avail:
        non_final = [o for o in avail if o < len(parent)]
        fork_offset = (non_final or avail)[0]

    # load the branch-point checkpoint into a fresh model/optimizer
    model = NanoLM(vocab=train.get_vocab(), d_model=64, n_layer=2, max_pos=64, seed=parent_seed)
    opt = train.AdamW(model.p, lr=0.02)
    checkpoint.load(fork_offset, model, opt)
    selector = opus.OpusSelector(model, eligible, seq_len=64)

    fork_rows = []
    for step in range(fork_offset, min(fork_offset + k, mixture.total_steps())):
        selector.set_context(step, {})  # fresh window accounting on the branch
        b = batch.build_batch(step, fork_seed, eligible, mixture.floored_weights(step),
                              seq_len=64, n_seqs=4, selector=selector)
        loss, _ = train.train_step(model, opt, b)
        fork_rows.append({"step": step, "batch_hash": b["batch_hash"],
                          "loss": round(float(loss), 6),
                          "shards": sorted({s["shard_id"] for s in b["sources"]})})

    # proofs
    diverged = fork_rows[0]["batch_hash"] != parent[fork_offset]
    finite = all(np.isfinite(r["loss"]) for r in fork_rows)
    train_ids = {m["shard_id"] for m in eligible}
    clean = all(sid in train_ids for r in fork_rows for sid in r["shards"])

    (train.checkpoint.paths.LEDGERS / "fork.json").write_text(
        json.dumps({"fork_offset": fork_offset, "parent_seed": parent_seed,
                    "fork_seed": fork_seed, "shared_prefix_hashes": parent[:fork_offset],
                    "fork_rows": fork_rows}, indent=2), encoding="utf-8")

    ok = diverged and finite and clean
    if log:
        log(f"branch_forked from_offset={fork_offset} parent_seed={parent_seed} fork_seed={fork_seed}")
        log(f"[{'PASS' if ok else 'FAIL'}] fork_diverges_and_valid "
            f"diverged={diverged} firewall_clean={clean} steps={len(fork_rows)}")
    return ok, {"fork_offset": fork_offset, "diverged": diverged, "firewall_clean": clean,
                "shared_prefix_len": fork_offset, "fork_steps": len(fork_rows),
                "parent_branch_hash": parent[fork_offset][:16],
                "fork_branch_hash": fork_rows[0]["batch_hash"][:16]}
