"""Replay proof — reconstruct an earlier interval from the ledger and prove it matches.

Replay reads only the consumption ledger's recorded provenance (shard_id, start, end per
source span), re-slices the immutable shards, re-packs, and re-hashes. If the reconstructed
batch ids, token spans, and hashes equal the originally recorded ones, the historical data
stream is provably reconstructible from the ledger + immutable shards alone.
"""
from . import pack, shard
from .batch import batch_hash
from .ledger import Ledger


def reconstruct_batch(rec):
    """Rebuild a batch purely from its ledger record's provenance."""
    spans, lanes = [], []
    for s in rec["sources"]:
        toks = shard.load_tokens(s["shard_id"])[s["start"]:s["end"]]
        spans.append(toks)
        lanes.append(s["lane"])
    packs = pack.pack_spans(spans, rec["seq_len"], span_lanes=lanes)[:rec["n_seqs"] or 4]
    return packs


def run_replay_proof(a=4, b=12, log=None):
    cons = Ledger().read_consumption()
    b = min(b, len(cons))
    mismatches = []
    sample = None
    for step in range(a, b):
        rec = cons[step]
        packs = reconstruct_batch(rec)
        h = batch_hash(packs)
        if sample is None:  # first reconstructed batch: a visible recorded-vs-replay pair
            sample = {"step": step, "recorded": rec["batch_hash"], "replay": h}
        id_ok = (rec["batch_id"] == f"b{step:06d}")
        if h != rec["batch_hash"] or not id_ok:
            mismatches.append({"step": step, "recorded": rec["batch_hash"][:16],
                               "replay": h[:16], "id_ok": id_ok})
    ok = not mismatches
    if log:
        log(f"historical_stream_replayed interval=[{a},{b}) batches={b - a}")
        log(f"[{'PASS' if ok else 'FAIL'}] replay_hash_matched "
            f"interval=[{a},{b}) mismatches={len(mismatches)}")
    return ok, {"interval": [a, b], "n_batches": b - a, "mismatches": mismatches,
                "sample": sample,
                "sample_hash": cons[a]["batch_hash"] if b > a else None}
