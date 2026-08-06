"""Manifest validation — re-derive every hash from the shard bytes on disk.

This is what makes shard integrity *evidence* rather than a claim: for each manifest
we recompute the content hash from the shard payload and confirm it matches, and we
confirm the recorded tokenizer_hash equals the live frozen tokenizer's hash.
"""
import hashlib

import numpy as np

from . import shard
from .tokenizer import get_tokenizer


def validate_all(log=None):
    """Returns (ok, report). Verifies content hashes, tokenizer hash, token counts."""
    tok = get_tokenizer()
    manifests = shard.load_manifests()
    report = {"n_shards": len(manifests), "tokenizer_hash": tok.tokenizer_hash,
              "shards": [], "content_ok": True, "tokenizer_ok": True}

    for m in manifests:
        arr = shard.load_tokens(m["shard_id"])
        recomputed = hashlib.sha256(arr.tobytes()).hexdigest()
        content_ok = (recomputed == m["content_hash"]) and (arr.size == m["n_tokens"])
        tok_ok = (m["tokenizer_hash"] == tok.tokenizer_hash)
        report["content_ok"] &= content_ok
        report["tokenizer_ok"] &= tok_ok
        report["shards"].append({
            "shard_id": m["shard_id"], "lane": m["lane"], "split": m["split"],
            "n_tokens": m["n_tokens"], "content_ok": content_ok, "tokenizer_ok": tok_ok,
        })

    # one value certifying the whole immutable input set: sha256 over the sorted per-shard
    # content hashes (order-independent, deterministic -> reproducible across runs/machines).
    report["root_hash"] = hashlib.sha256(
        "".join(sorted(m["content_hash"] for m in manifests)).encode()).hexdigest()

    ok = report["content_ok"] and report["tokenizer_ok"] and len(manifests) > 0
    if log:
        if report["tokenizer_ok"]:
            log(f"[PASS] tokenizer_hash_verified {tok.tokenizer_hash[:16]}")
        else:
            log("[FAIL] tokenizer_hash_verified")
        log(f"[{'PASS' if ok else 'FAIL'}] manifest_validated "
            f"shards={len(manifests)} content_ok={report['content_ok']}")
    return ok, report
