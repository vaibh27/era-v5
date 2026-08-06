"""Evaluation/validation firewall.

Two guarantees:
  1. Only train-split shards are eligible for loss-bearing batches; eval shards are
     blocked. `assert_no_eval_consumed` re-checks this against the consumption ledger
     during audit, so a leak fails the run.
  2. Decontamination: no eval doc may be a near-duplicate of a train doc. Measured as
     per-doc max Jaccard of token 5-gram shingles vs same-lane train docs (the MinHash/
     Jaccard near-dup approach). This catches whole-instance leakage without falsely
     flagging templated lanes that legitimately share sentence *structure*.
"""
from . import shard

SHINGLE = 5
DUP_JACCARD = 0.8  # >= this vs any train doc == near-duplicate leakage


def partition(manifests):
    eligible = [m for m in manifests if m["split"] == "train"]
    blocked = [m for m in manifests if m["split"] == "eval"]
    return eligible, blocked


def _shingles(tokens, n=SHINGLE):
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _docs_by_lane(manifests):
    """Group shingles by ORIGINAL doc (chunks share a base id before '#'), so
    decontamination compares whole docs, not internal shard chunks."""
    by_lane = {}
    for m in manifests:
        toks = shard.load_tokens(m["shard_id"])
        acc = {}
        for doc_id, s, e in shard.iter_docs(m, toks):
            base = doc_id.split("#")[0]
            acc.setdefault(base, set()).update(_shingles(toks[s:e].tolist()))
        for base, sh in acc.items():
            by_lane.setdefault(m["lane"], []).append((base, sh))
    return by_lane


def decontaminate(manifests, log=None):
    """Assert no eval doc is a near-duplicate of a same-lane train doc. Returns (ok, report)."""
    eligible, blocked = partition(manifests)
    train_by_lane = _docs_by_lane(eligible)
    eval_by_lane = _docs_by_lane(blocked)

    worst = 0.0
    worst_pair = None
    n_checked = 0
    for lane, eval_docs in eval_by_lane.items():
        train_docs = train_by_lane.get(lane, [])
        for ev_id, ev_sh in eval_docs:
            n_checked += 1
            for tr_id, tr_sh in train_docs:
                j = _jaccard(ev_sh, tr_sh)
                if j > worst:
                    worst, worst_pair = j, (ev_id, tr_id)

    ok = worst < DUP_JACCARD
    report = {"eval_docs_checked": n_checked, "worst_jaccard": round(worst, 4),
              "threshold": DUP_JACCARD, "shingle": SHINGLE, "worst_pair": worst_pair}
    if log:
        log(f"[{'PASS' if ok else 'FAIL'}] eval_decontaminated "
            f"worst_jaccard={worst:.4f} threshold={DUP_JACCARD} docs={n_checked}")
    return ok, report


def assert_no_eval_consumed(consumed_shard_ids, manifests, log=None):
    """Given shard ids that entered training batches, prove none are eval-split."""
    _, blocked = partition(manifests)
    blocked_ids = {m["shard_id"] for m in blocked}
    leaked = set(consumed_shard_ids) & blocked_ids
    ok = not leaked
    if log:
        log(f"[{'PASS' if ok else 'FAIL'}] eval_shard_blocked "
            f"blocked_shards={len(blocked_ids)} leaked={sorted(leaked)}")
    return ok, sorted(leaked)
