"""Sequence packing with correct masks — the "different data types packed correctly" invariant.

Multiple document spans are concatenated into a fixed-length sequence to avoid padding
waste. Packing is only correct if the model cannot bleed information across document
boundaries, so each pack carries:

  - input_ids  : the packed tokens (last token of each doc has no next-token target)
  - labels     : next-token targets; -100 (IGNORE) at doc-final positions and padding
  - seg_ids    : per-position document index within the pack (block-diagonal attention:
                 token i may attend to token j iff seg_ids[i]==seg_ids[j] and j<=i)
  - pos_ids    : position within each document, RESET to 0 at every doc start
  - loss_mask  : 1 where the position contributes to loss, else 0

References: block-diagonal attention + position-id reset for packed sequences
(arXiv 2107.02027 "packing without cross-contamination"; arXiv 2407.09105 packing+FlashAttn).
"""
import numpy as np

IGNORE = -100  # label value that contributes no loss (matches the usual convention)


def pack_spans(spans, seq_len, span_lanes=None):
    """Greedy first-fit packing of (tokens) spans into fixed-length packs.

    `spans` is a list of 1-D int arrays (one per doc-slice). Returns a list of packs,
    each a dict of int arrays of length `seq_len` plus `seg_lane` (lane string per segment,
    for per-lane loss attribution). A span longer than seq_len is truncated to seq_len.
    Utilization = loss-bearing / capacity.
    """
    if span_lanes is None:
        span_lanes = [None] * len(spans)
    packs = []
    cur = _new_pack(seq_len)
    used = 0
    seg = 0

    def flush():
        nonlocal cur, used, seg
        if used > 0:
            packs.append(cur)
        cur = _new_pack(seq_len)
        used = 0
        seg = 0

    for span, lane in zip(spans, span_lanes):
        span = np.asarray(span)
        if len(span) > seq_len:
            span = span[:seq_len]
        if len(span) == 0:
            continue
        if used + len(span) > seq_len:
            flush()
        s, e = used, used + len(span)
        cur["input_ids"][s:e] = span
        cur["seg_ids"][s:e] = seg
        cur["pos_ids"][s:e] = np.arange(len(span))
        # next-token labels within the doc; final token of the doc has no target
        cur["labels"][s:e - 1] = span[1:]
        cur["labels"][e - 1] = IGNORE
        cur["loss_mask"][s:e - 1] = 1
        cur["seg_lane"].append(lane)
        used = e
        seg += 1
    flush()
    return packs


def _new_pack(seq_len):
    return {
        "input_ids": np.zeros(seq_len, dtype=np.int64),
        "labels": np.full(seq_len, IGNORE, dtype=np.int64),
        "seg_ids": np.full(seq_len, -1, dtype=np.int64),  # -1 = padding
        "pos_ids": np.zeros(seq_len, dtype=np.int64),
        "loss_mask": np.zeros(seq_len, dtype=np.int64),
        "seg_lane": [],
    }


def per_lane_loss(packs, per_pos):
    """Attribute real per-position loss to lanes via seg_lane + seg_ids. Returns
    {lane: mean_loss} over loss-bearing positions of that lane."""
    from collections import defaultdict
    tot = defaultdict(float)
    cnt = defaultdict(int)
    for b, p in enumerate(packs):
        seg_lane = p["seg_lane"]
        for i in range(len(p["seg_ids"])):
            if p["loss_mask"][i] and p["seg_ids"][i] >= 0:
                lane = seg_lane[p["seg_ids"][i]]
                if lane is not None:
                    tot[lane] += float(per_pos[b, i])
                    cnt[lane] += 1
    return {lane: tot[lane] / cnt[lane] for lane in tot if cnt[lane]}


def attention_mask(seg_ids):
    """Block-diagonal causal mask from seg_ids. mask[i,j]=True iff j attends into i:
    same segment, j<=i, and neither is padding (seg_id<0)."""
    n = len(seg_ids)
    seg = seg_ids[:, None]
    segj = seg_ids[None, :]
    causal = np.tril(np.ones((n, n), dtype=bool))
    same = (seg == segj) & (seg >= 0) & (segj >= 0)
    return causal & same


def utilization(packs, seq_len):
    """Fraction of packed capacity that is loss-bearing (useful) tokens."""
    if not packs:
        return 0.0
    loss_tokens = sum(int(p["loss_mask"].sum()) for p in packs)
    return loss_tokens / (len(packs) * seq_len)
