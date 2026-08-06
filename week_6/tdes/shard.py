"""Immutable, content-addressed tokenized shards + their manifests.

One shard per (lane, split). A shard's payload is a flat uint16 array of the
concatenated token ids of its docs; its identity is the sha256 of those bytes
(content-addressed). Document boundaries are preserved in the manifest so packing
can build correct per-document masks later.

Manifests carry NO wall-clock fields, so a rebuild from the frozen corpus is
byte-identical — the reproducibility the audit checks.
"""
import hashlib
import json

import numpy as np

from . import paths
from .tokenizer import get_tokenizer

DTYPE = np.uint16  # vocab 10000 < 65536
LANE_ORDER = ["web", "indic", "code", "math"]
SPLIT_ORDER = ["train", "eval"]
CHUNK = 28  # docs are split into <=CHUNK-token pieces so every span fits a sequence
            # (< seq_len): no packing truncation, several pieces pack per sequence, and
            # token-aware lane balancing becomes possible regardless of raw doc length.


def _content_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _load_corpus():
    with open(paths.CORPUS, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_shards(log=None):
    """Tokenize the frozen corpus into (lane, split) shards. Returns list of manifests."""
    paths.ensure_dirs()
    # fresh rebuild: clear any prior shards/manifests so a changed corpus can't leave orphans
    for p in list(paths.SHARDS.glob("*.bin")) + list(paths.MANIFESTS.glob("*.json")):
        p.unlink()
    tok = get_tokenizer()
    docs = _load_corpus()

    # deterministic grouping: fixed lane/split order, docs in corpus file order
    groups = {(l, s): [] for l in LANE_ORDER for s in SPLIT_ORDER}
    for d in docs:
        groups[(d["lane"], d["split"])].append(d)

    manifests = []
    for lane in LANE_ORDER:
        for split in SPLIT_ORDER:
            group = groups[(lane, split)]
            if not group:
                continue
            tokens, doc_starts, doc_ids = [], [], []
            for d in group:
                ids = tok.encode(d["text"])
                # split each doc into <=CHUNK-token pieces, each a packing segment
                for ci in range(0, max(1, len(ids)), CHUNK):
                    chunk = ids[ci:ci + CHUNK]
                    if not chunk:
                        continue
                    doc_starts.append(len(tokens))
                    tokens.extend(chunk)
                    doc_ids.append(f"{d['doc_id']}#{ci // CHUNK}")
            arr = np.asarray(tokens, dtype=DTYPE)
            content_hash = _content_hash(arr)
            shard_id = content_hash[:16]

            arr.tofile(paths.SHARDS / f"{shard_id}.bin")
            manifest = {
                "shard_id": shard_id,
                "lane": lane,
                "split": split,
                "tokenizer_hash": tok.tokenizer_hash,
                "content_hash": content_hash,
                "dtype": "uint16",
                "n_tokens": int(arr.size),
                "n_docs": len(doc_ids),
                "doc_starts": doc_starts,
                "doc_ids": doc_ids,
            }
            with open(paths.MANIFESTS / f"{shard_id}.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            manifests.append(manifest)
            if log:
                log(f"shard_created lane={lane} split={split} id={shard_id} "
                    f"tokens={arr.size} docs={len(doc_ids)}")
    return manifests


def load_manifests():
    out = []
    for p in sorted(paths.MANIFESTS.glob("*.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def load_tokens(shard_id) -> np.ndarray:
    return np.fromfile(paths.SHARDS / f"{shard_id}.bin", dtype=DTYPE)


def iter_docs(manifest, tokens=None):
    """Yield (doc_id, start, end) spans for each doc in a shard."""
    if tokens is None:
        tokens = load_tokens(manifest["shard_id"])
    starts = manifest["doc_starts"]
    n = manifest["n_tokens"]
    for i, (doc_id, start) in enumerate(zip(manifest["doc_ids"], starts)):
        end = starts[i + 1] if i + 1 < len(starts) else n
        yield doc_id, start, end
