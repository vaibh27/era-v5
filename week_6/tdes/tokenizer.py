"""Frozen byte-level BPE tokenizer.

The tokenizer is a *frozen artifact*: it is vendored as assets/tokenizer.json (a
byte-level BPE trained in week2) and never retrained here. Its identity is the
sha256 of the exact file bytes — the `tokenizer_hash` that every shard manifest
records and the manifest validator re-verifies.

Format (week2 custom BPE):
  - base tokens 0..255 are raw bytes
  - `merges` is an ordered list of [left_id, right_id, new_id]; list order == rank
  - vocab_size == 256 + len(merges)
"""
import hashlib
import json
from pathlib import Path

ASSET = Path(__file__).parent / "assets" / "tokenizer.json"


class Tokenizer:
    def __init__(self, path: Path = ASSET):
        self.path = Path(path)
        raw = self.path.read_bytes()
        # Frozen identity: hash the exact on-disk bytes.
        self.tokenizer_hash = hashlib.sha256(raw).hexdigest()
        spec = json.loads(raw)
        merges = spec["merges"]

        # rank[(a, b)] = (rank, new_id). Lower rank = higher merge priority.
        self.rank = {(a, b): (i, c) for i, (a, b, c) in enumerate(merges)}
        self.vocab_size = 256 + len(merges)

        # id -> bytes, built in dependency order (merges only reference lower ids).
        self.id_to_bytes = [bytes([i]) for i in range(256)]
        for a, b, c in merges:
            assert c == len(self.id_to_bytes), "merges must be contiguous from 256"
            self.id_to_bytes.append(self.id_to_bytes[a] + self.id_to_bytes[b])

    def encode(self, text: str) -> list[int]:
        """Deterministic byte-level BPE: repeatedly merge the lowest-rank adjacent pair."""
        ids = list(text.encode("utf-8"))
        if not ids:
            return ids
        while len(ids) >= 2:
            # find the adjacent pair with the best (lowest) rank
            best_rank = None
            best_pos = -1
            best_new = None
            for i in range(len(ids) - 1):
                hit = self.rank.get((ids[i], ids[i + 1]))
                if hit is not None and (best_rank is None or hit[0] < best_rank):
                    best_rank, best_new, best_pos = hit[0], hit[1], i
            if best_pos < 0:
                break
            ids[best_pos : best_pos + 2] = [best_new]
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.id_to_bytes[i] for i in ids).decode("utf-8", errors="replace")


_default = None


def get_tokenizer() -> Tokenizer:
    """Process-wide singleton so the frozen hash is loaded once."""
    global _default
    if _default is None:
        _default = Tokenizer()
    return _default
