"""Append-only consumption + learning ledgers — "what did I eat, and what did I learn."

  consumption.jsonl : one line per training step — the batch id/hash, the exact source
                      spans (shard_id, lane, start, end), packing utilization, loss-token
                      count. This is the record replay/audit reconstructs batches from.
  learning.jsonl    : one line per step — the real loss, per-lane loss attributed back to
                      source spans, and the lr. This links what was learned to source data.

Both are strictly append-only (open in 'a' mode); offsets are line counts, so a
checkpoint's ledger_offset == number of steps recorded == next step index.
"""
import json

from . import paths


class Ledger:
    def __init__(self):
        self.consumption = paths.LEDGERS / "consumption.jsonl"
        self.learning = paths.LEDGERS / "learning.jsonl"

    def reset(self):
        paths.ensure_dirs()
        self.consumption.write_text("")
        self.learning.write_text("")

    def append_consumption(self, batch, opus_records=None):
        rec = {
            "step": batch["step"],
            "batch_id": batch["batch_id"],
            "batch_hash": batch["batch_hash"],
            "n_seqs": batch["n_seqs"],
            "seq_len": batch["seq_len"],
            "util": round(batch["util"], 6),
            "n_loss_tokens": batch["n_loss_tokens"],
            "sources": batch["sources"],
            "opus": opus_records or [],
        }
        with open(self.consumption, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def append_learning(self, step, loss, per_lane_loss, lr):
        rec = {"step": step, "loss": round(float(loss), 6),
               "per_lane_loss": {k: round(float(v), 6) for k, v in per_lane_loss.items()},
               "lr": lr}
        with open(self.learning, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def offset(self):
        """Number of steps recorded == next step index (checkpoints reference this)."""
        if not self.consumption.exists():
            return 0
        with open(self.consumption, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def read_consumption(self):
        with open(self.consumption, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def read_learning(self):
        with open(self.learning, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def truncate_to(self, offset):
        """Drop ledger rows at/after `offset` — used on resume to discard any steps
        that were recorded after the last checkpoint (so no batch is double-counted)."""
        for path in (self.consumption, self.learning):
            if not path.exists():
                continue
            rows = path.read_text(encoding="utf-8").splitlines()
            keep = [r for r in rows if json.loads(r)["step"] < offset]
            path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
