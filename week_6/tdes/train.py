"""Training loop — pull batch, real forward/backward/update, record both ledgers.

Wires the deterministic batch core to a real optimizer and the append-only ledgers, and
saves checkpoints tied to ledger offsets. Supports a deliberate crash (`crash_at`) and
resume (`resume=True`), which Phase 5 uses to prove exact continuation.
"""
import numpy as np

from . import batch, checkpoint, firewall, manifest, mixture, opus, shard
from .ledger import Ledger
from .model import NanoLM


class Crash(Exception):
    """Deliberate crash to exercise recovery."""


class Interrupted(Exception):
    """Raised when `should_stop` fires: a checkpoint has been flushed at the current offset
    and the run can be resumed. Carries that offset."""

    def __init__(self, offset):
        super().__init__(f"interrupted; checkpoint saved at offset {offset}")
        self.offset = offset


class AdamW:
    def __init__(self, params, lr=0.01, betas=(0.9, 0.95), wd=0.01, eps=1e-8):
        self.lr, self.b1, self.b2, self.wd, self.eps = lr, betas[0], betas[1], wd, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * (mhat / (np.sqrt(vhat) + self.eps) + self.wd * params[k])


def _stack(packs):
    return (np.stack([p["input_ids"] for p in packs]),
            np.stack([p["pos_ids"] for p in packs]),
            np.stack([p["seg_ids"] for p in packs]),
            np.stack([p["labels"] for p in packs]))


def train_step(model, opt, b):
    from . import pack
    ii, pos, seg, lab = _stack(b["packs"])
    loss, cache, per_pos = model.forward(ii, pos, seg, lab)
    grads = model.backward(cache)
    opt.step(model.p, grads)
    per_lane = pack.per_lane_loss(b["packs"], per_pos)
    return loss, per_lane


def setup(log=None):
    """Build + validate shards, run the firewall, return eligible (train) manifests."""
    shard.build_shards(log=log)
    ok, _ = manifest.validate_all(log=log)
    assert ok, "manifest validation failed"
    mans = shard.load_manifests()
    ok_d, _ = firewall.decontaminate(mans, log=log)
    assert ok_d, "decontamination failed"
    eligible, _ = firewall.partition(mans)
    return mans, eligible


def _window_lane_tokens(rows, window):
    """Per-lane source-token totals for steps in `window` (for floor enforcement)."""
    acc = {}
    for r in rows:
        if mixture.window_of(r["step"]) != window:
            continue
        for s in r["sources"]:
            acc[s["lane"]] = acc.get(s["lane"], 0) + s["n_tokens"]
    return acc


def run_training(seed=42, steps=None, ckpt_every=None, lr=0.01, crash_at=None,
                 resume=False, eligible=None, log=None, policy=None, should_stop=None):
    """Run (or resume) training. Returns (model, opt, ledger, selector).

    `policy` (ckpt_policy.CheckpointPolicy) decides when to checkpoint; when None a periodic
    policy is used with `ckpt_every` steps, defaulting to the Young/Daly-derived
    `ckpt_policy.DEMO_INTERVAL`. `should_stop` is an optional no-arg callable — when it returns
    True at a step boundary, a checkpoint is flushed at the current offset and `Interrupted`
    is raised (used by run_pipeline for just-in-time checkpoint-on-signal)."""
    from . import ckpt_policy
    steps = steps or mixture.total_steps()
    if policy is None:
        interval = ckpt_policy.DEMO_INTERVAL if ckpt_every is None else ckpt_every
        policy = ckpt_policy.CheckpointPolicy(mode="periodic", interval=interval, total_steps=steps)
    if eligible is None:
        _, eligible = setup(log=log)

    model = NanoLM(vocab=get_vocab(), d_model=64, n_layer=2, max_pos=64, seed=seed)
    opt = AdamW(model.p, lr=lr)
    ledger = Ledger()
    selector = opus.OpusSelector(model, eligible, seq_len=64)

    if resume:
        offset = checkpoint.latest_offset()
        assert offset is not None, "no checkpoint to resume from"
        meta = checkpoint.load(offset, model, opt)
        ledger.truncate_to(offset)   # drop any steps recorded past the checkpoint
        start = offset
        if log:
            log(f"run_resumed from_offset={offset} expected_next_hash={meta['next_batch_hash'][:16]}")
    else:
        ledger.reset()
        checkpoint.clear()   # fresh run: no stale checkpoints from a prior run
        start = 0

    for step in range(start, steps):
        lw = mixture.floored_weights(step)
        deficits = mixture.floor_deficits(
            _window_lane_tokens(ledger.read_consumption(), mixture.window_of(step)))
        selector.set_context(step, deficits)
        n0 = len(selector.records)
        b = batch.build_batch(step, seed, eligible, lw, seq_len=64, n_seqs=4,
                              selector=selector)
        opus_recs = selector.records[n0:]  # OPUS decisions made for this step
        loss, per_lane = train_step(model, opt, b)
        ledger.append_consumption(b, opus_recs)
        ledger.append_learning(step, loss, per_lane, lr)
        if log:
            log(f"step={step} stage={mixture.stage_at(step)['name']} loss={loss:.4f} "
                f"util={b['util']:.3f} batch={b['batch_hash'][:12]}")

        if policy.should_checkpoint(step + 1):
            # checkpoint offset = next step; record the hash resume must reproduce.
            # Computed with the CURRENT model state (== the state resume will reload), so
            # it captures OPUS decisions faithfully.
            nxt = _expected_next_batch(step + 1, seed, eligible, model, ledger)
            checkpoint.save(step + 1, seed, model, opt, mixture.stage_at(step)["name"], nxt)
            policy.retain()
            if log:
                log(f"[PASS] checkpoint_saved offset={step + 1}")

        # just-in-time: a stop signal flushes a checkpoint at the current offset (if the
        # policy didn't just save one) so at most this step's work is lost on resume.
        if should_stop is not None and should_stop():
            if not policy.should_checkpoint(step + 1):
                nxt = _expected_next_batch(step + 1, seed, eligible, model, ledger)
                checkpoint.save(step + 1, seed, model, opt, mixture.stage_at(step)["name"], nxt)
                policy.retain()
                if log:
                    log(f"[PASS] jit_checkpoint_saved offset={step + 1}")
            raise Interrupted(step + 1)

        if crash_at is not None and step == crash_at:
            raise Crash(f"deliberate crash at step {step}")

    return model, opt, ledger, selector


def _expected_next_batch(next_step, seed, eligible, model, ledger):
    """Hash of the batch that will run at next_step, built with the current model state and
    a throwaway OPUS selector (so it matches resume exactly, OPUS decisions included)."""
    if next_step >= mixture.total_steps():
        return ""
    lw = mixture.floored_weights(next_step)
    deficits = mixture.floor_deficits(
        _window_lane_tokens(ledger.read_consumption(), mixture.window_of(next_step)))
    tmp = opus.OpusSelector(model, eligible, seq_len=64)
    tmp.set_context(next_step, deficits)
    b = batch.build_batch(next_step, seed, eligible, lw, seq_len=64, n_seqs=4,
                          selector=tmp)
    return b["batch_hash"]


def get_vocab():
    from .tokenizer import get_tokenizer
    return get_tokenizer().vocab_size
