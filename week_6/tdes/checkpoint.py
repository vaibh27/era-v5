"""Checkpoints tied to ledger offsets, with a two-tier (RAM + disk) read path.

A checkpoint stores model params, AdamW optimizer state, and the ledger offset (== the
number of steps consumed == the next step index). Because every batch is a pure function
of (seed, step), the RNG needs no saving: resume rebuilds batch #offset deterministically.
That offset linkage is what lets resume prove "next batch is exactly the expected batch".

Three production techniques, honestly scoped to a single-process numpy run:

* **Atomic writes** — payload and metadata go to temp files and are `os.replace`d into place
  (atomic rename). A crash mid-write never leaves a half-written checkpoint; the `.json`
  (which `latest_offset` keys on) appears only once both files are durably flushed.
* **In-memory tier (Gemini-style, arXiv/SOSP'23)** — `save` also caches the checkpoint's
  arrays in process RAM (as copies — AdamW updates params in place, so references would be
  corrupted). `load` serves from RAM when present, avoiding a disk read. This accelerates
  *in-process* recovery (e.g. the same-process crash→resume proof) but, being process-local,
  does NOT survive a real process kill — that path falls back to the durable disk copy.
* **Async writes (DataStates/CheckFreq-style)** — with `async_=True`, the disk write happens
  on a background thread while the RAM tier makes the checkpoint immediately readable; call
  `join_writes()` before relying on durability. At nano scale the latency saved is tiny; it
  demonstrates the mechanism that hides multi-GB checkpoint stalls at real scale.
"""
import json
import os
import threading

import numpy as np

from . import paths

_RAM = {}          # offset -> {"arrs": {key: ndarray copy}, "meta": {...}}
_WRITERS = []      # in-flight async write threads


def _atomic_write_bytes(path, write_fn):
    """Write via a temp file in the same dir, then atomically replace `path`."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_disk(offset, arrs, meta):
    # payload first, metadata last: latest_offset() globs the .json, so it must not appear
    # until its .npz is durably on disk.
    npz = paths.CHECKPOINTS / f"ckpt_{offset:06d}.npz"
    _atomic_write_bytes(npz, lambda f: np.savez(f, **arrs))
    meta_json = paths.CHECKPOINTS / f"ckpt_{offset:06d}.json"
    _atomic_write_bytes(meta_json, lambda f: f.write(json.dumps(meta, indent=2).encode("utf-8")))


def save(offset, seed, model, opt, stage, next_batch_hash, async_=False, in_memory=True):
    """offset = next step index. next_batch_hash = deterministic hash of the batch that
    resume must reproduce (recorded so the resume proof is self-contained)."""
    paths.ensure_dirs()
    arrs = {}
    for k, v in model.p.items():
        arrs[f"p::{k}"] = np.array(v)   # copy: decouple from live (in-place updated) params
    for k, v in opt.m.items():
        arrs[f"m::{k}"] = np.array(v)
    for k, v in opt.v.items():
        arrs[f"v::{k}"] = np.array(v)
    meta = {"offset": offset, "seed": seed, "adam_t": opt.t, "stage": stage,
            "next_batch_hash": next_batch_hash}
    if in_memory:
        _RAM[offset] = {"arrs": arrs, "meta": meta}
    if async_:
        t = threading.Thread(target=_write_disk, args=(offset, arrs, meta), daemon=True)
        t.start()
        _WRITERS.append(t)
    else:
        _write_disk(offset, arrs, meta)
    return offset


def join_writes():
    """Block until all async checkpoint writes have landed on disk (call before relying on
    durability, e.g. at the end of a run or before a cross-process resume)."""
    for t in _WRITERS:
        t.join()
    _WRITERS.clear()


def keep_last_n(n):
    """Retention: keep only the newest `n` checkpoints (by offset), deleting older
    .npz/.json pairs (and their RAM-tier entries) so storage stays bounded. n=None keeps
    everything (the demo's default, since its resume/replay proofs read older checkpoints)."""
    if n is None:
        return
    metas = sorted(paths.CHECKPOINTS.glob("ckpt_*.json"))
    for old in metas[:-n] if n > 0 else metas:
        stem = old.stem  # e.g. ckpt_000008
        old.unlink(missing_ok=True)
        (paths.CHECKPOINTS / f"{stem}.npz").unlink(missing_ok=True)
        _RAM.pop(int(stem.split("_")[1]), None)


def clear():
    """Remove all checkpoints (disk + RAM tier) — used at the start of a fresh (non-resume)
    run so a prior run's checkpoints can't be mistaken for this run's latest."""
    paths.ensure_dirs()
    join_writes()
    for p in list(paths.CHECKPOINTS.glob("ckpt_*.npz")) + list(paths.CHECKPOINTS.glob("ckpt_*.json")):
        p.unlink()
    _RAM.clear()


def latest_offset():
    metas = sorted(paths.CHECKPOINTS.glob("ckpt_*.json"))
    if not metas:
        return None
    return json.loads(metas[-1].read_text())["offset"]


def loaded_from_ram(offset):
    """True if `offset` would be served from the in-memory tier (no disk read)."""
    return offset in _RAM


def load(offset, model, opt):
    """Restore model+optimizer at `offset`. Reads the fast in-memory tier when present,
    else the durable disk copy. Arrays are copied into the live params (never aliased)."""
    if offset in _RAM:
        arrs, meta = _RAM[offset]["arrs"], _RAM[offset]["meta"]
        for k in list(model.p.keys()):
            model.p[k] = np.array(arrs[f"p::{k}"])
            opt.m[k] = np.array(arrs[f"m::{k}"])
            opt.v[k] = np.array(arrs[f"v::{k}"])
        opt.t = meta["adam_t"]
        return meta
    npz = np.load(paths.CHECKPOINTS / f"ckpt_{offset:06d}.npz")
    for k in list(model.p.keys()):
        model.p[k] = npz[f"p::{k}"]
        opt.m[k] = npz[f"m::{k}"]
        opt.v[k] = npz[f"v::{k}"]
    meta = json.loads((paths.CHECKPOINTS / f"ckpt_{offset:06d}.json").read_text())
    opt.t = meta["adam_t"]
    return meta
