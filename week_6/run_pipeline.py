"""Resumable training pipeline — the "real job", distinct from the reproducible demo.

Where `run_demo.py` always regenerates the graded `submission_artifacts/` from scratch,
`run_pipeline.py` is the operational flow you actually run and can interrupt:

    python run_pipeline.py                # run to completion (checkpointing as it goes)
    <Ctrl-C or kill>                      # flushes a just-in-time checkpoint, exits
    python run_pipeline.py                # resumes from that checkpoint, finishes the run

Because every batch is a pure function of (seed, step), a killed-then-resumed run converges
to the *same* final state as an uninterrupted one — resuming loses at most the in-flight
step. It writes to its own `pipeline_artifacts/` tree so it never disturbs the demo.

Checkpoint strategy (see tdes/ckpt_policy.py):
  --mode periodic   Young/Daly-sized interval, derived from a measured step time + checkpoint
                    cost and an assumed --mtbf (logged), not a magic constant.
  --mode jit        just-in-time only: checkpoint on the stop signal + final step; near-zero
                    steady-state overhead (a hard crash replays from the last committed step).
  --keep-last N     retain only the newest N checkpoints.

References: Young/Daly optimal interval; Just-In-Time Checkpointing (EuroSys 2024).
"""
import argparse
import signal
import sys
import time

from tdes import checkpoint, mixture, paths, train
from tdes.ckpt_policy import CheckpointPolicy, youngdaly_interval_steps
from tdes.ledger import Ledger


class Logger:
    def __init__(self):
        paths.ensure_dirs()
        self.f = open(paths.RUN_LOG, "w", encoding="utf-8")

    def __call__(self, msg):
        self.f.write(msg + "\n")
        self.f.flush()
        print(msg)

    def close(self):
        self.f.close()


def _calibrate(eligible, seed, steps, mtbf_s, log):
    """Measure per-step time and checkpoint cost (non-destructively), then derive the
    Young/Daly interval. Uses a throwaway model so real checkpoints are untouched."""
    from tdes import batch, opus
    from tdes.model import NanoLM
    model = NanoLM(vocab=train.get_vocab(), d_model=64, n_layer=2, max_pos=64, seed=seed)
    opt = train.AdamW(model.p, lr=0.01)
    sel = opus.OpusSelector(model, eligible, seq_len=64)
    sel.set_context(0, {})
    b = batch.build_batch(0, seed, eligible, mixture.floored_weights(0),
                          seq_len=64, n_seqs=4, selector=sel)
    t = time.time(); train.train_step(model, opt, b); step_time = time.time() - t
    t = time.time(); checkpoint.save(0, seed, model, opt, "calibration", ""); ckpt_cost = time.time() - t
    # remove only the calibration checkpoint (offset 0); leave any real checkpoints intact
    (paths.CHECKPOINTS / "ckpt_000000.npz").unlink(missing_ok=True)
    (paths.CHECKPOINTS / "ckpt_000000.json").unlink(missing_ok=True)
    interval = youngdaly_interval_steps(ckpt_cost, step_time, mtbf_s, lo=1, hi=steps)
    log(f"calibration step_time={step_time * 1000:.1f}ms checkpoint_cost={ckpt_cost * 1000:.1f}ms "
        f"assumed_mtbf={mtbf_s}s")
    log(f"youngdaly_interval = round(sqrt(2 * {ckpt_cost:.4f}s * {mtbf_s}s) / {step_time:.4f}s) "
        f"= {interval} steps")
    return interval


def _decide_resume(latest, total, args, log):
    """Resume vs fresh. Flags win; else prompt if interactive; else resume an incomplete run."""
    if latest is None:
        return False
    incomplete = latest < total
    if args.resume:
        return True
    if args.fresh:
        return False
    state = (f"an INCOMPLETE run at offset {latest}/{total}" if incomplete
             else f"a COMPLETED run (offset {latest})")
    if sys.stdin.isatty():
        ans = input(f"Found {state}. [R]esume or [f]resh start? [R/f] ").strip().lower()
        return not ans.startswith("f")
    # non-interactive: resume an incomplete run, otherwise start fresh
    log(f"non-interactive; found {state} -> {'resume' if incomplete else 'fresh'}")
    return incomplete


def main():
    ap = argparse.ArgumentParser(description="Resumable TDES training pipeline.")
    ap.add_argument("--mode", choices=["periodic", "jit"], default="periodic")
    ap.add_argument("--mtbf", type=float, default=30.0,
                    help="assumed mean-time-between-failures (s) for Young/Daly sizing")
    ap.add_argument("--keep-last", type=int, default=None, help="retain only newest N checkpoints")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true", help="force a fresh start")
    ap.add_argument("--resume", action="store_true", help="force resume from latest checkpoint")
    ap.add_argument("--crash-at", type=int, default=None, help="(testing) raise a deliberate crash")
    args = ap.parse_args()

    paths.use_artifacts_dir(paths.ROOT / "pipeline_artifacts")
    log = Logger()
    log("=== TDES pipeline (resumable) ===")

    _, eligible = train.setup(log=log)
    steps = args.steps or mixture.total_steps()

    latest = checkpoint.latest_offset()
    resume = _decide_resume(latest, steps, args, log)

    if args.mode == "jit":
        policy = CheckpointPolicy(mode="jit_only", total_steps=steps, keep_last=args.keep_last)
        log("checkpoint_policy mode=jit_only (checkpoint on signal + final step only)")
    else:
        interval = _calibrate(eligible, args.seed, steps, args.mtbf, log)
        policy = CheckpointPolicy(mode="periodic", interval=interval,
                                  total_steps=steps, keep_last=args.keep_last)
    log(f"checkpoint_policy {policy.describe()} resume={resume}")

    # just-in-time: a stop signal sets a flag; run_training flushes at the next step boundary
    stop = {"flag": False}

    def _on_signal(signum, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    log(f"-- training{' (resuming)' if resume else ''} --")
    try:
        train.run_training(seed=args.seed, steps=steps, resume=resume, eligible=eligible,
                           log=log, policy=policy, should_stop=lambda: stop["flag"],
                           crash_at=args.crash_at)
    except train.Interrupted as e:
        log(f"[PASS] interrupted_checkpoint_saved offset={e.offset} (re-run to resume)")
        log.close()
        return 130

    final = Ledger().read_consumption()
    log(f"[PASS] pipeline_complete steps={len(final)} "
        f"final_batch={final[-1]['batch_hash'][:12] if final else 'n/a'}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
