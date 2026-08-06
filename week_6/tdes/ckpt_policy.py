"""Checkpoint policy — WHEN to checkpoint, decoupled from HOW (checkpoint.py).

Two levers, straight from the literature, replace the old magic `ckpt_every=8`:

  * **periodic**, but sized by the **Young/Daly** optimum `T* ≈ sqrt(2·C·MTBF)` instead of a
    constant — C = checkpoint write cost, MTBF = mean time between failures. It balances
    write overhead (too frequent) against expected lost work on a crash (too rare). We derive
    it from *measured* cost/step-time and an assumed MTBF, and log the derivation, so the
    interval is justified rather than guessed.
  * **just-in-time (jit_only)** — don't checkpoint on a timer at all; persist only on a
    stop/failure signal (see `run_pipeline.py`) plus the final step. Near-zero steady-state
    overhead; a hard crash (no signal) simply replays from the last committed offset.

Plus **keep-last-N** retention so disk stays bounded.

References: Young (1974) / Daly (2006) optimal interval; "Just-In-Time Checkpointing"
(Gupta et al., EuroSys 2024); convergence-aware placement (COCI, 2024).
"""
import math

from . import checkpoint


def youngdaly_interval_steps(ckpt_cost_s, step_time_s, mtbf_s, lo=1, hi=None):
    """Young/Daly optimal checkpoint interval, expressed in STEPS.

    Compute the time-domain optimum `sqrt(2·C·MTBF)` (seconds) and convert to steps by
    dividing by the per-step wall time. Clamped to [lo, hi]. Degenerate inputs -> `lo`.
    """
    if step_time_s <= 0 or mtbf_s <= 0:
        return lo
    t_star_s = math.sqrt(2.0 * max(ckpt_cost_s, 0.0) * mtbf_s)
    steps = max(lo, round(t_star_s / step_time_s))
    if hi is not None:
        steps = min(steps, int(hi))
    return int(steps)


# The demo derives its interval via Young/Daly from these FIXED, representative constants
# (not machine-measured), so run_demo stays byte-reproducible while still being principled —
# the interval is a logged function of assumed cost/failure-rate, not a magic number.
# run_pipeline.py, by contrast, measures C and step time live. sqrt(2*0.02*150)/0.25 ≈ 9.8 -> 10.
DEMO_CKPT_COST_S = 0.02
DEMO_STEP_TIME_S = 0.25
DEMO_MTBF_S = 150.0
DEMO_INTERVAL = youngdaly_interval_steps(DEMO_CKPT_COST_S, DEMO_STEP_TIME_S, DEMO_MTBF_S)


class CheckpointPolicy:
    """Decides whether to checkpoint after a completed step, and handles retention.

    mode="periodic": checkpoint every `interval` steps (interval may be Young/Daly-derived).
    mode="jit_only": never checkpoint on a timer — only the final step and the JIT signal
                     flush (in the runner) persist state.
    The final step is always checkpointed so a completed run is fully recoverable.
    keep_last=None keeps every checkpoint (the demo default, whose proofs read old ones).
    """

    def __init__(self, mode="periodic", interval=8, total_steps=None, keep_last=None):
        assert mode in ("periodic", "jit_only")
        self.mode = mode
        self.interval = max(1, int(interval))
        self.total_steps = total_steps
        self.keep_last = keep_last

    def should_checkpoint(self, next_offset):
        """next_offset = step + 1 = number of completed steps."""
        if self.total_steps is not None and next_offset >= self.total_steps:
            return True  # always persist final state
        if self.mode == "jit_only":
            return False
        return next_offset % self.interval == 0

    def retain(self):
        checkpoint.keep_last_n(self.keep_last)

    def describe(self):
        return {"mode": self.mode, "interval": self.interval, "keep_last": self.keep_last}
