"""Mixture schedule + protected floors — the "what data, in what proportion, when" plane.

Two controls, straight from week5 §40:
  - a per-stage lane-weight schedule (the curriculum): weights shift across training
  - per-window protected floors: minimum lane shares the online selector may never cross,
    enforced per WINDOW of steps (a single batch may dip; a window may not)

The floor is the guardrail the OPUS selector optimizes *within*: OPUS may compress an
abundant lane chasing utility, but a window that would breach a floor forces the deficit
lane to be drawn before advancing (see opus.floor_override).
"""

# Lanes present in the nano corpus.
LANES = ["web", "indic", "code", "math"]

# Per-stage lane weights (fractions per stage). Trajectory mirrors week5: web recedes,
# indic climbs, code/math peak mid-run. Scaled to the 4 nano lanes (renormalized).
STAGES = [
    {"name": "S0", "steps": 8,  "weights": {"web": 0.55, "indic": 0.15, "code": 0.20, "math": 0.10}},
    {"name": "S1", "steps": 12, "weights": {"web": 0.45, "indic": 0.25, "code": 0.20, "math": 0.10}},
    {"name": "S2", "steps": 12, "weights": {"web": 0.35, "indic": 0.35, "code": 0.20, "math": 0.10}},
]

# Per-window protected floors (minimum share of loss-bearing tokens per window).
# indic is the identity lane -> strongest floor; code+math kept alive as a combined carrier.
FLOORS = {"indic": 0.20}
COMBINED_FLOORS = [({"code", "math"}, 0.20)]

# Enforcement aims a margin ABOVE the true floor (draw weighting + deficit detection), so
# small per-window sampling/OPUS variance still lands the actual share above the floor the
# audit checks. The audit verifies the true FLOORS above; this margin only affects targeting.
ENFORCE_MARGIN = 0.04

WINDOW = 8  # steps per floor-enforcement window


def total_steps():
    return sum(s["steps"] for s in STAGES)


def stage_at(step):
    """Return the stage dict active at a given step."""
    acc = 0
    for s in STAGES:
        if step < acc + s["steps"]:
            return s
        acc += s["steps"]
    return STAGES[-1]


def weights_at(step):
    """Planned lane weights at a step (normalized)."""
    w = dict(stage_at(step)["weights"])
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def floored_weights(step):
    """Planned weights with protected floors applied proactively (effective weight =
    max(planned, floor)), renormalized. Ensures floor lanes are drawn enough that the
    per-window floor is met; OPUS floor-override is the secondary guarantee."""
    w = dict(weights_at(step))
    for lane, fl in FLOORS.items():
        w[lane] = max(w.get(lane, 0), fl + ENFORCE_MARGIN)
    for lanes, fl in COMBINED_FLOORS:
        s = sum(w.get(l, 0) for l in lanes)
        if 0 < s < fl + ENFORCE_MARGIN:
            for l in lanes:
                w[l] = w.get(l, 0) * (fl + ENFORCE_MARGIN) / s
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def window_of(step):
    return step // WINDOW


def floor_deficits(window_lane_tokens):
    """Given {lane: loss_tokens} accumulated so far in the current window, return
    {lane: shortfall_fraction} for any floor not yet met. Used to force floor draws."""
    total = sum(window_lane_tokens.values())
    if total == 0:
        return {}
    deficits = {}
    for lane, floor in FLOORS.items():
        share = window_lane_tokens.get(lane, 0) / total
        if share < floor + ENFORCE_MARGIN:
            deficits[lane] = floor - share
    for lanes, floor in COMBINED_FLOORS:
        share = sum(window_lane_tokens.get(l, 0) for l in lanes) / total
        if share < floor + ENFORCE_MARGIN:
            # attribute the deficit to the scarcest member of the group
            scarcest = min(lanes, key=lambda l: window_lane_tokens.get(l, 0))
            deficits[scarcest] = max(deficits.get(scarcest, 0), floor - share)
    return deficits


def compile_schedule():
    """Human/machine-readable compiled schedule for evidence."""
    return {
        "lanes": LANES,
        "stages": STAGES,
        "floors": FLOORS,
        "combined_floors": [{"lanes": sorted(ls), "floor": f} for ls, f in COMBINED_FLOORS],
        "window": WINDOW,
        "total_steps": total_steps(),
    }
