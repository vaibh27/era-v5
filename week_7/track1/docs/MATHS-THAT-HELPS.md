# Mathematics that attacks the limits (deep search, 2026-08-14)

Each plain-language limit (see `LIMITS-PLAIN.md`) mapped to established/relevant math,
with what it fixes and what stays hard. Verified against sources where cited.

## The headline reframe
The ML arithmetic-embedding line (AOE, the CRT folders, the forks here) effectively **reinvented the
Residue Number System but skipped ~50 years of RNS computer-arithmetic engineering** that already
solves several limits called walls above: magnitude comparison, overflow detection, and error
correction. Bringing those in — plus modern looped-transformer / Abelian-network learnability tools —
is itself the novel-for-ML move.

## Mapping: limit → math that helps → status

| # | Limit (plain) | Math that helps | What it does |
|---|---|---|---|
| 2 | residues can't tell size/order | **RNS diagonal function / core functions** | Exact magnitude comparison & sign detection *without* converting out of residues. **Limit dissolves (bounded).** |
| 6 | "exact OR size-aware, never both" | RNS residues (exact +,×) **+ diagonal function** (order) | You get **both at once** in the bounded regime. The "core wall" is only a wall for *unbounded*. |
| 1 | silently wraps past range | **Redundant RNS (RRNS)** + core function → **overflow/range detection**; **base extension** to grow range | Removes the *silent*: you can detect overflow and dynamically extend range. Bound itself stays (pigeonhole). |
| 8 | one slip kills long multiply (`p^(L²)`) | **Redundant RNS error-correcting codes** | Extra moduli detect up to `n−k+1` and correct `⌊(n−k)/2⌋` residue errors — a built-in ECC. **Directly counters** the compounding fragility of an imperfect learned atom. |
| 9 | big base (256) needs a huge table | **Use RNS small-prime moduli instead of positional byte-limbs** | Each modulus is small (e.g. mod 5,7,11,13), so each multiply table is *tiny* and per-channel — the `base²` blow-up disappears. |
| 7 | `[x, log x]` desyncs after one op | **Logarithmic Number System (LNS)** + **Gaussian logarithm** (log-sum-exp) | Addition in log-domain is a *defined, tabulated* function φ⁺; the "channel-coherence" problem is exactly what LNS solved decades ago. |
| 5 | log breaks on 0 / negatives | LNS (explicit sign bit; special zero code); **complex-log** (sign = phase) | Standard handling; zero remains a flagged special case. |
| 10,11 | it's memorization; learning the structure fails (~21%) | **Abelian Neural Networks** (`x∘y=φ⁻¹(φ(x)+φ(y))`, invertible φ, *universal approximator* of abelian group ops); equivariant / group-homomorphism priors | An inductive bias that makes gradient descent *discover the coordinate system where the op is addition* — the learnable version of "find the log/CRT structure." (NIF builds on this.) |
| 12 | needs step-by-step (carry ripple) | **Looped / recurrent-depth Transformers** + **RASP-L / n-RASP-L** | Adaptive loop count scales computation with input length → *learned* iterative algorithms that length-generalize (e.g. addition trained ≤11 digits → 16+). Replaces the hand-wired schoolbook loop with a learned one. |
| 3 | only works if you already know the number | (representation vs. computation) — addressed by the looped model *computing* residues, not by a static code | Partial: the looped controller can produce residues from a reasoning process. |
| 4 | log loses exact digits | Use **RNS for exact**; reserve LNS for the deliberately-approximate magnitude channel | Not a fix — a division of labor (exact channel vs. size channel). |
| 13 | "all of mathematics" (undecidable) | — | Fundamental wall (Hilbert's 10th / Gödel). No fix; state the boundary. |
| 14 | text+math integration untested | engineering: the report's Hybrid embedding + a looped model | Buildable; just not run yet. |

## The four big reframes (what actually changed)

1. **"Residues are size-blind" (limit 2) is solvable with known math (not done here).** The **diagonal / core function** maps RNS →
   integers monotonically, giving exact magnitude comparison and sign detection without leaving
   residue form. So limit 6's "core wall" collapses in the bounded regime: **RNS + diagonal function
   = exact `+`,`×` AND ordering simultaneously.**

2. **The headline fragility (limit 8, `p^(L²)`) has a classical antidote: Redundant RNS.** Add a few
   redundant moduli and you get an error-correcting code over the residues — an imperfect learned
   atom's occasional slips get *detected and corrected* instead of compounding. This is the single
   most promising import for the learnability program.

3. **The big-table wall (limit 9) is an artifact of using base-256 *positional* limbs.** Switch to
   RNS *small-prime* moduli and every per-channel multiply table is tiny — no `base²` ROM. The folders'
   limb track chose the representation that maximizes the table; RNS chooses the one that minimizes it.

4. **Learnability + recurrence (limits 10–12) now have concrete tools.** *Abelian Networks* give a
   proven inductive bias for discovering "operation = addition in learned coordinates" (attacks the
   21% discovery failure); *looped transformers* give a learned, length-generalizing replacement for
   the hand-wired schoolbook loop (attacks the "the algorithm was coded by hand" critique).

## What stays genuinely hard (no math rescues these)
- **Unbounded *exact* in fixed size** (limit 1 core) — pigeonhole; needs growing resource (more moduli
  / recurrence / time). p-adics/profinite don't escape it (still need ~log(n) digits).
- **Undecidability** (limit 13) — fundamental.
- **Whether the learnability tools actually work *here*** — Abelian-net + looped-transformer + RRNS is a
  *hypothesis stack*, unproven on this problem. That's the experiment.

## The composed design this suggests
Exact **RNS with small primes** (tiny per-channel tables) **+ redundant moduli** (error correction &
overflow) **+ a diagonal-function channel** (magnitude/order) **+ an LNS magnitude register** (approx
reasoning) as the representation; a **looped/recurrent-depth transformer** with an **Abelian-network
invertible-φ** inductive bias as the model that *learns to run* the residue arithmetic rather than
having it hand-wired. This plugs limits 2,5,6,7,8,9 (math) and targets 10,11,12 (learning).

## Verified sources
- Abelian Neural Networks — arXiv **2102.12232** (universal approx of abelian group ops via invertible φ).
- Looped Transformers for Length Generalization — arXiv **2409.15647** (Fan, Du, Ramchandran, Lee; ICLR 2025; RASP-L, adaptive loop count).
- Neural Isomorphic Fields — arXiv **2601.12095** (relates to Abelian-net; an earlier verified cite).
- RNS diagonal/core function — computer-arithmetic literature (e.g. *Information Processing Letters* diagonal-function papers; MDPI hardware-diagonal-function). Magnitude comparison & sign detection without conversion.
- Redundant RNS error correction — RRNS(n,k): detect `n−k+1`, correct `⌊(n−k)/2⌋` (standard result; John D. Cook overview; IEEE literature).
- Logarithmic Number System / Gaussian logarithm — Parhami, "Computing with LNS arithmetic"; log-sum-exp φ⁺/φ⁻ addition.
