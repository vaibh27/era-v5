# The limits, in plain language — what fails and why

Every limit found across the report, the CRT/limb folders, and the experiments.
Each: the limit → an example that breaks → why it breaks. No jargon.

## A) The exact "remainder / clock" number code (CRT/RNS)

**1. It silently wraps around past its range.**
- Example: with the small setup (remainders by 5, 7, 11, good up to 385), ask `20 × 20 = 400`. It confidently returns **15**, not 400.
- Why: it only tracks remainders, like a clock. 3 hours past 11 o'clock is "2 o'clock," not "14." Once the true answer passes the limit, it can't tell 400 from 15 — and gives no warning.

**2. It has no sense of size or order.**
- Example: "is 100 bigger than 99?" As remainder-fingerprints, 100 = (0,2,1) and 99 = (4,1,0). Nothing there says 100 > 99; 99 and 100 look as unrelated as 99 and 6.
- Why: remainders are a fingerprint, not a ruler. Great for exact +/×, useless for "which is bigger" or "is this answer even plausible."

**3. It only works if you already know the number.**
- Example: a word problem where the model must *reason out* that the answer is 63 — the code can't help it get there; it can only bookkeep 63 once you already have it.
- Why: encoding means "compute the remainders of n," which needs n. It's a calculator you can only use *after* you've found the number.

## B) The "size / log" number code (the fix for size-blindness)

**4. It loses the exact digits.**
- Example: `12345 × 67890 = 838,102,050`. This code says "about 8.38 × 10⁸" — right size, right first few digits, but it can't tell you whether the answer ends in …050 or …051.
- Why: a fixed number line squeezes infinitely many big numbers into finite precision, so nearby huge numbers round to the same spot. Good for "how big," bad for "exactly which."

**5. It breaks on zero and negatives.**
- Example: `5 × 0` or `-3 × 4` — the log of 0 is undefined and the log of a negative isn't a normal number, so both need special patches bolted on.
- Why: the log trick only works for positive numbers; zero and negatives are exceptions.

## C) The core wall — you can't get both

**6. Exact OR size-aware, never both in one code.**
- Example: you want `emb(81)` to (a) decode back to exactly 81 AND (b) know 81 > 80. The remainder code does (a) not (b); the log code does (b) not (a). No single fixed-size vector does both.
- Why: addition wants numbers spaced evenly on a line; multiplication wants them arranged by their factors. Those two layouts are incompatible, so one vector can't be arranged both ways at once.

**7. Gluing value + log together desyncs after one step.**
- Example: store 9 as (9, log 9). Add two of them: the value half says **18** (correct); the log half says log 9 + log 9 = log **81**. After one addition the two halves disagree.
- Why: adding the value part and adding the log part compute different things; keeping them consistent needs an expensive re-sync after *every* operation.

## D) The digit-by-digit (limb / schoolbook) approach

**8. Long multiplication collapses — one slip kills the whole answer.**
- Example: multiplying two 30-digit numbers needs 900 single-digit products, and every one must be right. Even at 99.9% per digit, all-900-correct is only ~40%. Push to ~55 digits and it's essentially 0%.
- Why: multiplication cross-multiplies every digit by every digit, and the final number is wrong if even one is off. Errors don't average out — they pile up.

**9. A big digit-alphabet needs a huge memorized table.**
- Example: using bytes (256 "digits", to match Kronecker), a small network gets only 22% of single-digit products right — so any real multiply is hopeless. You must grow it to ~600K parameters just to memorize the 256×256 table.
- Why: the network doesn't "understand" multiplication, it memorizes the table — and the table grows with the square of the alphabet size.

**10. It's a lookup table + a hand-written algorithm, not learned reasoning.**
- Example: the 100%-accurate multiplier is really a memorized single-digit table wrapped in the long-multiplication steps written by hand in Python. Remove that hand-written loop and give the network the two whole numbers directly → it drops to ~21%.
- Why: the actual "multiplying" is the algorithm coded by hand; the network is just the digit lookup. Take away the scaffolding and there's no intelligence left.

**11. Learning the structure from scratch fails.**
- Example: train a plain embedding to "make emb(a)×emb(b) = emb(a×b)" from many examples → it reaches only ~21%. It never discovers the clean structure that would make it exact; it memorizes some and flails on the rest.
- Why: the exact structure is a needle in a haystack — ordinary training doesn't naturally push toward it. (Whether a smarter training setup can is the open question to attack.)

**12. It can't be done in one shot — it needs step-by-step.**
- Example: `999999 + 1 = 1000000` — the carry ripples through all six digits. A single fixed operation can't push a carry across an unknown number of positions.
- Why: carrying is inherently sequential (each carry feeds the next), so you need a loop / step-by-step computation, not one fixed move. This is why models need chain-of-thought.

## E) The grand vision

**13. "Describe all of mathematics" is impossible.**
- Example: "does this whole-number equation have a solution?" (a Diophantine equation) — there's no general procedure that always answers, for any embedding.
- Why: exact whole-number math contains genuinely unanswerable questions. You can encode the symbols, but you can't guarantee an answer to every question.

## F) The unification (text + math)

**14. The "append math to word embeddings" piece is built but never tested.**
- Example: give a small model "nine times nine" in words — does it route the numbers into the math part, compute, and still handle the sentence? And do "nine" and "9" map to the same number? Never run.
- Why: the code exists, but the experiment showing math helps *without hurting* language has not been done — so it's an untested promise, not a demonstrated ability.

---
### The one-sentence summary
The exact approaches work only inside a fixed range and can't sense size; the size-aware approach can't be exact; you can't have both in one code; and the only thing that does unbounded exact arithmetic (digit-by-digit) works by memorizing a lookup table and running an algorithm written by hand — the moment the network is asked to supply the reasoning itself, it fails (~21%).
