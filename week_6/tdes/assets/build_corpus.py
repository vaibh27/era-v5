"""One-time frozen-asset builder for the demo corpus (NOT run by run_demo.py).

Produces `corpus.jsonl` — the immutable input to the pipeline. Provenance:
  - web lane   : real general-web docs from week4/work/sample.jsonl (source=cc, raw
                 multilingual CommonCrawl boilerplate; text capped to WEB_MAXCHARS)
  - indic lane : real hi/ta/te docs from week4/work/sample.jsonl (source=indiccorp)
  - code lane  : deterministically generated tiny Python snippets
  - math lane  : deterministically generated worked arithmetic problems

Determinism: fixed selection order + fixed generation templates + a fixed RNG seed,
so re-running reproduces byte-identical corpus.jsonl. Splits: every 8th doc within a
lane is held out as `eval` (never enters a loss-bearing batch — see firewall.py).

Run manually to regenerate:  python tdes/assets/build_corpus.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # week6/ on path
from tdes.tokenizer import get_tokenizer  # noqa: E402

HERE = Path(__file__).parent
WEEK4 = HERE / ".." / ".." / ".." / "week4" / "work" / "sample.jsonl"
OUT = HERE / "corpus.jsonl"

NEAR = 0.7            # keep a doc only if its token-5gram Jaccard to every kept doc < NEAR
                      # (stricter than firewall's 0.8, so decontamination passes with margin)

PER_WEB = 80          # English web docs
PER_INDIC_LANG = 40   # per hi/ta/te -> 120 indic docs
N_CODE = 70
N_MATH = 70
MIN_CHARS = 24        # skip trivially short real docs
WEB_MAXCHARS = 400    # cap messy web docs to keep them nano-sized
EVAL_EVERY = 8        # every 8th doc in a lane -> eval split


def take_real():
    """Single deterministic pass: first PER_WEB web (source=cc) + PER_INDIC_LANG each hi/ta/te."""
    web = []
    indic = {"hi": [], "ta": [], "te": []}
    with open(WEEK4, encoding="utf-8") as f:
        for line in f:
            if len(web) >= PER_WEB and all(len(v) >= PER_INDIC_LANG for v in indic.values()):
                break
            d = json.loads(line)
            t = d.get("text", "").strip()
            if len(t) < MIN_CHARS:
                continue
            if d.get("source") == "cc" and len(web) < PER_WEB:
                web.append(t[:WEB_MAXCHARS])
            lg = d.get("lang")
            if lg in indic and len(indic[lg]) < PER_INDIC_LANG:
                indic[lg].append(t)
    return web, indic


FN_NAMES = ["add", "sub", "mul", "mod", "maxv", "minv", "square", "cube", "negate",
            "absval", "inc", "dec", "average", "double", "half", "clamp", "scale",
            "diff", "total", "power", "combine", "reduce_sum", "shift", "blend"]
VARS = ["x", "y", "a", "b", "n", "m", "p", "q", "val", "acc", "lo", "hi", "idx", "cnt", "s", "t"]
OPS = ["+", "-", "*", "%", "//"]


def gen_code(n, rng):
    """Diverse tiny Python snippets across several structural templates."""
    docs = []
    for _ in range(n):
        fn = rng.choice(FN_NAMES)
        a, b, c = rng.sample(VARS, 3)
        op, op2 = rng.choice(OPS), rng.choice(OPS)
        k = rng.randint(2, 20)
        tmpl = rng.randint(0, 4)
        if tmpl == 0:
            body = f"def {fn}({a}, {b}):\n    return {a} {op} {b}"
        elif tmpl == 1:
            body = f"def {fn}({a}):\n    return {a} {op} {k}"
        elif tmpl == 2:
            body = f"def {fn}({a}, {b}):\n    {c} = {a} {op} {b}\n    return {c}"
        elif tmpl == 3:
            body = f"def {fn}({a}, {b}, {c}):\n    return {a} {op} {b} {op2} {c}"
        else:
            body = (f"def {fn}(items):\n    {a} = 0\n    for {b} in items:\n"
                    f"        {a} = {a} {op} {b}\n    return {a}")
        if rng.random() < 0.5:
            body += f"\nprint({fn}({rng.randint(1, 99)}, {rng.randint(1, 99)}))"
        docs.append(body)
    return docs


NAMES = ["Asha", "Ravi", "Meera", "Arjun", "Priya", "Kabir", "Sana", "Dev", "Nisha", "Rahul"]
OBJS = ["apples", "books", "coins", "pens", "mangoes", "cards", "seeds", "stamps", "beads"]


def gen_math(n, rng):
    """Diverse worked arithmetic across several phrasings and one/two-step problems."""
    docs = []
    for _ in range(n):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["+", "-", "*"])
        v = {"+": a + b, "-": a - b, "*": a * b}[op]
        tmpl = rng.randint(0, 4)
        if tmpl == 0:
            t = f"Q: What is {a} {op} {b}? Step: compute {a} {op} {b} = {v}. Answer: {v}."
        elif tmpl == 1:
            t = f"Compute {a} {op} {b}. Working: {a} {op} {b} equals {v}. Result: {v}."
        elif tmpl == 2 and op == "+":
            nm, ob = rng.choice(NAMES), rng.choice(OBJS)
            t = f"{nm} has {a} {ob} and gets {b} more, so {nm} now has {v} {ob}."
        elif tmpl == 3 and op == "*":
            ob = rng.choice(OBJS)
            t = f"A box holds {a} {ob}. Across {b} boxes there are {v} {ob} in total."
        else:
            c = rng.randint(2, 20)
            v2 = v + c
            t = f"Two-step: ({a} {op} {b}) + {c} = {v} + {c} = {v2}."
        docs.append(t)
    return docs


def _shingles(tokens, k=5):
    if len(tokens) < k:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def _jaccard(a, b):
    return (len(a & b) / len(a | b)) if (a and b) else 0.0


def near_dedup(cands, target, tok, kept):
    """Greedy: keep a candidate only if its token-5gram Jaccard to every kept doc < NEAR.
    Guarantees the final corpus has no near-duplicate pair (train or eval, any lane)."""
    out = []
    for t in cands:
        sh = _shingles(tok.encode(t))
        if all(_jaccard(sh, k) < NEAR for k in kept):
            kept.append(sh)
            out.append(t)
            if len(out) >= target:
                break
    return out


def with_meta(lane, texts):
    """Assign doc_id + deterministic train/eval split."""
    out = []
    for i, t in enumerate(texts):
        split = "eval" if (i % EVAL_EVERY == EVAL_EVERY - 1) else "train"
        out.append({"doc_id": f"{lane}-{i:04d}", "lane": lane, "split": split, "text": t})
    return out


def main():
    rng = random.Random(20260804)  # fixed seed -> reproducible synthetic lanes
    tok = get_tokenizer()
    web, real_indic = take_real()
    indic = []
    for lg in ("hi", "ta", "te"):
        indic += real_indic[lg][:PER_INDIC_LANG]

    kept = []  # shared shingle memory -> cross-lane near-dup-free too
    lanes = {
        "web": near_dedup(web, PER_WEB, tok, kept),
        "indic": near_dedup(indic, PER_INDIC_LANG * 3, tok, kept),
        "code": near_dedup(gen_code(N_CODE * 6, rng), N_CODE, tok, kept),
        "math": near_dedup(gen_math(N_MATH * 6, rng), N_MATH, tok, kept),
    }

    docs = []
    for lane in ("web", "indic", "code", "math"):
        docs += with_meta(lane, lanes[lane])

    with open(OUT, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    from collections import Counter
    counts = Counter((d["lane"], d["split"]) for d in docs)
    print(f"wrote {len(docs)} docs -> {OUT}")
    for (lane, split), c in sorted(counts.items()):
        print(f"  {lane:6} {split:5} {c}")


if __name__ == "__main__":
    main()
