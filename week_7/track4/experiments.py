"""Problem 4 proofs: characters-as-waves (Fourier superposition) round-trips, keeps anagrams
distinct, has no 32-cap, and doesn't collide where V1 does. Writes plots/ + prints tables."""
import os, random, string
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from codec import FourierCharCodec, V1Codec, s2b, b2s

OUT = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT, exist_ok=True)
random.seed(0); np.random.seed(0)
ALPHA = string.ascii_letters + string.digits


def rword(L):
    return "".join(random.choice(ALPHA) for _ in range(L))


def char_acc(a, b):
    return sum(x == y for x, y in zip(a, b)) / max(len(a), 1)


# ---- Exp 1: round-trip accuracy vs length, for several D (dimension knob) ----
def exp1():
    lengths = list(range(1, 65))
    Ds = [512, 1024, 2048, 4096]
    trials = 15
    curves = {}
    for D in Ds:
        fc = FourierCharCodec(D=D)
        accs = []
        for L in lengths:
            a = np.mean([char_acc(b := s2b(rword(L)), fc.decode(fc.encode(b), L)) for _ in range(trials)])
            accs.append(a)
        curves[D] = accs
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for D in Ds:
        ax.plot(lengths, curves[D], lw=2, label=f"D={D} (real dim {2*D})")
    ax.axvline(32, color="red", ls=":", alpha=0.6, label="V1 hard cap = 32")
    ax.set_title("Characters-as-waves: exact round-trip while capacity lasts, then graceful\n(no hard 32 cap; bigger D = more characters before crosstalk)", fontsize=10, fontweight="bold")
    ax.set_xlabel("word length (chars)"); ax.set_ylabel("per-char round-trip accuracy")
    ax.set_ylim(0, 1.03); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "exp1_roundtrip.png"), dpi=120); plt.close(fig)
    print("[Exp1] round-trip accuracy at selected lengths")
    print(f"{'L':>4} | " + " | ".join(f"D={D:>4}" for D in Ds))
    for L in (5, 10, 20, 32, 40, 64):
        print(f"{L:4d} | " + " | ".join(f"{curves[D][L-1]:6.3f}" for D in Ds))


# ---- Exp 2: anagrams stay distinct (order carried by position phase) ----
def exp2():
    fc = FourierCharCodec(D=2048)
    print("\n[Exp2] anagrams are DISTINCT (position phase carries order)")
    for group in [["cat", "act", "tac"], ["listen", "silent", "enlist"], ["abc", "bca", "cab"]]:
        embs = {w: fc.encode(s2b(w)) for w in group}
        # all decode correctly
        ok = all(b2s(fc.decode(embs[w], len(w))) == w for w in group)
        # pairwise distances all clearly > 0
        dists = [np.linalg.norm(embs[a] - embs[b]) for i, a in enumerate(group) for b in group[i+1:]]
        print(f"  {group}: all decode OK={ok}, min pairwise L2={min(dists):.3f} "
              f"(a naive amplitude-sum would collide; position phase keeps them apart)")


# ---- Exp 3: no collisions where V1 crops (long shared-prefix words) ----
def exp3():
    fc = FourierCharCodec(D=4096)
    v1 = V1Codec()
    N = 200
    prefix = "a" * 32
    v1_coll = f_coll = 0
    for _ in range(N):
        s1 = prefix + rword(10); s2 = prefix + rword(10)
        if s1 == s2:
            continue
        b1, b2 = s2b(s1), s2b(s2)
        v1_coll += np.linalg.norm(v1.encode(b1) - v1.encode(b2)) < 1e-9
        f_coll += np.linalg.norm(fc.encode(b1) - fc.encode(b2)) < 1e-9
    print("\n[Exp3] distinct strings sharing a 32-byte prefix (where V1 crops)")
    print(f"  V1 (Kronecker) collision rate: {v1_coll/N:.0%}")
    print(f"  Fourier-chars collision rate : {f_coll/N:.0%}")


# ---- Exp 4: concrete long words, no cap ----
def exp4():
    fc = FourierCharCodec(D=4096)
    print("\n[Exp4] concrete words (D=4096, real dim 8192 = same as V1 but NO 32 cap)")
    for w in ["a", "supercalifragilisticexpialidocious",
              "pneumonoultramicroscopicsilicovolcanoconiosis"]:
        b = s2b(w); rec = b2s(fc.decode(fc.encode(b), len(b)))
        print(f"  len {len(w):2d}: {'OK' if rec == w else 'XX'}  {w[:45]}")


if __name__ == "__main__":
    exp1(); exp2(); exp3(); exp4()
    print("\nplots ->", OUT)
