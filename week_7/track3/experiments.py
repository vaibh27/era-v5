"""
Track 3 experiments: prove the Fourier-position codec removes V1's 32-cap.

Writes PNG plots to plots/ and prints a results table.
Runs in ~seconds on CPU, numpy only.
"""
import os
import random
import string

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from codec import V1Codec, FourierPosCodec, s2b, b2s, char_accuracy

OUT = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT, exist_ok=True)
random.seed(0)
np.random.seed(0)

ALPHA = string.ascii_letters + string.digits


def rword(L):
    return "".join(random.choice(ALPHA) for _ in range(L))


def word_exact(orig, rec):
    return 1.0 if list(orig) == list(rec) else 0.0


# =============================================================== Exp 1: acc vs length
def exp1_accuracy_vs_length():
    lengths = list(range(1, 129))
    trials = 20
    v1 = V1Codec()
    fours = {k: FourierPosCodec(k=k) for k in (16, 32, 64)}

    def curve(codec):
        ca, we = [], []
        for L in lengths:
            cs, es = [], []
            for _ in range(trials):
                b = s2b(rword(L))
                rec = codec.decode(codec.encode(b), L)
                cs.append(char_accuracy(b, rec))
                es.append(word_exact(b, rec))
            ca.append(np.mean(cs)); we.append(np.mean(es))
        return np.array(ca), np.array(we)

    v1_ca, v1_we = curve(v1)
    four_ca = {k: curve(c)[0] for k, c in fours.items()}
    four_we = {k: curve(c)[1] for k, c in fours.items()}

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(lengths, v1_ca, "k--", lw=2, label="V1 one-hot (cap 32, D=8192)")
    for k in (16, 32, 64):
        ax[0].plot(lengths, four_ca[k], lw=2,
                   label=f"Fourier k={k} (D={256*k})")
        ax[0].axvline(k, color="gray", ls=":", alpha=0.4)
    ax[0].axvline(32, color="red", ls=":", alpha=0.6)
    ax[0].set_title("Per-character reconstruction accuracy")
    ax[0].set_xlabel("word length (bytes)"); ax[0].set_ylabel("char accuracy")
    ax[0].set_ylim(0, 1.03); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(lengths, v1_we, "k--", lw=2, label="V1 one-hot")
    for k in (16, 32, 64):
        ax[1].plot(lengths, four_we[k], lw=2, label=f"Fourier k={k}")
    ax[1].set_title("Whole-word EXACT reconstruction rate")
    ax[1].set_xlabel("word length (bytes)"); ax[1].set_ylabel("exact-match rate")
    ax[1].set_ylim(0, 1.03); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("V1 hard-crops at 32; Fourier is exact up to k, then degrades gracefully",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp1_accuracy_vs_length.png"), dpi=120)
    plt.close(fig)

    # numbers for README
    print("\n[Exp1] char accuracy at selected lengths")
    print(f"{'L':>5} | {'V1':>6} | {'F k=16':>7} | {'F k=32':>7} | {'F k=64':>7}")
    for L in (16, 32, 33, 48, 64, 96, 128):
        i = L - 1
        print(f"{L:5d} | {v1_ca[i]:6.3f} | {four_ca[16][i]:7.3f} | "
              f"{four_ca[32][i]:7.3f} | {four_ca[64][i]:7.3f}")
    return v1_ca, four_ca


# ========================================================= Exp 2: adversarial collision
def exp2_collisions():
    """Equal-length strings sharing the first 32 bytes but differing after
    -> V1 collides EXACTLY (suffix past byte 32 is discarded), Fourier keeps them distinct."""
    v1 = V1Codec()
    fp = FourierPosCodec(k=32)
    N = 300
    prefix = "a" * 32
    suffix_len = 12                        # same length for both -> isolates content, not 1/sqrt(L)
    v1_dists, f_dists = [], []
    v1_coll = f_coll = 0
    for _ in range(N):
        s1 = prefix + rword(suffix_len)
        s2 = prefix + rword(suffix_len)
        if s1 == s2:
            continue
        b1, b2 = s2b(s1), s2b(s2)
        d_v1 = np.linalg.norm(v1.encode(b1) - v1.encode(b2))
        d_f = np.linalg.norm(fp.encode(b1) - fp.encode(b2))
        v1_dists.append(d_v1); f_dists.append(d_f)
        v1_coll += (d_v1 < 1e-9); f_coll += (d_f < 1e-9)
    v1_dists, f_dists = np.array(v1_dists), np.array(f_dists)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(v1_dists, bins=40, alpha=0.7, label=f"V1 one-hot (collision rate {v1_coll/len(v1_dists):.0%})")
    ax.hist(f_dists, bins=40, alpha=0.7, label=f"Fourier k=32 (collision rate {f_coll/len(f_dists):.0%})")
    ax.set_title("Distinct strings sharing a 32-byte prefix:\nV1 gives IDENTICAL embeddings (distance 0), Fourier keeps them apart",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("L2 distance between the two embeddings"); ax.set_ylabel("count")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp2_collisions.png"), dpi=120)
    plt.close(fig)

    print("\n[Exp2] adversarial collisions (shared 32-byte prefix, differing suffix)")
    print(f"  V1      : collision rate {v1_coll/len(v1_dists):.1%}, mean L2 dist {v1_dists.mean():.4f}")
    print(f"  Fourier : collision rate {f_coll/len(f_dists):.1%}, mean L2 dist {f_dists.mean():.4f}")
    return v1_coll / len(v1_dists), f_coll / len(f_dists)


# ================================================= Exp 3: dimensionality vs max exact length
def exp3_dim_vs_maxlen():
    """For each k, the max length with >=99% char-acc (~ exact-decode capacity) and its dim."""
    ks = [8, 16, 24, 32, 48, 64, 96, 128]
    trials = 30
    maxlens, dims = [], []
    for k in ks:
        fp = FourierPosCodec(k=k)
        best = 0
        for L in range(1, 4 * k + 1):
            acc = np.mean([char_accuracy(b := s2b(rword(L)), fp.decode(fp.encode(b), L))
                           for _ in range(trials)])
            if acc >= 0.99:
                best = L
            else:
                break
        maxlens.append(best); dims.append(fp.dim)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(dims, maxlens, "o-", lw=2, label="Fourier: max length with ≥99% exact decode")
    ax.plot([256 * 32], [32], "rs", ms=12, label="V1: hard cap 32 @ D=8192 (crops beyond)")
    for k, d, m in zip(ks, dims, maxlens):
        ax.annotate(f"k={k}", (d, m), textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.set_title("Losslessly-decodable length scales with the tunable knob k\n(V1's 32 is a fixed wall)",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("embedding dimension D = 256·k"); ax.set_ylabel("max exactly-decodable length")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp3_dim_vs_maxlen.png"), dpi=120)
    plt.close(fig)

    print("\n[Exp3] dimensionality vs max exactly-decodable length")
    print(f"{'k':>5} | {'D=256k':>7} | {'max-exact-len':>13}")
    for k, d, m in zip(ks, dims, maxlens):
        print(f"{k:5d} | {d:7d} | {m:13d}")
    print(f"{'V1':>5} | {256*32:7d} | {'32 (hard cap)':>13}")
    return ks, dims, maxlens


# ================================================= Exp 4: concrete long-word round-trip
def exp4_real_words_demo():
    words = [
        "pneumonoultramicroscopicsilicovolcanoconiosis",          # 45
        "Rindfleischetikettierungsueberwachungsaufgabenuebertragungsgesetz",  # 65 (German)
        "supercalifragilisticexpialidocious",                     # 34
        "a",                                                      # 1 (short)
    ]
    v1 = V1Codec(); fp = FourierPosCodec(k=64)
    print("\n[Exp4] concrete round-trip (V1 vs Fourier k=64, D=16384)")
    print(f"{'len':>4} | {'V1 ok?':>6} | {'Fourier ok?':>11} | word")
    for w in words:
        b = s2b(w)
        v1_rec = b2s(v1.decode(v1.encode(b), len(b)))
        f_rec = b2s(fp.decode(fp.encode(b), len(b)))
        print(f"{len(b):4d} | {('YES' if v1_rec==w else 'NO '):>6} | "
              f"{('YES' if f_rec==w else 'NO '):>11} | {w[:40]}{'...' if len(w)>40 else ''}")
        if v1_rec != w:
            print(f"        V1 recovered: {v1_rec!r}  <- cropped/garbled past byte 32")


if __name__ == "__main__":
    exp1_accuracy_vs_length()
    exp2_collisions()
    exp3_dim_vs_maxlen()
    exp4_real_words_demo()
    print("\nPlots written to", OUT)
    print("done.")
