"""
Track 3 — Dynamic-length Kronecker codec.

Two deterministic byte-string -> vector codecs, both viewed as a 256 x (pos-dim) grid:
  - V1  : one-hot position over 32 slots  (hard 32-cap, dim 8192)
  - FourierPos : smooth sinusoidal position basis phi(p) in R^k  (no cap, dim 256*k)

Pure numpy. Encoding/decoding of arbitrary-length strings.
"""
import numpy as np

BYTES = 256


# ----------------------------------------------------------------------------- V1
class V1Codec:
    """Kronecker V1: onehot256(byte) (x) onehot32(pos), 1/sqrt(L) normalized. Cap = 32."""

    def __init__(self, max_pos=32):
        self.max_pos = max_pos
        self.pos_dim = max_pos
        self.dim = BYTES * max_pos

    def grid(self, b):
        """Return the 256 x max_pos grid M for byte list b (ints)."""
        L = len(b)
        M = np.zeros((BYTES, self.max_pos), dtype=np.float64)
        if L == 0:
            return M
        scale = 1.0 / np.sqrt(L)
        for p, byte in enumerate(b):          # p = 0-indexed position
            if p >= self.max_pos:
                break                          # <-- the crop: positions >32 dropped
            M[byte, p] += scale
        return M

    def encode(self, b):
        return self.grid(b).reshape(-1)

    def decode(self, M_flat, L):
        """Recover bytes. Only positions < max_pos are physically present."""
        M = M_flat.reshape(BYTES, self.max_pos)
        out = []
        for p in range(L):
            if p >= self.max_pos:
                out.append(-1)                 # unrecoverable (was cropped)
            else:
                out.append(int(np.argmax(M[:, p])))
        return out


# -------------------------------------------------------------------------- Fourier
class FourierPosCodec:
    """
    Kronecker with a smooth sinusoidal position basis:
        phi(p)_{2i}   = sin(w_i * p)
        phi(p)_{2i+1} = cos(w_i * p)      i = 0..k/2-1
        phi normalized to unit L2 norm.
    kappa(b) = 1/sqrt(L) * sum_p onehot256(b_p) (x) phi(p).   dim = 256 * k, no length cap.

    Frequency schedule:
      - 'linear' (default): w_i evenly spread in (0, pi]  -> low mutual coherence, well
        conditioned; the matched basis. Gives EXACT decode whenever L <= k.
      - 'geometric': transformer-style w_i = base^(-2i/k) (smooth; poor for decoding).

    Decode is least-squares (pseudo-inverse), not matched filter:
      M = (1/sqrt(L)) X Phi  with X the 256xL one-hot-per-column byte indicator and
      Phi = phi(1..L) (L x k).  Recover  Xhat = sqrt(L) * M @ pinv(Phi)  (256 x L),
      byte(p) = argmax_v Xhat[v, p].  Exact when L <= k (Phi full row rank); graceful for L > k.
    """

    def __init__(self, k=32, schedule="linear", base=10000.0):
        assert k % 2 == 0, "k must be even (sin/cos pairs)"
        self.k = k
        self.schedule = schedule
        self.base = base
        self.pos_dim = k
        self.dim = BYTES * k
        half = k // 2
        if schedule == "linear":
            # evenly spread frequencies in (0, pi] -> near-equiangular position frame
            self.omega = np.pi * (np.arange(1, half + 1) / (half + 1))
        elif schedule == "geometric":
            self.omega = base ** (-2.0 * np.arange(half) / k)
        else:
            raise ValueError(schedule)

    def phi(self, positions):
        """positions: 1-indexed int array (P,) -> unit-norm features (P, k)."""
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 1)  # (P,1)
        ang = positions * self.omega.reshape(1, -1)                          # (P, half)
        feat = np.empty((positions.shape[0], self.k), dtype=np.float64)
        feat[:, 0::2] = np.sin(ang)
        feat[:, 1::2] = np.cos(ang)
        feat /= np.linalg.norm(feat, axis=1, keepdims=True)                  # unit norm
        return feat

    def grid(self, b):
        """Return 256 x k grid M: row v = 1/sqrt(L) * sum_{p: b_p=v} phi(p)."""
        L = len(b)
        M = np.zeros((BYTES, self.k), dtype=np.float64)
        if L == 0:
            return M
        pos = np.arange(1, L + 1)              # 1-indexed positions
        feats = self.phi(pos)                  # (L, k)
        scale = 1.0 / np.sqrt(L)
        np.add.at(M, np.asarray(b), scale * feats)
        return M

    def encode(self, b):
        return self.grid(b).reshape(-1)

    def decode(self, M_flat, L):
        """Least-squares decode. Decoder is given L (as V1 effectively is via nonzero columns)."""
        if L == 0:
            return []
        M = M_flat.reshape(BYTES, self.k)
        Phi = self.phi(np.arange(1, L + 1))                # (L, k)
        Xhat = np.sqrt(L) * (M @ np.linalg.pinv(Phi))      # (256, L)
        return list(np.argmax(Xhat, axis=0).astype(int))


# ------------------------------------------------------------------------- helpers
def s2b(s):
    """string -> list of byte ints (utf-8)."""
    return list(s.encode("utf-8"))


def b2s(b):
    return bytes(x for x in b if x >= 0).decode("utf-8", errors="replace")


def char_accuracy(orig, rec):
    """fraction of positions correctly recovered (rec may contain -1 for cropped)."""
    n = len(orig)
    if n == 0:
        return 1.0
    return sum(1 for o, r in zip(orig, rec) if o == r) / n


if __name__ == "__main__":
    # smoke test
    v1 = V1Codec()
    fp = FourierPosCodec(k=32)
    word = "internationalization_and_localization_" * 2   # length ~76 > 32
    b = s2b(word)
    for name, c in [("V1", v1), ("Fourier", fp)]:
        M = c.encode(b)
        rec = c.decode(M, len(b))
        print(f"{name:8s} dim={c.dim:6d} len={len(b)} char_acc={char_accuracy(b, rec):.3f}")
