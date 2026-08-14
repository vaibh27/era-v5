"""
Problem 4 — a REAL Fourier alternative to Kronecker: represent each character as a wave and
just ADD them to make a word.

Kronecker (V1) uses a PRODUCT (byte outer-product position) -> a one-hot grid of dim 256*32.
The Fourier alternative uses a SUPERPOSITION (Holographic / VSA binding):
  - each byte value c is a fixed wave  chi_c in C^D  (unit-modulus random phases -> a "wave");
  - each position p is a fixed rotation ro_p in C^D  (phase ramp -> a time-shift);
  - a word is the SUM of its character-waves, each rotated to its position:
        X(word) = (1/sqrt(L)) * sum_p  chi_{b_p} (*) ro_p            [(*) = elementwise]
  Dimension D is a free knob, INDEPENDENT of alphabet size and word length (no 256*32 blow-up,
  no 32 cap). Decoding is Fourier analysis: rotate back and correlate.

Reversible: to read position p, unbind by conj(ro_p) and pick the character whose wave correlates
most. Exact when D is large enough relative to length L (signal ~ D, crosstalk ~ sqrt(D*L)).

Pure numpy.
"""
import numpy as np

BYTES = 256


class FourierCharCodec:
    def __init__(self, D=1024, seed=0):
        self.D = D
        rng = np.random.default_rng(seed)
        # each character = a fixed wave (unit-modulus complex spectrum)
        self.chi = np.exp(2j * np.pi * rng.random((BYTES, D)))
        # each position = a fixed rotation; ro_p = base_phase ** p  (a genuine per-step rotation)
        self.omega = np.exp(2j * np.pi * rng.random(D))           # per-bin rotation rate
        self._chi_conj = np.conjugate(self.chi)

    def ro(self, p):
        """rotation for position p (0-indexed): omega ** p."""
        return self.omega ** p

    def encode(self, b):
        """byte list -> complex vector in C^D (the summed character-waves)."""
        L = len(b)
        X = np.zeros(self.D, dtype=np.complex128)
        if L == 0:
            return X
        for p, c in enumerate(b):
            X += self.chi[c] * self.ro(p)
        return X / np.sqrt(L)

    def decode(self, X, L):
        """recover the L bytes: unbind each position, correlate with every character wave."""
        out = []
        for p in range(L):
            y = X * np.conjugate(self.ro(p))              # unbind position p
            scores = (self._chi_conj @ y).real            # correlate with all 256 character-waves
            out.append(int(np.argmax(scores)))
        return out

    def features(self, b):
        """real feature vector (cos/sin) of dim 2D for an ML model."""
        X = self.encode(b)
        return np.concatenate([X.real, X.imag])


# ------------------------------------------------------------------- V1 (Kronecker) for contrast
class V1Codec:
    """Kronecker V1: onehot256(byte) x onehot32(pos), dim 8192, hard 32 cap."""
    def __init__(self, max_pos=32):
        self.max_pos = max_pos
        self.dim = BYTES * max_pos

    def encode(self, b):
        M = np.zeros((BYTES, self.max_pos))
        L = len(b)
        if L == 0:
            return M.reshape(-1)
        for p, c in enumerate(b):
            if p >= self.max_pos:
                break
            M[c, p] += 1.0 / np.sqrt(L)
        return M.reshape(-1)


def s2b(s):
    return list(s.encode("utf-8"))


def b2s(b):
    return bytes(x for x in b if 0 <= x < 256).decode("utf-8", errors="replace")


if __name__ == "__main__":
    fc = FourierCharCodec(D=2048)
    for w in ["a", "apple", "internationalization", "cat", "act", "tac"]:
        b = s2b(w)
        rec = b2s(fc.decode(fc.encode(b), len(b)))
        print(f"{w:24s} dim={2*fc.D:5d}  ->  {rec:24s}  {'OK' if rec == w else 'XX'}")
