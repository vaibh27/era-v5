"""Kronecker codec, tokenizer, and data generation for the prefix-copy task.

Kronecker input embedding (deterministic, no lookup table):
    kappa(bytes) in R^(256*MAXLEN); for byte b at position p: index = b*MAXLEN + p, value 1/sqrt(L).
This is the flattened (byte x position) one-hot grid, exactly the V1 codec restricted to MAXLEN.
It is invertible per position (argmax over the 256 bytes in each position column).
"""
import numpy as np

MAXLEN = 8          # byte positions (input codec length AND output grid positions)
N_BYTES = 256       # byte alphabet
PAD_CLASS = 256     # extra output class = "no byte / end" -> 257 classes total
N_CLASSES = 257
D = N_BYTES * MAXLEN  # 2048

# Special tokens rendered as single control bytes so everything is uniform bytes.
SPECIALS = {"<bos>": bytes([2]), "<sep>": bytes([3]), "<eos>": bytes([4]), "<unk>": bytes([5])}


def token_to_bytes(tok: str) -> bytes:
    if tok in SPECIALS:
        return SPECIALS[tok]
    return tok.encode("utf-8")


def kappa(tok: str) -> np.ndarray:
    """Deterministic Kronecker embedding of a token, R^D."""
    b = token_to_bytes(tok)[:MAXLEN]
    v = np.zeros(D, dtype=np.float32)
    L = len(b)
    if L == 0:
        return v
    for p, byte in enumerate(b):
        v[byte * MAXLEN + p] = 1.0
    v /= np.sqrt(L)
    return v


def kappa_decode(v: np.ndarray) -> bytes:
    """Invert kappa: for each position, argmax over the 256 byte rows; empty column ends the word."""
    out = []
    for p in range(MAXLEN):
        col = v[p::MAXLEN]  # entries byte*MAXLEN+p for byte=0..255
        if col.max() <= 0:
            break
        out.append(int(col.argmax()))
    return bytes(out)


def token_to_grid_target(tok: str) -> np.ndarray:
    """Per-position byte-class targets for the bytegrid head, shape [MAXLEN], PAD after the bytes."""
    b = token_to_bytes(tok)[:MAXLEN]
    t = np.full(MAXLEN, PAD_CLASS, dtype=np.int64)
    for p, byte in enumerate(b):
        t[p] = byte
    return t


def grid_logits_to_token(grid: np.ndarray) -> str:
    """Decode a [MAXLEN, N_CLASSES] logit grid: argmax per position, stop at PAD, bytes -> string."""
    idx = grid.argmax(axis=-1)  # [MAXLEN]
    out = []
    for c in idx:
        if c == PAD_CLASS:
            break
        out.append(int(c))
    try:
        s = bytes(out).decode("utf-8")
    except Exception:
        return "<invalid>"
    # map control bytes back to special names
    for name, bb in SPECIALS.items():
        if bytes(out) == bb:
            return name
    return s


# ---------------- vocabulary + data ----------------

def make_word_pools(n_train_vocab: int, n_oov: int, seed: int = 0, minlen: int = 3, maxlen: int = 6):
    rng = np.random.default_rng(seed)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))

    def gen(n, exclude):
        words = set()
        while len(words) < n:
            L = int(rng.integers(minlen, maxlen + 1))
            w = "".join(rng.choice(letters, size=L))
            if w not in exclude:
                words.add(w)
        return sorted(words)

    train_words = gen(n_train_vocab, set())
    oov_words = gen(n_oov, set(train_words))
    return train_words, oov_words


class Vocab:
    """Fixed vocabulary for the softmax baseline: specials + training words + <unk>."""
    def __init__(self, train_words):
        self.itos = ["<bos>", "<sep>", "<eos>", "<unk>"] + list(train_words)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.unk = self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def index(self, tok):
        return self.stoi.get(tok, self.unk)


def make_sequences(word_pool, n_seq, seed=0):
    """Each sequence: <bos> W <sep> W <eos>  (5 word-positions)."""
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n_seq):
        w = word_pool[int(rng.integers(len(word_pool)))]
        seqs.append(["<bos>", w, "<sep>", w, "<eos>"])
    return seqs
