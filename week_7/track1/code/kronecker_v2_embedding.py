r"""
Kronecker Embedding V2 — a single embedding model for BOTH text and math.

    E(token) = [  Kronecker text (byte x position, 32 slots)  ||  CRT math register  ]
                  the "32 existing spaces" for words              appended math dims

  * EVERY token gets the deterministic Kronecker byte x position code (Kronecker V1): a
    tokenizer-free, reversible word embedding — the "32 existing spaces".
  * NUMBER tokens ALSO fill the appended MATH register, so a FIXED operator does EXACT
    arithmetic on the math dims (9+9 -> 18, 9*9 -> 81) and decodes back. Words -> math dims 0.
  * A gate (is-number) routes each token. Arithmetic touches ONLY the math subspace.

The MATH register is arithmetic_embedding's verified ContinuousCRTEmbedding (exact +, x via fixed
bilinear tensors; add/mul = 1.0 over all operand pairs; reversible via the CRT). We build the text+math
model ON TOP of that code (this is the HybridArithmeticEmbedding idea with its learned semantic side
replaced by the Kronecker byte x position text code). A single trainable W_proj -> d_model makes it a
drop-in input embedding for a transformer (Kronecker V1 style); decoding uses the FIXED features.

Requires torch (pip install -r arithmetic_embedding/requirements.txt).
"""
import os
import sys
import math

import torch
import torch.nn as nn

# use arithmetic_embedding's verified CRT math register
_AE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arithmetic_embedding", "code")
if _AE not in sys.path:
    sys.path.insert(0, _AE)
from continuous_crt import ContinuousCRTEmbedding  # noqa: E402  (their exact, reversible math register)

BYTES = 256


# =============================================================== Kronecker TEXT (byte x position)
class KroneckerText(nn.Module):
    """V1 code: onehot256(byte) (x) onehot32(position), 1/sqrt(L) normalized. Reversible per slot."""

    def __init__(self, max_pos: int = 32):
        super().__init__()
        self.max_pos = max_pos
        self.dim = BYTES * max_pos

    def encode(self, token: str) -> torch.Tensor:
        b = list(str(token).encode("utf-8"))[: self.max_pos]
        M = torch.zeros(BYTES, self.max_pos)
        if b:
            s = 1.0 / math.sqrt(len(b))
            for p, byte in enumerate(b):
                M[byte, p] += s
        return M.reshape(-1)

    def decode(self, v: torch.Tensor, length: int) -> str:
        M = v.reshape(BYTES, self.max_pos)
        out = []
        for p in range(min(length, self.max_pos)):
            col = M[:, p]
            out.append(int(col.argmax().item()) if col.max().item() > 0 else 0)
        return bytes(out).decode("utf-8", "replace")


def is_number(tok: str) -> bool:
    return str(tok).lstrip("-").isdigit()


# =============================================================== the combined V2 embedding model
class KroneckerV2Embedding(nn.Module):
    """[ Kronecker text  ||  CRT math ]. One embedding for words AND numbers."""

    def __init__(self, moduli=(5, 7, 11, 13, 17, 19), max_pos: int = 32, d_model: int | None = None):
        super().__init__()
        self.text = KroneckerText(max_pos)
        self.math = ContinuousCRTEmbedding(moduli)       # arithmetic_embedding's verified code
        self.tdim, self.mdim = self.text.dim, self.math.dim
        self.dim = self.tdim + self.mdim
        self.capacity = self.math.system.capacity        # exact math range 0..capacity-1
        self.proj = nn.Linear(self.dim, d_model, bias=False) if d_model else None

    # ---- fixed [text || math] features for one token (decodable) ----
    def embed_features(self, tok: str) -> torch.Tensor:
        text = self.text.encode(tok)
        if is_number(tok) and 0 <= int(tok) < self.capacity:
            m = self.math.encode(int(tok)).float()
        else:
            m = torch.zeros(self.mdim)
        return torch.cat([text, m])

    def forward(self, tokens) -> torch.Tensor:
        """tokens: list[str] -> [N, dim] fixed features, or [N, d_model] if a projection is set."""
        feats = torch.stack([self.embed_features(t) for t in tokens])
        return self.proj(feats) if self.proj is not None else feats

    # ---- split / decode (on the FIXED features) ----
    def text_of(self, feat: torch.Tensor, length: int) -> str:
        return self.text.decode(feat[: self.tdim], length)

    def has_math(self, feat: torch.Tensor) -> bool:
        return feat[self.tdim:].max().item() > 0

    def number_of(self, feat: torch.Tensor):
        m = feat[self.tdim:]
        return None if m.max().item() <= 0 else int(self.math.decode(m).item())

    # ---- arithmetic: fixed operator on the math dims, returns a fresh embedding of the result ----
    def _op(self, fa: torch.Tensor, fb: torch.Tensor, name: str) -> torch.Tensor:
        m = getattr(self.math, name)(fa[self.tdim:], fb[self.tdim:])   # their exact +/x
        n = int(self.math.decode(m).item())
        return self.embed_features(str(n))

    def add(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        return self._op(fa, fb, "add")

    def mul(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        return self._op(fa, fb, "mul")


if __name__ == "__main__":
    E = KroneckerV2Embedding()
    print(f"Kronecker V2 embedding: dim = {E.dim}  (text {E.tdim} + math {E.mdim}), "
          f"exact math range 0..{E.capacity-1}")
    print(f"math register = {type(E.math).__module__}.{type(E.math).__name__} "
          f"(from arithmetic_embedding)\n")

    print("--- words: text embedding works, math dims empty ---")
    for w in ["apple", "a", "cat"]:
        f = E.embed_features(w)
        print(f"  {w:8s} -> text '{E.text_of(f, len(w))}', has_math={E.has_math(f)}")

    print("\n--- numbers: BOTH a text embedding AND a math value ---")
    for n in ["9", "42", "255"]:
        f = E.embed_features(n)
        print(f"  {n:8s} -> text '{E.text_of(f, len(n))}', math value = {E.number_of(f)}")

    print("\n--- arithmetic on the math dims (fixed operator, their CRT tensors) ---")
    nine = E.embed_features("9")
    print(f"  9 + 9   -> {E.number_of(E.add(nine, nine))}   (want 18)")
    print(f"  9 * 9   -> {E.number_of(E.mul(nine, nine))}   (want 81)")
    print(f"  12 * 11 -> {E.number_of(E.mul(E.embed_features('12'), E.embed_features('11')))}   (want 132)")

    print("\n--- a mixed 'sentence' embeds uniformly (words + numbers) ---")
    for tok in ["buy", "9", "apples", "for", "42", "rupees"]:
        f = E.embed_features(tok)
        print(f"  {tok:8s} -> {'number=' + str(E.number_of(f)) if E.has_math(f) else 'word'}")

    print("\n--- benchmark-ready: project fixed features to d_model with one trainable W_proj ---")
    Eproj = KroneckerV2Embedding(d_model=128)
    out = Eproj(["buy", "9", "apples"])
    print(f"  forward(['buy','9','apples']) -> {tuple(out.shape)} tensor (plug into a transformer)")
