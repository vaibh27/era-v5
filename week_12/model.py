import torch
import torch.nn as nn

VOCAB_SIZE = 4096
SEQ_LEN = 64
D_MODEL = 1024
N_LAYERS = 6
N_HEADS = 8
FFN_MULT = 4


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class DemoLM(nn.Module):
    """A small decoder-only transformer, sized to give Adam optimizer
    state a memory footprint worth partitioning across virtual GPUs."""

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_mult=FFN_MULT,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_mult) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x, attn_mask=self.causal_mask[:t, :t])
        x = self.ln_f(x)
        return self.head(x)


def make_batch(batch_size, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
    return idx.to(device), targets.to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = DemoLM()
    n = count_params(m)
    print(f"params: {n:,} ({n * 4 / 1e6:.1f} MB fp32)")
    x, y = make_batch(4)
    out = m(x)
    print("output shape:", out.shape)
