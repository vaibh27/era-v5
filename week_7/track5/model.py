"""Tiny GPT with a swappable output head: 'softmax' (V-way) vs 'bytegrid' (MAXLEN x 257).

Input embedding is the deterministic Kronecker codec + a single trainable projection W_proj,
shared by both variants so the ONLY difference is the output head.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kron import D, MAXLEN, N_CLASSES


class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, head: str, vocab_size: int, block_size: int,
                 d_model=64, n_layer=2, n_head=2):
        super().__init__()
        assert head in ("softmax", "bytegrid")
        self.head_type = head
        self.d_model = d_model
        self.block_size = block_size

        # Input side: fixed Kronecker features (fed in) -> single trainable projection.
        self.w_proj = nn.Linear(D, d_model, bias=False)
        self.pos = nn.Parameter(torch.zeros(block_size, d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

        if head == "softmax":
            self.head = nn.Linear(d_model, vocab_size)
        else:
            self.head = nn.Linear(d_model, MAXLEN * N_CLASSES)

    def head_param_count(self):
        return sum(p.numel() for p in self.head.parameters())

    def forward(self, kappa_feats):
        # kappa_feats: [B, T, D] fixed Kronecker input features
        B, T, _ = kappa_feats.shape
        x = self.w_proj(kappa_feats) + self.pos[:T]
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1).to(x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        out = self.head(x)
        if self.head_type == "bytegrid":
            out = out.view(B, T, MAXLEN, N_CLASSES)
        return out

    def loss(self, kappa_feats, sm_targets, grid_targets):
        out = self.forward(kappa_feats)
        if self.head_type == "softmax":
            # out [B,T,V], targets [B,T]
            return F.cross_entropy(out.reshape(-1, out.size(-1)), sm_targets.reshape(-1))
        else:
            # out [B,T,MAXLEN,257], targets [B,T,MAXLEN]
            C = out.size(-1)
            return F.cross_entropy(out.reshape(-1, C), grid_targets.reshape(-1))
