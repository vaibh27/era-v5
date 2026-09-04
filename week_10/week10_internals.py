import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer

# ============================================================
# Configuration
# ============================================================
import random
import numpy as np

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)

seq_len = 128
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"device: {device}")
num_proc = max(1, (os.cpu_count() or 2) - 2)
N_ROWS = 20000

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size
print(f"vocab_size: {vocab_size}")

# ============================================================
# Dataset
# ============================================================

def truncate_and_pad(batch):

    processed, real_lens = [], []
    for ids in batch["input_ids"]:
        real_len = min(len(ids), seq_len + 1)
        ids = ids[:seq_len + 1]
        if len(ids) < seq_len + 1:
            ids = ids + [tokenizer.pad_token_id] * (seq_len + 1 - len(ids))
        processed.append(ids)
        real_lens.append(real_len)
    return {"input_ids": processed, "real_len": real_lens}


def get_dataset():
    """Load, tokenize, and pad TinyStories. Returns processed dataset with input_ids and real_len."""
    import os
    cache_dir = "/tmp/tinystories_processed"
    
    # If cached dataset exists, load it (fully deterministic)
    if os.path.exists(cache_dir):
        from datasets import load_from_disk
        dataset = load_from_disk(cache_dir)
        print(f"dataset loaded from cache: {len(dataset)} rows")
        return dataset
    
    # Set all seeds for fully deterministic dataset processing
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)
    
    # Use num_proc=1 for deterministic processing (multiprocessing RNG not seeded)
    dataset = load_dataset("roneneldan/TinyStories", split='train')
    dataset = dataset.filter(
        lambda batch: [len(text) > 0 for text in batch['text']],
        batched=True, num_proc=1,
    )
    dataset = dataset.shuffle(SEED).select(range(N_ROWS))

    encoded_dataset = dataset.map(lambda batch: tokenizer(batch['text']), batched=True, num_proc=1)

    update_dataset = encoded_dataset.map(truncate_and_pad, batched=True, num_proc=1)
    print(f"dataset ready: {len(update_dataset)} rows")
    
    # Save to cache for future runs
    update_dataset.save_to_disk(cache_dir)
    print(f"dataset cached to {cache_dir}")
    return update_dataset


def get_batch(batch_size, train_seq_len, dataset=None, step=None):
    if dataset is None:
        dataset = get_dataset()

    if step is not None:
        torch.manual_seed(SEED + step)

    indices = torch.randint(0, len(dataset), (batch_size,)).tolist()
    raw = torch.tensor(dataset[indices]["input_ids"])[:, :train_seq_len + 1].to(device)
    real_lens = dataset[indices]["real_len"]
    tokens, targets = raw[:, :-1], raw[:, 1:]
    pad_mask = torch.zeros(batch_size, train_seq_len, dtype=torch.bool)
    for row, real_len in enumerate(real_lens):
        capped = min(real_len, train_seq_len + 1)
        pad_mask[row, :max(capped - 1, 0)] = True
    return tokens, targets, pad_mask.to(device)

# ============================================================
# Model
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = num_heads * head_size
        
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.projection = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, hidden):
        batch, t, _ = hidden.shape
        
        q = self.q_proj(hidden).view(batch, t, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(hidden).view(batch, t, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(hidden).view(batch, t, self.num_heads, self.head_size).transpose(1, 2)
        
        # MPS doesn't support dropout in SDPA; apply manually if training on MPS
        dropout_p = 0.0
        if self.training and hidden.device.type != 'mps':
            dropout_p = self.attn_dropout.p
        
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True
        )
        
        # Apply attention dropout manually for MPS
        if self.training and hidden.device.type == 'mps':
            attn_out = self.attn_dropout(attn_out)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, t, self.n_embd)
        return self.dropout(self.projection(attn_out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, hidden):
        return self.net(hidden)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attn = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, hidden):
        hidden = hidden + self.self_attn(self.layer_norm1(hidden))
        hidden = hidden + self.ffwd(self.layer_norm2(hidden))
        return hidden


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_encoding_table = nn.Embedding(seq_len, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx):
        batch, t = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_encoding_table(torch.arange(t, device=idx.device))
        hidden = tok_emb + pos_emb
        hidden = self.blocks(hidden)
        return self.ln_f(hidden)


class OutputHead(nn.Module):
    def __init__(self, tied_embedding=None):
        super().__init__()
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        if tied_embedding is not None:
            self.lm_head.weight = tied_embedding.weight

    def forward(self, hidden):
        return self.lm_head(hidden)
