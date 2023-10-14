import torch
from torch import nn
import torch.nn.functional as F


class Head(nn.Module):

    """Head module for the transformer block, which applies self-attention to the input"""

    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.query = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embed, config.head_size, bias=False)

        self.register_buffer(
            "trail", torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x):
        # B = batch size, T = sequence length, C = number of channels
        B, T, C = x.shape

        # compute queries, keys and values
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        v = self.value(x)  # (B, T, C)

        # scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)

        wei = F.softmax(wei, dim=-1)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention module, which performs multiple parallel self-attention operations"""

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C * n_head)
        out = self.dropout(self.proj(out))  # project back to (B, T, C)
        return out


class FeedFoward(nn.Module):
    """a simple linear layerm for computing the feed-forward part of the transformer block"""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: a self-attention layer followed by a feed-forward layer
    with a residual connection"""

    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedFoward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        # apply residual connection to the input (x + self-attention(x)) & normalize
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Classifier(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        self.token_embedding_table = nn.Embedding(
            config.vocab_size, config.n_embed, padding_idx=0
        )
        self.position_embedding_table = nn.Embedding(
            config.block_size, config.n_embed, padding_idx=0
        )

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.output_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, labels=None):
        # idx: (B, T)
        # labels: (B, T)
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, C)

        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)

        logits = self.lm_head(x)  # (B, T, C)
        logits = logits[:, -1, :]  # (B, C)

        if labels is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, labels)

        return logits, loss
