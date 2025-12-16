import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration class for model hyperparameters"""
    vocab_size: int = 32  # 26 letters + padding + '<sos>' + '<eos>' + nulls for better dimension alignment
    block_size: int = 18
    n_emb: int = 64
    n_heads: int = 2
    head_size: int = n_emb // n_heads
    n_blocks: int = 2
    dropout: float = 0.2
    exp: int = 1
    pool: str = 'cls'
    mask_attn: bool = True


class AttentionLayer(nn.Module):
    """Single Attention Layer Module"""
    def __init__(self, config: Config = None, **kwargs):
        super().__init__()
        self.config = Config(**kwargs) if config is None else config

        # ensure that we can evenly split the embedded input across the number of heads
        assert config.n_emb % config.n_heads == 0, "Embedding Space not evenly divisible amongst attention heads"

        self.attn = nn.Linear(config.n_emb, 3 * config.n_emb, bias=False)  # for query, key, value -- split into K, Q, V during forward
        self.proj = nn.Linear(config.n_emb, config.n_emb)      # "projection" layer

        self.dropout = nn.Dropout(self.config.dropout)
        
    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        qkv = self.attn(x)  # (B, T, 3 * C)

        q, k, v = qkv.split(C, dim=2)  # split into query, key, value -- each (B, T, C)

        # each is (B, nh, T, hs) where nh:=n_head and hs:=head_size -- this effectively creates multiple attention heads
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)  

        
        # apply attention mask (if provided)
        if attn_mask is not None:
            attn_mask = attn_mask.view(B, 1, 1, T).repeat(1, self.config.n_heads, T, 1)  # (B, nh, T, T)

        # flashattention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.config.dropout)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.proj(out))  # "residual" dropout

        return out
    

class FeedForward(nn.Module):
    """Feed Forward Module"""
    def __init__(self, config: Config = None, **kwargs):
        super().__init__()
        self.config = Config(**kwargs) if config is None else config

        self.net = nn.Sequential(
            nn.Linear(self.config.n_emb, self.config.exp * self.config.n_emb),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.config.exp * self.config.n_emb, self.config.n_emb),
            nn.Dropout(self.config.dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """Transformer Block Module"""
    def __init__(self, config: Config = None, **kwargs):
        super().__init__()
        self.config = Config(**kwargs) if config is None else config

        self.ln1 = nn.LayerNorm(self.config.n_emb)
        self.ln2 = nn.LayerNorm(self.config.n_emb)
        self.attn = AttentionLayer(self.config)
        self.ffwd = FeedForward(self.config)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ffwd(self.ln2(x))
        return x
    

class PhonemeBERT(nn.Module):
    """BERT-like model for word to phonetic embedding mapping"""
    def __init__(self, config: Config = None, **kwargs):
        super().__init__()
        self.config = Config(**kwargs) if config is None else config
        assert self.config.pool.lower() in ['mean', 'cls'], "Pooling method must be either 'mean' or 'cls'"
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if 'cls' in self.config.pool.lower():
            self.config.block_size += 1  # update for cls token
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.n_emb))

        self.wte = nn.Embedding(self.config.vocab_size, self.config.n_emb)
        self.wpe = nn.Embedding(self.config.block_size, self.config.n_emb)
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_blocks)])
        self.ln_f = nn.LayerNorm(self.config.n_emb)

        self.lm_head = nn.Linear(self.config.n_emb, self.config.n_emb)

        # final mlp projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.config.n_emb, self.config.n_emb),
            nn.GELU(approximate='tanh'),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.n_emb, 50)
        )
    
    def forward(self, X, Y):
        B, T = X.shape

        mask = (X != 0).float().unsqueeze(-1)
        attn_mask = None

        # idx and targets are both (B, T) tensor of integers                    dimension tracking:
        tok_emb = self.wte(X)                                                      # (B,T,C)
        if self.config.pool.lower() == 'cls':
            tok_emb = torch.cat((self.cls_token.repeat(B, 1, 1), tok_emb), dim=1)  # (B,T+1,C) prepend class embedding

            if self.config.mask_attn:
                # add masking for attention -- mask padding tokens and ensure cls token is NOT masked
                cls_mask = torch.ones(B, 1, dtype=torch.float, device=X.device)
                attn_mask = torch.cat((cls_mask, mask.squeeze(-1)), dim=1)         # (B, T+1)

        pos_emb = self.wpe(torch.arange(tok_emb.shape[1], device=X.device))        # (T+1,C)
        x = tok_emb + pos_emb                                                      # (B,T+1,C)
        for block in self.blocks:
            x = block(x, attn_mask)                                                # (B,T+1,C)
        x = self.ln_f(x)                                                           # (B,T+1,C)
        pre_pool = self.lm_head(x)                                                 # (B,T+1,C)

        if 'mean' == self.config.pool.lower():
            pooled = (pre_pool * mask).sum(dim=1) / mask.sum(dim=1)                # (B, C)
        elif 'cls' == self.config.pool.lower():
            pooled = pre_pool[:, 0, :]                                             # (B, C)

        logits = self.projection(pooled)                                           # (B, 50)
        target = torch.ones(logits.shape[0], device=self.device)
        criterion = nn.CosineEmbeddingLoss()
        loss = criterion(logits, Y, target)

        return logits, loss