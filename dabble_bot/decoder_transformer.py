import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ----- training functions -----
def train_epoch(model, dataset, optimizer, batch_size, track_losses: bool = False):
    """Train for one epoch."""
    model.train()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches = len(dl) 
    total_loss = 0

    losses = []
    for xb, yb in dl:

        # shift xb and yb contexts
        xb = xb[:, :-1]  # ensure input is always current char
        yb = yb[:, 1:]   # ensure target is always next char

        optimizer.zero_grad()
        logits = model(xb)
        
        loss = nn.CrossEntropyLoss()(logits, yb)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if track_losses: losses.append(loss.item())

    return total_loss / n_batches, losses


@torch.no_grad()
def eval_loss(model, dataset, sample_frac=0.2):
    """Evaluate model loss on a random sample of data"""
    # Store and set evaluation mode
    was_training = model.training
    model.eval()

    X, Y = dataset.encoded_words, dataset.embeddings
    try:
        n_samples = max(1, int(X.shape[0] * sample_frac))
        indices = torch.randperm(X.shape[0], device=X.device)[:n_samples]
        
        # shift contexts
        X, Y = X[:, :-1], Y[:, 1:]

        logits = model(X[indices])
        loss = nn.CrossEntropyLoss()(logits, yb)
        
        return loss.item()
    
    finally:
        # Always restore training state, even if error occurs
        if was_training:
            model.train()


def training_loop(model, train_data, test_data, optimizer, track_losses: bool = False,
               batch_size: int = 64, n_epochs: int = 10, eval_sample_frac: float = 0.5):
    """Full training loop over multiple epochs."""
    
    start = time.time()
    losses = []
    for epoch in range(n_epochs):
        train_loss, eloss = train_epoch(model, train_data, optimizer, batch_size)
        test_loss = eval_loss(model, test_data, sample_frac=eval_sample_frac)
        # training_losses.append(train_loss)
        run_time = time.time() - start
        eta = divmod((run_time / (epoch + 1)) * (n_epochs - epoch - 1), 60)
        print(
            f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
            + " | ETA: {min:,.0f} min, {sec:,.0f} sec".format(min=eta[0], sec=eta[1])
        )
        if track_losses: losses.extend(eloss)

    if track_losses: 
        plt.plot(losses);


# ----- model components -----
@dataclass(eq=False)  # eq=False enables child classes to still be hashable (long story)
class Config:
    """Configuration class for model hyperparameters"""
    vocab_size: int = 29  # vocab_size: 27 (a-z + padding) + 2 (SOS, EOS) = ~29
    block_size: int = 17
    n_emb: int = 64
    n_heads: int = 2
    head_size: int = n_emb // n_heads
    n_blocks: int = 2
    dropout: float = 0.2
    exp: int = 1


class CausalAttentionLayer(nn.Module):
    """Single Attention Layer Module"""
    def __init__(self, config: Config = None, **kwargs):
        super().__init__()
        self.config = Config(**kwargs) if config is None else config

        # ensure that we can evenly split the embedded input across the number of heads
        assert config.n_emb % config.n_heads == 0, "Embedding Space not evenly divisible amongst attention heads"

        self.attn = nn.Linear(config.n_emb, 3 * config.n_emb, bias=False)  # for query, key, value -- split into K, Q, V during forward
        self.proj = nn.Linear(config.n_emb, config.n_emb)  # "projection" layer
        self.register_buffer(
            'tril', torch.tril(torch.ones(self.config.block_size, self.config.block_size))  # (T, T)
        )
        self.dropout = nn.Dropout(self.config.dropout)
        
    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        qkv = self.attn(x)  # (B, T, 3 * C)

        q, k, v = qkv.split(C, dim=2)  # split into query, key, value -- each (B, T, C)

        # each is (B, nh, T, hs) where nh:=n_head and hs:=head_size -- this effectively creates multiple attention heads
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)  

        # QK attention with standardization -- (B, nh, T, T)
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        weights = nn.Softmax(dim=-1)(weights)
        weights = self.dropout(weights)     # "attention" dropout

        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
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
        self.attn = CausalAttentionLayer(self.config)
        self.ffwd = FeedForward(self.config)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ffwd(self.ln2(x))
        return x
    

class DecoderGPT(nn.Module):
    """Transformer decoder for phoneme-embedding to word mapping"""
    def __init__(self, config: Config = None, **kwargs):
        super().__init__()
        self.config = Config(**kwargs) if config is None else config

        self.wte = nn.Embedding(29, self.config.n_emb)  # 26 letters + padding + '<sos>' + '<eos>'
        self.wpe = nn.Embedding(self.config.block_size, self.config.n_emb)
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_blocks)])
        self.ln_f = nn.LayerNorm(self.config.n_emb)

        self.lm_head = nn.Linear(self.config.n_emb, self.config.vocab_size)

    
    def forward(self, phonetic_input, char_input):
        B, T = char_input.shape

        # init char seqeunce with sos token (27)
        tok_emb = self.wte(input)                                      # (B, T, C)
        pos_emb = self.wpe(torch.arange(T, device=char_input.device))  # (T, C)
        phonetic_input = phonetic_input.unsqueeze(1).repeat(1, T, 1)   # (B, T, 50)
        x = tok_emb + pos_emb                                          # (B, T, C)
        x = torch.cat([x, phonetic_input], dim=2)                      # (B, T, C+50)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits