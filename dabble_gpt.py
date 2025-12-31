from dataclasses import dataclass
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
token_enc = tiktoken.get_encoding('gpt2')

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257  # GPT-2 vocab size -- 50,000 BPE merges + 256 bytes + <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_emb: int = 768
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout: float = 0.0


class LoRALayer(nn.Module):
    """LoRA layers for optional fine-tuning of pretrained models with low-rank adaptation"""
    def __init__(self, original_linear, rank=8, alpha=None):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha if alpha is not None else 2 * rank
        self.A = nn.Parameter(torch.empty((original_linear.in_features, rank)))  # nn.Parameter automatically moves to model's device
        self.B = nn.Parameter(torch.empty((rank, original_linear.out_features)))
        
        # initialize A and B -- using default init for linear layers
        self.A = nn.init.uniform_(self.A, -1/math.sqrt(original_linear.in_features), 1/math.sqrt(original_linear.in_features))
        self.B = nn.init.zeros_(self.B)  # init B w/ zeros enables first passes to have identical behavior as oriiginal linear layer
                                         # this is crucial for retaining pre-trained knowledge at the start of fine-tuning
        self.scaling = self.alpha / self.rank  # scaling factor allows us to adjust the magnitude of the LoRA update

        # freeze original linear layer
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
       return self.original_linear(x) + (x @ self.A @ self.B) * self.scaling
       

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_emb % config.n_head == 0, "Embedding Space not evenly divisible amongst attention heads"
        self.config = config
        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)  
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)      

        self.dropout = nn.Dropout(config.dropout)
        # "bias" matchs GPT2 naming convention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                          .view(1, 1, config.block_size, config.block_size))

        self.kv_cache = None  # init module-level kv cache -- will need to clear this at each new inference loop

    def forward(self, x):
        """Ordinary forward with flash attention"""
        B, T, C = x.size()  # batch size, sequence length, emb dim (n_emb)

        nh = self.config.n_head  # number of attention heads
        hs = C // nh             # size of each attention head

        qkv = self.c_attn(x)                      # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=2)             # (B, T, C) each
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        
        # FlashAttention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.config.dropout)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side --> (B, T, C)
        y = self.c_proj(y)   # linear "projection"
        y = self.dropout(y)  # "residual" dropout
        return y
    
    def cache_forward(self, x):
        """KV-cached forward for inference"""
        B, T, C = x.size()  # batch size, sequence length (typically 1), emb dim (n_emb)

        nh = self.config.n_head  # number of attention heads
        hs = C // nh             # size of each attention head

        qkv = self.c_attn(x)                      # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=2)             # (B, T, C) each
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        
        # concat past kv if they exist
        if self.kv_cache is not None:
            past_k, past_v = self.kv_cache
            k = torch.cat((past_k, k), dim=-2)  # (B, nh, T_full, hs) concat along sequence dim -- this adds context memory
            v = torch.cat((past_v, v), dim=-2)  # (B, nh, T_full, hs)

        self.kv_cache = (k, v)  # update kv_cache 
        T_full = k.size(-2)     # current k and v are both full sequence

        # compute attentwion weights
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.config.n_head)  # (B, nh, T, T_full)

        # causal mask with updatd sequence length
        # updated mask gets masks for current full sequence row, up to full sequence length
        # e.g. if x is 3rd token in sequence, T_full-T:T_full gets row 2 and :T_full gets first 3 mask vals in row
        att = att.masked_fill(self.bias[:, :, T_full-T:T_full, :T_full] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v          # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side --> (B, T, C)
        y = self.c_proj(y)   # linear "projection"
        y = self.dropout(y)  # "residual" dropout
        
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_emb, 4 * config.n_emb)  # 4* expansion comes from GPT2 paper
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_emb, config.n_emb)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)   # (B, T, C)
        self.attn = CausalSelfAttention(config)  # (B, T, C)
        self.ln_2 = nn.LayerNorm(config.n_emb)   # (B, T, C)
        self.mlp = MLP(config)                   # (B, T, C)

    def forward(self, x, use_kv_cache: bool = False):
        if not use_kv_cache: x = x + self.attn(self.ln_1(x))  
        else: x = x + self.attn.cache_forward(self.ln_1(x))  
        x = x + self.mlp(self.ln_2(x))   
        return x


class GPT(nn.Module):
    def __init__(self, config: Config = None, **kwargs):
        super().__init__()
        self.config = Config(**kwargs) if config is None else config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_emb),  # token embeddings
            wpe = nn.Embedding(self.config.block_size, self.config.n_emb),  # position embeddings
            h = nn.ModuleList([Block(self.config) for layer in range(self.config.n_layer)]),
            ln_f = nn.LayerNorm(self.config.n_emb)
        ))
        self.lm_head = nn.Linear(self.config.n_emb, self.config.vocab_size, bias=False)

        # weight tying -- the token embedding weights are shared with the output projection weights
        self.transformer.wte.weight = self.lm_head.weight

        self.to(self.config.device)

    def forward(self, idx, targets=None, use_kv_cache: bool = False, pos_offset: int = 0):
        """
        Forward pass
        Args:
            idx - (tensor) input to forward
            target - (tensor) target for loss evaluation, default None
            use_kv_cache - (bool) whether to use kv caching, default False
            pos_offset - (int) offset input position in sequence for pos embeddings, necessary for kv cache with streaming input
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence length {T} > model block size {self.config.block_size}."

        pos = torch.arange(pos_offset, pos_offset + T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_emb)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_emb)
        x = tok_emb + pos_emb   # there is implicit broadcasting here -- (B, T, n_emb) + (T, n_emb) --> (B, T, n_emb)

        # forward pass through transformer blocks
        for block in self.transformer.h:
            x = block(x, use_kv_cache=use_kv_cache)
        
        x = self.transformer.ln_f(x)  # final layer norm
        logits = self.lm_head(x)      # logits for each token in vocab -- (B, T, vocab_size)
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits

    @torch.no_grad()
    def generate(self, seq, max_new_tokens: int | None = None, top_k: int = 50, use_kv_cache: bool = True) -> str: 
        """
        Generate sequences from model
        Args:
            seq: Prompt sequence
            max_new_tokens: Max new tokens to generate, limit 1024 if None. Dfaults to None
            top_k: Limit token sample to top k logits. Defaults to 50
            use_kv_cache: Generate with KV caching if True
        """
        if self.training: self.eval()
        if isinstance(seq, str):
            seq = torch.tensor(token_enc.encode(seq)).reshape(1, -1).to(self.config.device)

        # clear kv_caches from the model
        for name, module in self.named_modules():
            if isinstance(module, CausalSelfAttention):
                setattr(module, 'kv_cache', None)

        if max_new_tokens is not None:
            token_lim = min((self.config.block_size - len(seq)), max_new_tokens)
        else:
            token_lim = (self.config.block_size - len(seq))
        
        
        current_pos = 0
        for i in range(token_lim):
            if use_kv_cache:
                if i == 0:
                    x = seq  # full prompt on first pass
                    pos_offset = 0
                    current_pos = x.size(1)
                else:
                    x = seq[:, -1:]  # all other passes, use last token
                    pos_offset = current_pos
                    current_pos += 1
                logits = self(x, targets=None, use_kv_cache=True, pos_offset=pos_offset)
            else:
                logits = self(seq, targets=None, use_kv_cache=False)

            logits = logits[:, -1, :]  # consider only the last token

            # crop to top_k logits if provided
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            seq_next = torch.multinomial(probs, num_samples=1)  # sample from distribution
            if seq_next.item() == token_enc.max_token_value:
                return token_enc.decode(seq.cpu().detach().numpy()[0])   # break if eos token predicted
            seq = torch.cat((seq, seq_next), dim=1)  # append sampled index to running sequence
        
        return token_enc.decode(seq.cpu().detach().numpy()[0])

    @classmethod
    def from_pretrained(cls, model_type: str = 'gpt2', lora_rank: int = 0, lora_alpha: int = None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, "Unsupported model type"
        from transformers import GPT2LMHeadModel

        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_emb=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_emb=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_emb=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_emb=1600),
        }[model_type]
        config_args.update(dict(block_size=1024, vocab_size=50257))
        config = Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]  # attn.bias not in the pretrained model

        # init huggingface model and get its state dict
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.bias') and not k.endswith('.attn.masked_bias')]

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            # special treatment for the Conv1D weights we need to transpose
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"shape mismatch: first 2 dim of {sd_hf[k].shape} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())  # .t() transposes first two dimensions
            # vanilla copy over the other parameters
            else:
                assert sd_hf[k].shape == sd[k].shape, f"shape mismatch: {sd_hf[k].shape} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # if using lora, freeze parameters and replace linear layers with LoRA layers
        if lora_rank > 0:
            # freeze all parameters initially
            for p in model.parameters():
                p.requires_grad = False  
            # replace linear layers with LoRA layers (in CausalAttention and MLP)
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(k in name for k in['c_attn', 'c_proj', 'c_fc']):
                    parent_name = '.'.join(name.split('.')[:-1])
                    attr_name = name.split('.')[-1]
                    
                    parent = model.get_submodule(parent_name) if parent_name else model

                    # Replace with LoRA version
                    lora_linear = LoRALayer(module, rank=lora_rank, alpha=lora_alpha)
                    setattr(parent, attr_name, lora_linear)
                
            model.to(model.config.device)  # move model with LoRA layers to cuda if available

        return model