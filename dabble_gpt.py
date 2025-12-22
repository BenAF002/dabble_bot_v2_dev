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

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_emb)

        qkv = self.c_attn(x)           # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=2)  # each (B, T, C)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2) 
        
        # FlashAttention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.config.dropout)
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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  
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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence length {T} > model block size {self.config.block_size}."

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_emb)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_emb)
        x = tok_emb + pos_emb   # there is implicit broadcasting here -- (B, T, n_emb) + (T, n_emb) --> (B, T, n_emb)

        # forward pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)  # final layer norm

        logits = self.lm_head(x)      # logits for each token in vocab -- (B, T, vocab_size)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits

    @torch.no_grad()
    def generate(self, seq, max_new_tokens: int | None = None, top_k: int | None = 50):
        if self.training: self.eval()
        if isinstance(seq, str):
            seq = torch.tensor(token_enc.encode(seq)).reshape(1, -1).to(self.config.device)

        if max_new_tokens is not None:
            token_lim = min(self.config.block_size, max_new_tokens)
        else:
            token_lim = self.config.block_size
        
        for _ in range(token_lim):
            logits = self(seq, targets=None)
            logits = logits[:, -1, :]  # consider only the last token
            # crop to top_k logits if provided
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            seq_next = torch.multinomial(probs, num_samples=1)  # sample from distribution
            # if seq_next.item() == token_enc.max_token_value:
            #     return token_enc.decode(seq.detach().numpy())   # break if eos token predicted
            seq = torch.cat((seq, seq_next), dim=1)  # append sampled index to running sequence

        return token_enc.decode(seq.detach().numpy())

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
        config_args.update(dict(block_size=1024, vocab_size=50257))  # common args for all GPT-2 models
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
            for name, module in model.named_modules():
                
                for p in module.parameters():
                    p.requires_grad = False  # freeze all parameters initially

                if isinstance(module, nn.Linear) and ('c_attn' in name or 'c_proj' in name):  # only apply LoRA to attn and proj layers
                    parent_name = '.'.join(name.split('.')[:-1])
                    attr_name = name.split('.')[-1]
                    
                    parent = model.get_submodule(parent_name) if parent_name else model

                    # Replace with LoRA version
                    lora_linear = LoRALayer(module, rank=lora_rank, alpha=lora_alpha)
                    setattr(parent, attr_name, lora_linear)

        return model