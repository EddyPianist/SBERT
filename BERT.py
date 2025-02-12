import torch
import torch.nn as nn
from torch.nn import functional as F
import math

###TASK ONE: Sentence Transformer Implementation

#most of the implementations are following the Sentence BERT (SBERT)
#implement basic componets for transformer: 
#self-attn
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd


    def forward(self, x, attn_mask):
        B, nT, C = x.shape
        x = self.c_attn(x)
        q, k, v = x.split(self.n_embd, dim = 2)

        q = q.view(B, nT, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, nT, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, nT, self.n_head, C // self.n_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 /math.sqrt(k.size(-1)))
         
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1, seq_len]
        attn_mask = attn_mask.expand(-1, 12, nT , -1)   # expand attn_mask: shape: [batch_size, num_heads, seq_len, seq_len]
        
        attn = torch.masked_fill(attn, attn_mask == 0, float('-inf'))
        attn = F.softmax(attn, dim = -1)


        x = attn @ v
        x = x.transpose(1, 2).contiguous().view(B, nT, C)
        x = self.c_proj(x)
        return x
    

#fully connected layers
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
#transformer block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


#Implement the sentence transformer
class SBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),                   #token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),                   #position embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd), 
        ))
        self.pooling = nn.AdaptiveAvgPool1d(1)     #pooling token embeddings to get a fixed length

        #init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal(module.weight, mean = 0.0, std = 0.02)
                
    def forward(self, sf):
        attn_mask = sf["attention_mask"]  
        idx = sf["input_ids"]
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"
        pos = torch.arange(0, T, dtype=torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x, attn_mask)
        
        x = self.transformer.ln_f(x)
        x = x.transpose(2, 1)
        ste_embd = self.pooling(x)              #different from traditional transformer, we use a pooling operation to get sentence embedding
        ste_embd = ste_embd.squeeze(-1)
        
        return ste_embd
    


        

