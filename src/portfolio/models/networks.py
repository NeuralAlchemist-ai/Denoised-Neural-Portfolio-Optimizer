import torch 
import torch.nn as nn

import torch
import torch.nn as nn


class ScoreHead(nn.Module):
    """Shared output head: LayerNorm -> Linear -> one scalar per asset"""
    def __init__(self, d_model, scale=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)
        self.scale = scale

    def forward(self, h):                       
        return self.scale * self.fc(self.norm(h)).squeeze(-1)


class MLPBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.ff(self.norm(x)))


class DeepMarkowitz_MLP(nn.Module):
    """Shared per-asset MLP"""
    def __init__(self, in_dim=10, d_model=32, d_ff=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.blocks = nn.Sequential(
            *[MLPBlock(d_model, d_ff, dropout) for _ in range(num_layers)]
        )
        self.head = ScoreHead(d_model)

    def forward(self, x):
        return self.head(self.blocks(self.embed(x)))


class DeepMarkowitz_MHA(nn.Module):
    def __init__(self, in_dim=10, d_model=32, n_heads=4, d_ff=64,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.head = ScoreHead(d_model)

    def forward(self, x):
        return self.head(self.encoder(self.embed(x)))  

class DeepMarkowitz_MLA(nn.Module):
    """Multihead Latent Attention"""
    def __init__(self, d_model=32, n_heads=4, q_rank=8, kv_rank=8, 
                 qk_head_dim=8, v_head_dim=8, d_ff=64, dropout=0.1):
        super(DeepMarkowitz_MLA, self).__init__()

        self.q_down_proj = nn.Linear(d_model, q_rank, bias=False)
        self.q_layernorm = nn.LayerNorm(q_rank)

        self.kv_down_proj = nn.Linear(d_model, kv_rank, bias=False)
        self.kv_up_proj = nn.Linear(kv_rank, d_model, bias=False)
        self.kv_layernorm = nn.LayerNorm(kv_rank)

        self.q_up_weights = nn.Parameter(torch.empty(n_heads, qk_head_dim, q_rank))
        self.k_up_weights = nn.Parameter(torch.empty(n_heads, qk_head_dim, kv_rank))
        self.v_up_weights = nn.Parameter(torch.empty(n_heads, v_head_dim, kv_rank))        


        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_up_weights)
        nn.init.xavier_uniform_(self.k_up_weights)
        nn.init.xavier_uniform_(self.v_up_weights)

    def forward(self, x): 
        pass
        
    
        
        
