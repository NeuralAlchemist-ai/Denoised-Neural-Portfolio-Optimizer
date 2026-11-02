import torch 
import torch.nn as nn
import torch.nn.functional as F

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

class MLA_Attention(nn.Module):
    """Multi-head Latent Attention over assets.
    B batch, N assets, E d_model, H heads, d qk dim, v v dim,
    q q_rank, k kv_rank."""
    def __init__(self, d_model=32, n_heads=4, q_rank=8, kv_rank=8,
                 qk_head_dim=8, v_head_dim=8):
        super().__init__()
        self.qk_head_dim = qk_head_dim

        self.q_down_proj = nn.Linear(d_model, q_rank, bias=False)
        self.q_layernorm = nn.LayerNorm(q_rank)
        self.kv_down_proj = nn.Linear(d_model, kv_rank, bias=False)
        self.kv_layernorm = nn.LayerNorm(kv_rank)

        self.q_up_weights = nn.Parameter(torch.empty(n_heads, qk_head_dim, q_rank))
        self.k_up_weights = nn.Parameter(torch.empty(n_heads, qk_head_dim, kv_rank))
        self.v_up_weights = nn.Parameter(torch.empty(n_heads, v_head_dim, kv_rank))
        for w in (self.q_up_weights, self.k_up_weights, self.v_up_weights):
            nn.init.normal_(w, std=w.shape[-1] ** -0.5)

        self.out_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=False)

    def forward(self, x):                                    # (B, N, E)
        B, N, _ = x.shape
        c_kv = self.kv_layernorm(self.kv_down_proj(x))       # (B, N, k)
        c_q = self.q_layernorm(self.q_down_proj(x))          # (B, N, q)

        Q = torch.einsum("bnq,hdq->bhnd", c_q, self.q_up_weights)
        Q_abs = torch.einsum("bhnd,hdk->bhnk", Q, self.k_up_weights)
        score = torch.einsum("bhnk,bmk->bhnm", Q_abs, c_kv) / self.qk_head_dim ** 0.5
        score = F.softmax(score, dim=-1)                     # (B, H, N, N)

        out_lat = torch.einsum("bhnm,bmk->bhnk", score, c_kv)
        ctx = torch.einsum("bhnk,hvk->bnhv", out_lat, self.v_up_weights)
        return self.out_proj(ctx.reshape(B, N, -1))          # (B, N, E)


class MLA_Block(nn.Module):
    def __init__(self, d_model=32, d_ff=64, dropout=0.1, **mla_kwargs):
        super().__init__()
        self.attn = MLA_Attention(d_model=d_model, **mla_kwargs)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        return x + self.dropout(self.ffn(self.norm2(x)))


class DeepMarkowitz_MLA(nn.Module):
    def __init__(self, in_dim=10, d_model=32, n_heads=4, q_rank=8, kv_rank=8,
                 qk_head_dim=8, v_head_dim=8, d_ff=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.blocks = nn.Sequential(*[
            MLA_Block(d_model=d_model, d_ff=d_ff, dropout=dropout,
                      n_heads=n_heads, q_rank=q_rank, kv_rank=kv_rank,
                      qk_head_dim=qk_head_dim, v_head_dim=v_head_dim)
            for _ in range(num_layers)
        ])
        self.head = ScoreHead(d_model)

    def forward(self, x):                                    # (B, N, in_dim)
        return self.head(self.blocks(self.embed(x)))         # (B, N)