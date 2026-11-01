import torch
import torch.nn as nn
from typing import Literal
import torch.nn.functional as F

import torch

def cap_weights(w: torch.Tensor, w_max: torch.Tensor, L: float = 1.0, iters: int = 5) -> torch.Tensor:
    w_max = w_max.view(-1, 1) if w_max.dim() > 0 else w_max
    
    w_scaled = (w / torch.clamp(w.sum(-1, keepdim=True), min=1e-8)) * L

    for _ in range(iters):
        excess = torch.clamp(torch.abs(w_scaled) - w_max, min=0.0) * torch.sign(w_scaled)
        
        if torch.all(excess < 1e-5):
            break
            
        w_clipped = torch.minimum(w_scaled, w_max)
        
        space_left = torch.clamp(w_max - torch.abs(w_clipped), min=0.0) * torch.sign(w_clipped)
        
        share = space_left * torch.abs(w_clipped) * torch.sign(w_clipped)
        
        total_excess = excess.sum(-1, keepdim=True)
        total_share = share.sum(-1, keepdim=True)
        
        w_scaled = torch.where(
            total_share > 1e-8,
            w_clipped + (total_excess * share / total_share),
            w_clipped + (total_excess * space_left / torch.clamp(space_left.sum(-1, keepdim=True), min=1e-8))
        )
        
    return w_scaled


def markowitz_layer(u, covarience, weights_option: Literal['long-only', 'long-short', 'neutral-only'], gamma=1.0, ridge=1e-2, t=1.0, w_max=0.15, L=1.0, eps=1e-8):
    w_max_scaled = torch.tensor(w_max * L, device=u.device, dtype=u.dtype)
    I = torch.eye(covarience.shape[1], device=u.device)
    z = torch.linalg.solve(gamma*covarience + ridge*I, u)
    z_norm = z / (torch.std(z, dim=-1, keepdim=True) + eps)

    if weights_option == "long-only":
            w = F.softmax(z_norm / t, dim=-1)
            w_norm = cap_weights(w, w_max_scaled, L)

    elif weights_option == "long-short":
            raw_w = F.softmax(z_norm / t, dim=-1) - F.softmax(-z_norm / t, dim=-1)
            w = raw_w / torch.sum(torch.abs(raw_w), dim=-1, keepdim=True)

            w_capped = cap_weights(w, w_max_scaled, L)
            w_norm = w_capped / torch.clamp(torch.sum(torch.abs(w_capped), dim=-1, keepdim=True), min=1e-8)

    elif weights_option == "neutral-only":
            z_centered = z_norm - torch.mean(z_norm, dim=-1, keepdim=True)
            l1_norm = torch.linalg.norm(z_centered, ord=1, dim=-1, keepdim=True)
            w = z_centered / torch.clamp(l1_norm, min=1e-8)

            w_capped = torch.clamp(w, min = -w_max_scaled, max = w_max_scaled)
            w_neutral = w_capped - torch.mean(w_capped, dim=-1, keepdim=True)
            w_norm = w_neutral / torch.clamp(torch.sum(torch.abs(w_neutral), dim=-1, keepdim=True), min=1e-8) * L
    return w_norm