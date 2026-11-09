import torch
from typing import Literal
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class STECapWeights(torch.autograd.Function):
    """Straight-through estimator for the capping function to allow gradients to pass through."""
    @staticmethod
    def forward(ctx, w, w_max, L, iters, as_absolute):
        return cap_weights(w, w_max, L, iters, as_absolute)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

def cap_weights_ste(w: torch.Tensor, w_max: torch.Tensor, L: float = 1.0, iters: int = 5, as_absolute: bool = False):
    return STECapWeights.apply(w, w_max, L, iters, as_absolute)

def cap_weights(w: torch.Tensor, w_max: torch.Tensor, L: float = 1.0, iters: int = 5, as_absolute: bool = False):
    """Iteratively cap per-asset weights and redistribute the excess.

    Args:
        w:           Raw weight tensor  (batch, n_assets).
        w_max:       Maximum allowed weight per asset (scalar or batch tensor).
        L:           Target gross exposure / leverage  (sum |w| = L).
        iters:       Maximum redistribution iterations before returning.
        as_absolute: If True, cap |w| (long-short / neutral mode).
                     If False, cap  w  (long-only mode).

    Returns:
        Weight tensor with all |w_i| <= w_max and sum(|w|) ≈ L.
    """
    w_max = w_max.view(-1, 1) if w_max.dim() > 0 else w_max

    # Normalise to target leverage before the capping loop.
    if not as_absolute:
        w_scaled = (w / torch.clamp(w.sum(-1, keepdim=True), min=1e-8)) * L
    else:
        w_scaled = w / torch.clamp(torch.abs(w), min=1e-8) * L

    for _ in range(iters):
        # Amount by which each asset exceeds the cap (always >= 0).
        excess = torch.clamp(torch.abs(w_scaled) - w_max, min=0.0)

        if torch.all(excess < 1e-5):
            break

        # Clip weights and compute remaining capacity per asset.
        if not as_absolute:
            w_clipped = torch.minimum(w_scaled, w_max)
            space_left = torch.clamp(w_max - w_clipped, min=0.0)
            share = space_left * torch.sign(w_clipped)
        else:
            w_clipped = torch.clamp(w_scaled, -w_max, w_max)
            space_left = torch.clamp(w_max - torch.abs(w_clipped), min=0.0)
            share = space_left * torch.sign(w_clipped)

        total_excess = excess.sum(-1, keepdim=True)
        # Use |share| so negative weights don't cancel positive capacity.
        total_share = torch.sum(torch.abs(share), dim=-1, keepdim=True)

        # Redistribute excess proportionally; fall back to uniform if no
        # capacity is available (edge case: all assets already at w_max).
        w_scaled = torch.where(
            total_share > 1e-8,
            w_clipped + (total_excess * share / total_share),
            w_clipped + (total_excess * space_left / torch.clamp(torch.abs(space_left).sum(-1, keepdim=True), min=1e-8))
        )

    return w_scaled

def markowitz_layer(u, covarience, weights_option: Literal['long-only', 'long-short', 'neutral-only'], gamma=1.0, ridge=1e-2, t=1.0, w_max=0.15, L=1.0, eps=1e-8):
    """Differentiable Markowitz portfolio optimisation layer with STE support."""
    w_max_scaled = torch.tensor(w_max * L, device=u.device, dtype=u.dtype)

    # Regularised system: (gamma * Sigma + lambda * I) z = mu
    I = torch.eye(covarience.shape[1], device=u.device)
    A = gamma * covarience + ridge * covarience.diagonal().mean() * I

    z = torch.linalg.solve(A, u)

    # Standardise scores to decouple temperature from scale.
    z_norm = z / (torch.std(z, dim=-1, keepdim=True) + eps)

    if weights_option == "long-only":
        # Используем СТЕ-версию капинга
        w = F.softmax(z_norm / t, dim=-1)
        w_norm = cap_weights_ste(w, w_max_scaled, L, iters=5, as_absolute=False)

    elif weights_option == "long-short":
        raw_w = F.softmax(z_norm / t, dim=-1) - F.softmax(-z_norm / t, dim=-1)
        w_norm = cap_weights_ste(raw_w, w_max_scaled, L, iters=5, as_absolute=True)
        w_norm = w_norm / torch.clamp(torch.sum(torch.abs(w_norm), dim=-1, keepdim=True), min=eps) * L

    elif weights_option == "neutral-only":
        z_centered = z_norm - torch.mean(z_norm, dim=-1, keepdim=True)
        w = cap_weights_ste(z_centered, w_max_scaled, L, iters=5, as_absolute=True)
        w_norm = w - torch.mean(w, dim=-1, keepdim=True)
        w_norm = w_norm / torch.clamp(torch.sum(torch.abs(w_norm), dim=-1, keepdim=True), min=eps) * L

    return w_norm
