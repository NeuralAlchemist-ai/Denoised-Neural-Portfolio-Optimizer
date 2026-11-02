import torch
from typing import Literal
import torch.nn.functional as F
#TODO Implement STE (Straight-Through Estimator) for preventing zeroing gradients
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
    """Differentiable Markowitz portfolio optimisation layer.

    Solves the regularised mean-variance problem and maps the solution to
    portfolio weights according to the chosen construction strategy.

    Args:
        u:              Predicted expected returns  (batch, n_assets).
        covarience:     Covariance matrix           (batch, n_assets, n_assets).
        weights_option: Portfolio construction mode:
                          - 'long-only'   : softmax weights, all >= 0, sum = 1.
                          - 'long-short'  : signed weights, gross exposure = L.
                          - 'neutral-only': signed weights, net exposure = 0.
        gamma:  Risk-aversion coefficient  (higher -> lower risk).
        ridge:  Ridge regularisation added to the covariance diagonal.
        t:      Softmax temperature  (lower -> more concentrated bets).
        w_max:  Per-asset weight cap as a fraction of L.
        L:      Target gross leverage.
        eps:    Numerical stability floor.

    Returns:
        Portfolio weight tensor  (batch, n_assets).
    """
    w_max_scaled = torch.tensor(w_max * L, device=u.device, dtype=u.dtype)

    # Regularised system: (gamma * Sigma + lambda * I) z = mu  =>  z ≈ Sigma^{-1} mu / gamma
    I = torch.eye(covarience.shape[1], device=u.device)
    z = torch.linalg.solve(gamma * covarience + ridge * I, u)

    # Standardise scores to decouple temperature from scale.
    z_norm = z / (torch.std(z, dim=-1, keepdim=True) + eps)

    if weights_option == "long-only":
        # Softmax ensures positivity and sums to 1; cap_weights enforces w_max.
        w = F.softmax(z_norm / t, dim=-1)
        w_norm = cap_weights(w, w_max_scaled, L)

    elif weights_option == "long-short":
        # Long leg minus short leg gives a bipolar score; cap absolute exposure.
        raw_w = F.softmax(z_norm / t, dim=-1) - F.softmax(-z_norm / t, dim=-1)
        w_norm = cap_weights(raw_w, w_max_scaled, L, as_absolute=True)
        w_norm = w_norm / torch.clamp(torch.sum(torch.abs(w_norm), dim=-1, keepdim=True), min=eps) * L

    elif weights_option == "neutral-only":
        # Centre scores before capping to produce a zero-net-exposure portfolio.
        z_centered = z_norm - torch.mean(z_norm, dim=-1, keepdim=True)
        w = cap_weights(z_centered, w_max_scaled, L, as_absolute=True)
        # Re-centre after capping in case small asymmetries shifted the mean.
        w_norm = w - torch.mean(w, dim=-1, keepdim=True)
        w_norm = w_norm / torch.clamp(torch.sum(torch.abs(w_norm), dim=-1, keepdim=True), min=eps) * L

    return w_norm