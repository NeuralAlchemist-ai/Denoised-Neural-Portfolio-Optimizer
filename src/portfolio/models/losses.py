import torch
import torch.nn as nn

class SharpeLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, weights, next_r):
        portfolio_returns = torch.sum(weights * next_r, dim=-1) # (Batch_Size,)

        mean_returns =  torch.mean(portfolio_returns)
        
        portfolio_std = torch.std(portfolio_returns, unbiased=False)

        sharpe_ratio = (mean_returns) / (portfolio_std + self.eps)

        return -sharpe_ratio
        
class ICCLoss(nn.Module):
    def __init__(self, eps=1e-8, temperature=0.1):
        super().__init__()
        self.eps = eps
        self.temperature = temperature

    def forward(self, predicted_mu, real_mu):
        with torch.no_grad():
            real_ranks = torch.argsort(torch.argsort(real_mu, dim=-1), dim=-1).float() + 1.0

        scores_i = predicted_mu.unsqueeze(2) # (Batch_Size, N, 1)
        scores_j = predicted_mu.unsqueeze(1) # (Batch_Size, 1, N)
        delta_ij = scores_j - scores_i
        sigmoid_matrix = torch.sigmoid(delta_ij / self.temperature)

        num_assets = predicted_mu.shape[-1]
        eye = torch.eye(num_assets, device=predicted_mu.device).unsqueeze(0) # (1, N, N)
        mask = 1.0 - eye

        soft_ranking = 1 + torch.sum(sigmoid_matrix * mask, dim=2)

        N = float(num_assets)

        sum_sq_diff = torch.sum(torch.square(soft_ranking - real_ranks), dim=-1)
        
        spearman_correlation = 1.0 - (6.0 * sum_sq_diff) / (N * (N**2 - 1.0) + self.eps)
        return -torch.mean(spearman_correlation)

class CombinedLoss(nn.Module):
    def __init__(self, alpha, eps=1e-8, temperature=0.1):
        super().__init__()
        self.initial_alpha = alpha
        self.eps = eps
        self.temperature = temperature

        self.sharpe_loss = SharpeLoss(eps)
        self.icc_loss = ICCLoss(eps, temperature)

        initial_logit = torch.log(torch.tensor(self.initial_alpha) / (1.0 - self.initial_alpha))
        
        self.alpha_logit = nn.Parameter(initial_logit)
    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, weights, next_r, predicted_mu, real_mu):
        current_alpha = self.alpha
        return current_alpha * self.sharpe_loss(weights, next_r) + (1 - current_alpha) * self.icc_loss(predicted_mu, real_mu)
    