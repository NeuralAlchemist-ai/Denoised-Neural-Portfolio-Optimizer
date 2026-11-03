import torch.nn as nn
from portfolio.models.markowitz_layer import markowitz_layer


class DenoisedPortfolioWrapper(nn.Module):
    """Wraps a backbone network with a differentiable Markowitz portfolio layer.

    Forward pass:
        1. Backbone produces raw expected-return scores  mu  (B, N).
        2. Markowitz layer maps (mu, cov) → portfolio weights  (B, N).

    Args:
        config:       Model config dict (loaded from model.yaml).
        architecture: An instantiated backbone (DeepMarkowitz_MLP/MHA/MLA).
    """

    def __init__(self, config: dict, architecture: nn.Module, covarience):
        super().__init__()
        self.config = config
        self.architecture = architecture
        self.markowitz = markowitz_layer
        self.covarience = covarience

    def forward(self, X):
        """
        Args:
            X:   Feature tensor  (B, N, F).
            cov: Covariance matrix (B, N, N).

        Returns:
            mu:      Raw expected-return scores  (B, N).
            weights: Portfolio weights           (B, N).
        """
        mu = self.architecture(X)                    # (B, N)
        weights = self.markowitz(
            mu,
            self.covarience,
            weights_option=self.config["weights_option"],
            gamma=self.config["gamma"],
            ridge=self.config["ridge"],
            t=self.config["t"],
            w_max=self.config["w_max"],
            L=self.config["L"],
            eps=self.config["eps"],
        )
        return mu, weights
