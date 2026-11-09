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
        covarience:   Optional default covariance matrix.
    """

    def __init__(self, config: dict, architecture: nn.Module, covarience=None):
        super().__init__()
        self.config = config
        self.architecture = architecture
        self.markowitz = markowitz_layer
        self.covarience = covarience

    def forward(self, X, cov=None):
        """
        Args:
            X:   Feature tensor  (B, N, F).
            cov: Covariance matrix (B, N, N). If None, falls back to self.covarience.

        Returns:
            mu:      Raw expected-return scores  (B, N).
            weights: Portfolio weights           (B, N).
        """
        mu = self.architecture(X)                    # (B, N)
        cov_matrix = cov if cov is not None else self.covarience
        if cov_matrix is None:
            raise ValueError(
                "Covariance matrix must be provided either in forward(X, cov) or in __init__."
            )

        weights = self.markowitz(
            mu,
            cov_matrix,
            weights_option=self.config["weights_option"],
            gamma=self.config["gamma"],
            ridge=self.config["ridge"],
            t=self.config["t"],
            w_max=self.config["w_max"],
            L=self.config["L"],
            eps=self.config["eps"],
        )
        return mu, weights
