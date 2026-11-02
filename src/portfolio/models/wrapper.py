import torch
from torch.nn import nn
from src.portfolio.models.markowitz_layer import markowitz_layer
from typing import Literal
 
class DenoisedPortfolioWrapper(nn.Module):
    def __init__(self, config, architecture):
        super().__init__()
        self.config = config
        self.architecture = architecture
        self.markowitz = markowitz_layer

    def forward(self, X, cov):
        mu = self.architecture.forward(X)
        weights = self.markowitz(mu, cov, weights_option=self.config['weights_option'], gamma=self.config['gamma'], ridge=self.config['ridge'], t=self.config['t'], w_max=self.config['w_max'], L=self.config['L'], eps=self.config['eps'])
        return mu, weights