import torch
import pytest

from portfolio.models.networks import DeepMarkowitz_MLP, DeepMarkowitz_MHA

def test_model_dims():
    B,N,F = 8, 50, 10

    model = DeepMarkowitz_MLP()
    model.eval()
    with torch.no_grad():
        x = torch.randn(B,N,F)
        y = model(x)
        assert y.shape == (B, N)

    model = DeepMarkowitz_MHA()
    model.eval()
    with torch.no_grad():
        x = torch.randn(B,N,F)
        y = model(x)
        assert y.shape == (B, N)
    