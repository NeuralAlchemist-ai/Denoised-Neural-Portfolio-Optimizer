import torch
import pytest

from portfolio.models.networks import DeepMarkowitz_MLP, DeepMarkowitz_MHA, DeepMarkowitz_MLA

def test_model_dims():
    B,N,F = 8, 50, 10

    model_mlp = DeepMarkowitz_MLP()
    model_mha = DeepMarkowitz_MHA()
    model_mla = DeepMarkowitz_MLA()

    model_mlp.eval()
    model_mha.eval()
    model_mla.eval()
    
    with torch.no_grad():
        x = torch.randn(B,N,F)
        y_mlp = model_mlp(x)
        y_mha = model_mha(x)
        y_mla = model_mla(x)
        assert y_mlp.shape == (B, N)
        assert y_mha.shape == (B, N)
        assert y_mla.shape == (B, N)