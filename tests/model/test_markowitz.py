import pytest
import torch
from portfolio.models.markowitz_layer import markowitz_layer

B, N = 4, 12

@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(69)

@pytest.fixture
def mock_data():
    mu = torch.randn(B, N)
    A = torch.randn(B, N, N)
    cov = A @ A.transpose(-1, -2) / N + 0.1 * torch.eye(N)
    return mu, cov

@pytest.mark.parametrize("L", [1.0, 2.5])
@pytest.mark.parametrize("w_max", [0.15, 0.3])
def test_long_only(mock_data, L, w_max):
    mu, cov = mock_data
    w = markowitz_layer(mu, cov, L=L, w_max=w_max, weights_option="long-only")
    assert w.shape == (B, N)
    assert torch.all(w >= -1e-6)
    assert torch.all(w <= L * w_max + 1e-5)
    assert torch.allclose(w.sum(-1), torch.full((B,), L), atol=1e-5)

@pytest.mark.parametrize("L", [1.0, 2.5])
@pytest.mark.parametrize("w_max", [0.15, 0.3])
def test_long_short(mock_data, L, w_max):
    mu, cov = mock_data
    w = markowitz_layer(mu, cov, L=L, w_max=w_max, weights_option="long-short")
    assert w.shape == (B, N)
    limit = L * w_max + 1e-5
    assert torch.all(w <= limit)
    assert torch.all(w >= -limit)
    gross_exposure = torch.sum(torch.abs(w), dim=-1)
    assert torch.allclose(gross_exposure, torch.full((B,), L), atol=1e-4)
    assert torch.any(w < 0)


@pytest.mark.parametrize("L", [1.0, 2.5])
@pytest.mark.parametrize("w_max", [0.15, 0.3])
def test_neutral_only(mock_data, L, w_max):
    mu, cov = mock_data
    w = markowitz_layer(mu, cov, L=L, w_max=w_max, weights_option="neutral-only")
    assert w.shape == (B, N)
    limit = L * w_max + 1e-5
    assert torch.all(w <= limit)
    assert torch.all(w >= -limit)
    net_exposure = torch.sum(w, dim=-1)
    assert torch.allclose(net_exposure, torch.zeros(B), atol=1e-4)
    gross_exposure = torch.sum(torch.abs(w), dim=-1)
    assert torch.allclose(gross_exposure, torch.full((B,), L), atol=1e-4)
