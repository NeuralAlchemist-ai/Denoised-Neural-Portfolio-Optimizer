# Denoised Neural Portfolio Optimizer

**Deep Markowitz with Spectral Filtering** - end-to-end differentiable portfolio construction that combines Random Matrix Theory covariance denoising with neural return prediction and a fully differentiable Markowitz allocation layer.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![NumPy](https://img.shields.io/badge/NumPy-2.2%2B-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.3%2B-150458?logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.7%2B-F7931E?logo=scikit-learn&logoColor=white)
![CVXPY](https://img.shields.io/badge/CVXPY-1.7%2B-9B59B6)
![TensorBoard](https://img.shields.io/badge/TensorBoard-2.21%2B-FF6F00?logo=tensorflow&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?logo=astral&logoColor=white)
![version](https://img.shields.io/badge/version-0.1.0-informational)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

---

## Overview

Modern portfolio construction suffers from two compounding problems: covariance matrices estimated from historical data are contaminated with noise, and return predictions produced by standard neural networks are disconnected from the portfolio objective. **Denoised Neural Portfolio Optimizer** attacks both simultaneously:

1. **Spectral denoising** - Marchenko-Pastur Random Matrix Theory is used to separate genuine factor structure from random noise in the estimated covariance matrix before it enters the optimizer.
2. **Neural return predictor** - three interchangeable architectures (MLP, Multi-Head Attention, Multi-Head Latent Attention) map asset features to expected-return scores.
3. **Differentiable Markowitz layer** - a closed-form, GPU-compatible allocation layer converts scores and a denoised covariance matrix into portfolio weights, allowing end-to-end gradient-based training.
4. **Composite loss** - training jointly maximizes a differentiable Sharpe ratio and a soft Spearman rank-correlation (ICC) between predicted and realized returns.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
  - [Neural Architectures](#neural-architectures)
  - [Markowitz Layer](#markowitz-layer)
  - [Covariance Estimation & Denoising](#covariance-estimation--denoising)
  - [Classical Strategies](#classical-strategies)
  - [Loss Functions](#loss-functions)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Category | What's included |
|---|---|
| **Neural encoders** | `DeepMarkowitz_MLP`, `DeepMarkowitz_MHA` (Transformer), `DeepMarkowitz_MLA` (Multi-Head Latent Attention) |
| **Allocation layer** | Differentiable Markowitz — `long-only`, `long-short`, `neutral-only` modes with iterative weight capping |
| **Covariance estimators** | Sample, Ledoit-Wolf shrinkage, EWMA |
| **Spectral denoising** | Marchenko-Pastur filter — replaces noise eigenvalues with their mean, enforces positive definiteness |
| **Classical baselines** | Equal Weight (1/N), Minimum Variance, Maximum Sharpe |
| **Loss functions** | `SharpeLoss`, `ICCLoss` (differentiable Spearman correlation), `CombinedLoss` (weighted mix) |
| **Data pipeline** | `yfinance`-backed downloader, feature engineering, preprocessing |

---

## Installation

**Requirements:** Python ≥ 3.10, [uv](https://github.com/astral-sh/uv) (recommended) or pip.

```bash
git clone https://github.com/NeuralAlchemist-ai/Denoised-Neural-Portfolio-Optimizer.git
cd Denoised-Neural-Portfolio-Optimizer

# with uv (recommended)
uv sync

# or with pip
pip install -e .
```

---

## Quick Start

### Classical optimization with denoised covariance

```python
import pandas as pd
from portfolio.strategies.classical import MinimumVarianceStrategy, MaxSharpeStrategy

# returns: pd.DataFrame of shape (T, N) — daily log returns
strategy = MinimumVarianceStrategy(estimator="ledoit_wolf", denoise=True)
weights = strategy.compute_weights(returns)   # np.ndarray (N,)

tangency = MaxSharpeStrategy(estimator="ewma", denoise=True, half_life=63)
weights  = tangency.compute_weights(returns)
```

### Neural portfolio (end-to-end)

```python
import torch
import yaml
from portfolio.models.networks import DeepMarkowitz_MLA
from portfolio.models.wrapper import DenoisedPortfolioWrapper

config = yaml.safe_load(open("configs/model.yaml"))

architecture = DeepMarkowitz_MLA(in_dim=10, d_model=32, num_layers=2)
model = DenoisedPortfolioWrapper(config, architecture)

# X:   (Batch, N_assets, Features)
# cov: (Batch, N_assets, N_assets)  — pre-estimated, optionally denoised
mu, weights = model(X, cov)
```

### Training with composite loss

```python
from portfolio.models.losses import CombinedLoss

criterion = CombinedLoss(alpha=0.5, temperature=0.1)

# alpha controls the Sharpe / ICC trade-off
# alpha=1.0 → pure Sharpe,  alpha=0.0 → pure rank correlation
loss = criterion(weights, next_returns, predicted_mu, realized_mu)
loss.backward()
```

---

## Architecture

### Neural Architectures

All three encoders share the same interface:  
`forward(x: Tensor[B, N, F]) → mu: Tensor[B, N]`  
where B = batch, N = assets, F = features per asset.

| Class | Description |
|---|---|
| `DeepMarkowitz_MLP` | Independent per-asset MLP with residual MLPBlocks. Fast, no cross-asset interaction. |
| `DeepMarkowitz_MHA` | Standard Transformer encoder — captures cross-asset dependencies via Multi-Head Self-Attention. |
| `DeepMarkowitz_MLA` | Multi-Head Latent Attention (inspired by DeepSeek MLA) — compressed key/value projections reduce memory with minimal accuracy loss. |

All architectures share a `ScoreHead` output: `LayerNorm → Linear → scalar per asset`.

### Markowitz Layer

[`markowitz_layer`](src/portfolio/models/markowitz_layer.py) is a **fully differentiable** closed-form optimizer:

```
z = (γΣ + λI)⁻¹ μ        # regularised mean-variance solution
z_norm = z / std(z)        # scale-invariant scores
weights = portfolio_mode(z_norm / t)   # softmax / bipolar / centred
```

**Weight construction modes:**

| Mode | Description | Use case |
|---|---|---|
| `long-only` | Softmax → iterative capping to `w_max` | Equity funds, no shorting allowed |
| `long-short` | Bipolar softmax difference → absolute cap | Hedge funds, gross exposure = L |
| `neutral-only` | Mean-centred scores → absolute cap → re-centre | Market-neutral strategies |

`cap_weights` iteratively redistributes excess weight from capped assets to those with remaining capacity, preserving the target leverage `L`.

> **TODO:** Implement STE (Straight-Through Estimator) to prevent gradient zeroing at capped weights.

### Covariance Estimation & Denoising

Three estimators live in [`portfolio/covariance/estimators.py`](src/portfolio/covariance/estimators.py):

| Estimator | Key | Notes |
|---|---|---|
| Sample | `"sample"` | Unbiased, unreliable when N/T is large |
| Ledoit-Wolf | `"ledoit_wolf"` | Analytically optimal shrinkage; always PD |
| EWMA | `"ewma"` | Exponential decay; requires `half_life` kwarg |

All estimators can be wrapped with **Marchenko-Pastur denoising** ([`denoise_covariance`](src/portfolio/covariance/denoise.py)):

```python
from portfolio.covariance.denoise import denoise_covariance
clean_cov = denoise_covariance(cov_matrix, n_obs=252)
```

The filter identifies the MP upper spectral edge `λ₊ = σ²(1 + √(N/T))²`, replaces all noise eigenvalues with their mean (preserving total variance), and enforces positive definiteness via an eigenvalue floor.

### Classical Strategies

Located in [`portfolio/strategies/classical.py`](src/portfolio/strategies/classical.py).  
All strategies implement `BaseStrategy.compute_weights(returns_window) → np.ndarray`.

| Strategy | Class | Objective |
|---|---|---|
| Equal Weight | `EqualWeightStrategy` | 1/N across all assets |
| Minimum Variance | `MinimumVarianceStrategy` | min wᵀΣw  s.t. Σwᵢ=1, w≥0 |
| Maximum Sharpe | `MaxSharpeStrategy` | max (μᵀw − rᶠ) / √(wᵀΣw) |

`MinimumVarianceStrategy` and `MaxSharpeStrategy` accept any covariance estimator (string or callable) and an optional `denoise=True` flag.

### Loss Functions

[`portfolio/models/losses.py`](src/portfolio/models/losses.py) provides three `nn.Module` loss functions:

**`SharpeLoss`** — maximizes the batch-level Sharpe ratio:
```
L_sharpe = −  E[r_p] / std(r_p)
```

**`ICCLoss`** — differentiable Spearman rank correlation via soft ranking:
```
soft_rank_i = 1 + Σⱼ σ((μ̂ⱼ − μ̂ᵢ) / τ)
L_ICC = − mean( 1 − 6 Σᵢ(r̂ᵢ − rᵢ)² / (N(N²−1)) )
```

**`CombinedLoss`** — weighted combination:
```
L = α · L_sharpe + (1−α) · L_ICC
```
`alpha` can be annealed during training (e.g. `alpha = max(0.1, 1.0 − epoch/20)`).

---

## Configuration

All model hyperparameters are read from [`configs/model.yaml`](configs/model.yaml):

```yaml
weights_option: "long-only"   # long-only | long-short | neutral-only
gamma: 1.0                    # risk-aversion coefficient
ridge: 1.0e-2                 # covariance ridge regularisation
t: 1.0                        # softmax temperature
w_max: 0.15                   # per-asset weight cap (fraction of L)
L: 1.0                        # target gross leverage
eps: 1.0e-8                   # numerical stability floor
```

---

## Project Structure

```
Denoised-Neural-Portfolio-Optimizer/
├── configs/
│   └── model.yaml                  # Markowitz layer hyperparameters
├── src/portfolio/
│   ├── covariance/
│   │   ├── denoise.py              # Marchenko-Pastur spectral filter
│   │   └── estimators.py           # Sample, Ledoit-Wolf, EWMA
│   ├── data/
│   │   ├── download.py             # yfinance data downloader
│   │   ├── features.py             # Feature engineering
│   │   └── preprocess.py           # Preprocessing utilities
│   ├── models/
│   │   ├── networks.py             # MLP, MHA, MLA architectures
│   │   ├── markowitz_layer.py      # Differentiable Markowitz + cap_weights
│   │   ├── wrapper.py              # DenoisedPortfolioWrapper
│   │   ├── losses.py               # SharpeLoss, ICCLoss, CombinedLoss
│   │   └── train.py                # Training loop (WIP)
│   ├── strategies/
│   │   ├── base.py                 # BaseStrategy interface
│   │   ├── classical.py            # EqualWeight, MinVar, MaxSharpe
│   │   ├── neural.py               # Neural strategy wrapper (WIP)
│   │   └── hrp.py                  # HRP strategy (WIP)
│   ├── backtest/                   # Backtesting engine
│   └── viz/                        # Plotting utilities
├── tests/
│   ├── model/
│   │   ├── test_markowitz.py
│   │   └── test_model_dims.py
│   ├── test_denoise.py
│   ├── test_strategies.py
│   ├── test_features.py
│   └── test_preprocess.py
├── notebooks/                      # Exploratory analysis
├── pyproject.toml
└── LICENSE
```

---

## Testing

```bash
pytest tests/
```

The test suite covers Markowitz layer correctness across all weight modes, neural network input/output dimensions, covariance denoising, classical strategies, and data utilities.

---

## Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit with [conventional commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `refactor:`, `test:`
4. Open a pull request

**Current open TODOs:**
- [ ] Straight-Through Estimator (STE) for `cap_weights` to prevent gradient zeroing
- [ ] Hyperparameter annealing module for `alpha` in `CombinedLoss`
- [ ] HRP strategy implementation
- [ ] Neural strategy wrapper (`strategies/neural.py`)
- [ ] Full training loop (`models/train.py`)

---

## License

Distributed under the [MIT License](LICENSE). © 2026 Tsimur Bardziyan.
