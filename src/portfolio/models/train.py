import logging
from pathlib import Path
from typing import Optional

import torch
import yaml

from portfolio.models.networks import DeepMarkowitz_MHA, DeepMarkowitz_MLP, DeepMarkowitz_MLA
from portfolio.models.dataloader import return_dataloader
from portfolio.models.losses import CombinedLoss
from portfolio.models.wrapper import DenoisedPortfolioWrapper
from portfolio.covariance.estimators import (
    sample_covariance,
    ledoit_wolf_covariance,
    ewma_covariance,
)
from portfolio.covariance.denoise import denoise_covariance

logger = logging.getLogger(__name__)

# Registry of supported covariance estimators from portfolio.covariance.estimators
COVARIANCE_ESTIMATORS = {
    "sample": sample_covariance,
    "ledoit_wolf": ledoit_wolf_covariance,
    "ewma": ewma_covariance,
}


def _load_config(config_path: str = "configs/model.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_model(
    model: DenoisedPortfolioWrapper,
    features_path: str,
    prices_path: str,
    epochs: int,
    batch_size: int = 32,
    alpha: float = 0.2,
    cov_estimator: str = "ledoit_wolf",
    denoise: bool = True,
    lookback_days: int = 252,
    half_life: int = 63,
) -> DenoisedPortfolioWrapper:
    """Train a DenoisedPortfolioWrapper for one full run.

    Args:
        model: DenoisedPortfolioWrapper instance (backbone + Markowitz layer).
        features_path: Path to features tensor (.pt or .npz).
        prices_path: Path to prices parquet file.
        epochs: Number of training epochs.
        batch_size: Batch size for DataLoader.
        alpha: Weight for SharpeLoss vs ICCLoss in CombinedLoss.
        cov_estimator: Covariance estimator from estimators.py ('sample', 'ledoit_wolf', 'ewma').
        denoise: Whether to apply Marchenko-Pastur spectral denoising from denoise.py.
        lookback_days: Trailing window size for rolling covariance estimation.
        half_life: Decay half-life for EWMA covariance.
    """
    train_loader = return_dataloader(
        features_path=features_path,
        prices_path=prices_path,
        batch_size=batch_size,
        split="train",
        lookback_days=lookback_days,
        cov_estimator=cov_estimator,
        denoise=denoise,
        half_life=half_life,
    )
    criterion = CombinedLoss(alpha=alpha)

    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1e-3},
        {"params": criterion.parameters(), "lr": 1e-4},
    ])

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X, cov, y in train_loader:  # X:(B,N,F)  cov:(B,N,N) from estimators.py  y:(B,N)
            optimizer.zero_grad()

            # Wrapper feeds expected returns mu and covariance matrix cov into markowitz_layer
            predicted_mu, weights = model(X, cov)

            loss = criterion(weights, y, predicted_mu, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            logger.info("Epoch %d  avg loss: %.6f", epoch, epoch_loss / len(train_loader))

    return model


def evaluate_model(
    model: DenoisedPortfolioWrapper,
    features_path: str,
    prices_path: str,
    batch_size: int = 32,
    alpha: float = 0.2,
    cov_estimator: str = "ledoit_wolf",
    denoise: bool = True,
    lookback_days: int = 252,
    half_life: int = 63,
) -> None:
    """Evaluate a trained model on the test split using the specified covariance estimator."""
    test_loader = return_dataloader(
        features_path=features_path,
        prices_path=prices_path,
        batch_size=batch_size,
        split="test",
        lookback_days=lookback_days,
        cov_estimator=cov_estimator,
        denoise=denoise,
        half_life=half_life,
    )
    criterion = CombinedLoss(alpha=alpha)

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, cov, y in test_loader:
            predicted_mu, weights = model(X, cov)
            loss = criterion(weights, y, predicted_mu, y)
            total_loss += loss.item()

    logger.info("Test avg loss: %.6f", total_loss / len(test_loader))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = _load_config("configs/model.yaml")
    features_path = config["features_path"]
    prices_path = config["prices_path"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    alpha = config["alpha"]

    # Covariance configuration from model.yaml
    cov_estimator = config.get("cov_estimator", "ledoit_wolf")
    denoise = config.get("denoise", True)
    lookback_days = config.get("lookback_days", 252)
    half_life = config.get("half_life", 63)

    logger.info(
        "Using covariance estimator: %s (denoise=%s, lookback_days=%d)",
        cov_estimator,
        denoise,
        lookback_days,
    )

    backbones = {
        "DeepMarkowitz_MLP": DeepMarkowitz_MLP(),
        "DeepMarkowitz_MHA": DeepMarkowitz_MHA(),
        "DeepMarkowitz_MLA": DeepMarkowitz_MLA(),
    }

    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    for name, backbone in backbones.items():
        logger.info("=== Training %s ===", name)

        model = DenoisedPortfolioWrapper(config=config, architecture=backbone, covarience=COVARIANCE_ESTIMATORS.get(cov_estimator))
        model = train_model(
            model=model,
            features_path=features_path,
            prices_path=prices_path,
            epochs=epochs,
            batch_size=batch_size,
            alpha=alpha,
            cov_estimator=cov_estimator,
            denoise=denoise,
            lookback_days=lookback_days,
            half_life=half_life,
        )

        save_path = save_dir / f"{name}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info("Saved %s → %s", name, save_path)

        logger.info("=== Evaluating %s ===", name)
        evaluate_model(
            model=model,
            features_path=features_path,
            prices_path=prices_path,
            batch_size=batch_size,
            alpha=alpha,
            cov_estimator=cov_estimator,
            denoise=denoise,
            lookback_days=lookback_days,
            half_life=half_life,
        )
