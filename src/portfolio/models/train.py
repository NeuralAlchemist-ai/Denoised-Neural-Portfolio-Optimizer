import copy
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from portfolio.models.networks import DeepMarkowitz_MHA, DeepMarkowitz_MLP, DeepMarkowitz_MLA
from portfolio.models.dataloader import Features_Dataset, return_dataloader
from portfolio.models.losses import CombinedLoss
from portfolio.models.wrapper import DenoisedPortfolioWrapper
from portfolio.covariance.estimators import (
    sample_covariance,
    ledoit_wolf_covariance,
    ewma_covariance,
)
from portfolio.covariance.denoise import denoise_covariance

logger = logging.getLogger(__name__)

BACKBONE_REGISTRY = {
    "DeepMarkowitz_MLP": DeepMarkowitz_MLP,
    "DeepMarkowitz_MHA": DeepMarkowitz_MHA,
    "DeepMarkowitz_MLA": DeepMarkowitz_MLA,
}

COVARIANCE_ESTIMATORS = {
    "sample": sample_covariance,
    "ledoit_wolf": ledoit_wolf_covariance,
    "ewma": ewma_covariance,
}


def _load_config(config_path: str = "configs/model.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Optional[dict[str, Any]],
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Instantiate a PyTorch learning rate scheduler from configuration."""
    if not scheduler_config:
        return None

    sched_type = scheduler_config.get("type", "ReduceLROnPlateau")
    if sched_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 8),
            min_lr=scheduler_config.get("min_lr", 1e-5),
        )
    elif sched_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("T_max", 50),
            eta_min=scheduler_config.get("min_lr", 1e-5),
        )
    elif sched_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 20),
            gamma=scheduler_config.get("factor", 0.5),
        )
    else:
        logger.warning("Unsupported scheduler type '%s'. Proceeding without scheduler.", sched_type)
        return None


def evaluate_loader(
    model: DenoisedPortfolioWrapper,
    dataloader: DataLoader,
    criterion: CombinedLoss,
) -> tuple[float, float, float]:
    """Compute average (combined_loss, sharpe_loss, ic_loss) over a given DataLoader under torch.no_grad()."""
    model.eval()
    total_combined = 0.0
    total_sharpe = 0.0
    total_ic = 0.0
    num_batches = len(dataloader)
    if num_batches == 0:
        return 0.0, 0.0, 0.0

    with torch.no_grad():
        for X, cov, y in dataloader:
            predicted_mu, weights = model(X, cov)
            combined_loss, sharpe_loss, ic_loss = criterion(weights, y, predicted_mu, y)
            total_combined += combined_loss.item()
            total_sharpe += sharpe_loss.item()
            total_ic += ic_loss.item()

    return total_combined / num_batches, total_sharpe / num_batches, total_ic / num_batches


def train_model(
    model: DenoisedPortfolioWrapper,
    features_path: Optional[str] = None,
    prices_path: Optional[str] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    batch_size: int = 32,
    alpha: float = 0.2,
    lr: float = 1e-3,
    scheduler_config: Optional[dict[str, Any]] = None,
    early_stopping_patience: int = 15,
    cov_estimator: str = "sample",
    denoise: bool = True,
    lookback_days: int = 252,
    half_life: int = 63,
    train_end: str = "2022-12-31",
    val_ratio: float = 0.2,
    shuffle_train: bool = True,
) -> DenoisedPortfolioWrapper:
    """Train DenoisedPortfolioWrapper with validation evaluation, LR scheduler, and early stopping.

    Args:
        model: DenoisedPortfolioWrapper instance.
        features_path: Path to features tensor (.pt or .npz), used if train_loader is None.
        prices_path: Path to prices parquet file, used if train_loader is None.
        train_loader: Pre-built training DataLoader (optional).
        val_loader: Pre-built validation DataLoader (optional).
        epochs: Number of training epochs.
        batch_size: Batch size for DataLoader if constructing loaders.
        alpha: Weight for SharpeLoss vs ICCLoss (fixed hyperparameter, not trained).
        lr: Initial optimizer learning rate.
        scheduler_config: Config dictionary for learning rate scheduler.
        early_stopping_patience: Epochs without val_loss improvement before stopping.
        cov_estimator: Covariance estimator function name ('sample', 'ledoit_wolf', 'ewma').
        denoise: Whether to apply Marchenko-Pastur denoising.
        lookback_days: Trailing window for rolling covariance.
        half_life: EWMA half-life if cov_estimator is 'ewma'.
        train_end: Date separating pre-test and test periods.
        val_ratio: Fraction of pre-test period used for validation.
        shuffle_train: Whether to shuffle training batches (default: True).
    """
    # ── Build DataLoaders if not provided ─────────────────────────────────
    if train_loader is None:
        if features_path is None or prices_path is None:
            raise ValueError("features_path and prices_path are required if train_loader is None.")
        train_loader = return_dataloader(
            features_path=features_path,
            prices_path=prices_path,
            batch_size=batch_size,
            split="train",
            train_end=train_end,
            val_ratio=val_ratio,
            lookback_days=lookback_days,
            cov_estimator=cov_estimator,
            denoise=denoise,
            half_life=half_life,
            shuffle=shuffle_train,
        )

    if val_loader is None and features_path is not None and prices_path is not None:
        val_loader = return_dataloader(
            features_path=features_path,
            prices_path=prices_path,
            batch_size=batch_size,
            split="val",
            train_end=train_end,
            val_ratio=val_ratio,
            lookback_days=lookback_days,
            cov_estimator=cov_estimator,
            denoise=denoise,
            half_life=half_life,
            shuffle=False,
        )

    criterion = CombinedLoss(alpha=alpha)
    # alpha is kept as a fixed hyperparameter: only model parameters are optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = build_scheduler(optimizer, scheduler_config)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_combined = 0.0
        epoch_sharpe = 0.0
        epoch_ic = 0.0

        for X, cov, y in train_loader:
            optimizer.zero_grad()
            predicted_mu, weights = model(X, cov)
            combined_loss, sharpe_loss, ic_loss = criterion(weights, y, predicted_mu, y)
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_combined += combined_loss.item()
            epoch_sharpe += sharpe_loss.item()
            epoch_ic += ic_loss.item()

        n_batches = max(1, len(train_loader))
        avg_train_combined = epoch_combined / n_batches
        avg_train_sharpe = epoch_sharpe / n_batches
        avg_train_ic = epoch_ic / n_batches

        # Evaluate on validation set
        if val_loader is not None:
            avg_val_combined, avg_val_sharpe, avg_val_ic = evaluate_loader(model, val_loader, criterion)
        else:
            avg_val_combined = avg_train_combined
            avg_val_sharpe = avg_train_sharpe
            avg_val_ic = avg_train_ic

        # Step LR scheduler (ReduceLROnPlateau uses val combined loss)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_combined)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == epochs - 1:
            val_msg = (
                f" | Val: [Combined: {avg_val_combined:.4f} | Sharpe: {avg_val_sharpe:.4f} | IC: {avg_val_ic:.4f}]"
                if val_loader is not None
                else ""
            )
            logger.info(
                "Epoch %3d | Train: [Combined: %.4f | Sharpe: %.4f | IC: %.4f]%s | LR: %.2e",
                epoch,
                avg_train_combined,
                avg_train_sharpe,
                avg_train_ic,
                val_msg,
                current_lr,
            )

        # Early stopping & best model tracking based on combined validation loss
        if avg_val_combined < best_val_loss:
            best_val_loss = avg_val_combined
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                logger.info(
                    "Early stopping triggered at epoch %d (best val combined loss: %.6f)",
                    epoch,
                    best_val_loss,
                )
                break

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model


def time_series_cross_validation(
    backbone_name: str,
    config: dict[str, Any],
    features_path: str,
    prices_path: str,
    n_splits: int = 5,
    cv_epochs: int = 40,
    batch_size: int = 32,
    alpha: float = 0.2,
    lr: float = 1e-3,
    scheduler_config: Optional[dict[str, Any]] = None,
    cov_estimator: str = "sample",
    denoise: bool = True,
    lookback_days: int = 252,
    half_life: int = 63,
    train_end: str = "2022-12-31",
    shuffle_train: bool = True,
) -> dict[str, Any]:
    """Perform walk-forward time-series cross-validation without look-ahead leakage.

    Folds are expanding: fold k trains on blocks [0 .. k] and validates on block [k+1].
    """
    logger.info("--- Starting %d-fold Time-Series CV for %s ---", n_splits, backbone_name)

    # 1. Load full pre-test dates to compute chronological index folds
    full_dataset = Features_Dataset(
        features_path=features_path,
        prices_path=prices_path,
        split=None,
        lookback_days=lookback_days,
        cov_estimator=cov_estimator,
        denoise=denoise,
        half_life=half_life,
    )
    pre_test_mask = (full_dataset.dates <= pd.Timestamp(train_end))
    pre_test_indices = np.where(pre_test_mask)[0]
    total_samples = len(pre_test_indices)

    fold_size = total_samples // (n_splits + 1)
    if fold_size < 10:
        raise ValueError(
            f"Insufficient samples ({total_samples}) for {n_splits} splits; fold_size is {fold_size}."
        )

    fold_combined_scores = []
    fold_sharpe_scores = []
    fold_ic_scores = []
    criterion = CombinedLoss(alpha=alpha)

    for fold in range(n_splits):
        train_end_idx = (fold + 1) * fold_size
        val_end_idx = (fold + 2) * fold_size

        train_indices = pre_test_indices[:train_end_idx]
        val_indices = pre_test_indices[train_end_idx:val_end_idx]

        train_loader = return_dataloader(
            features_path=features_path,
            prices_path=prices_path,
            batch_size=batch_size,
            indices=train_indices,
            lookback_days=lookback_days,
            cov_estimator=cov_estimator,
            denoise=denoise,
            half_life=half_life,
            shuffle=shuffle_train,
        )
        val_loader = return_dataloader(
            features_path=features_path,
            prices_path=prices_path,
            batch_size=batch_size,
            indices=val_indices,
            lookback_days=lookback_days,
            cov_estimator=cov_estimator,
            denoise=denoise,
            half_life=half_life,
            shuffle=False,
        )

        # Fresh model instance per fold
        backbone_cls = BACKBONE_REGISTRY[backbone_name]
        fold_model = DenoisedPortfolioWrapper(config=config, architecture=backbone_cls())

        train_model(
            model=fold_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cv_epochs,
            alpha=alpha,
            lr=lr,
            scheduler_config=scheduler_config,
            early_stopping_patience=10,
        )

        val_combined, val_sharpe, val_ic = evaluate_loader(fold_model, val_loader, criterion)
        fold_combined_scores.append(val_combined)
        fold_sharpe_scores.append(val_sharpe)
        fold_ic_scores.append(val_ic)

        val_dates = full_dataset.dates[val_indices]
        logger.info(
            "Fold %d/%d [%s to %s] -> Combined: %.4f | Sharpe: %.4f | IC: %.4f",
            fold + 1,
            n_splits,
            val_dates[0].strftime("%Y-%m-%d"),
            val_dates[-1].strftime("%Y-%m-%d"),
            val_combined,
            val_sharpe,
            val_ic,
        )

    mean_cv_combined = float(np.mean(fold_combined_scores))
    std_cv_combined = float(np.std(fold_combined_scores))
    mean_cv_sharpe = float(np.mean(fold_sharpe_scores))
    mean_cv_ic = float(np.mean(fold_ic_scores))

    logger.info(
        "=== %s CV Result: Mean Combined = %.4f (± %.4f) | Mean Sharpe = %.4f | Mean IC = %.4f ===",
        backbone_name,
        mean_cv_combined,
        std_cv_combined,
        mean_cv_sharpe,
        mean_cv_ic,
    )

    return {
        "backbone": backbone_name,
        "fold_combined_scores": fold_combined_scores,
        "fold_sharpe_scores": fold_sharpe_scores,
        "fold_ic_scores": fold_ic_scores,
        "mean_val_loss": mean_cv_combined,
        "std_val_loss": std_cv_combined,
        "mean_sharpe_loss": mean_cv_sharpe,
        "mean_ic_loss": mean_cv_ic,
    }


def evaluate_model(
    model: DenoisedPortfolioWrapper,
    features_path: str,
    prices_path: str,
    batch_size: int = 32,
    alpha: float = 0.2,
    cov_estimator: str = "sample",
    denoise: bool = True,
    lookback_days: int = 252,
    half_life: int = 63,
    test_start: str = "2023-01-01",
) -> tuple[float, float, float]:
    """Evaluate a trained model on the holdout test split and return (combined, sharpe, ic)."""
    test_loader = return_dataloader(
        features_path=features_path,
        prices_path=prices_path,
        batch_size=batch_size,
        split="test",
        test_start=test_start,
        lookback_days=lookback_days,
        cov_estimator=cov_estimator,
        denoise=denoise,
        half_life=half_life,
        shuffle=False,
    )
    criterion = CombinedLoss(alpha=alpha)
    test_combined, test_sharpe, test_ic = evaluate_loader(model, test_loader, criterion)
    logger.info(
        "Holdout Test -> Combined Loss: %.6f | Sharpe Loss: %.6f | IC Loss: %.6f",
        test_combined,
        test_sharpe,
        test_ic,
    )
    return test_combined, test_sharpe, test_ic


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = _load_config("configs/model.yaml")
    features_path = config["features_path"]
    prices_path = config["prices_path"]
    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 32)
    alpha = config.get("alpha", 0.2)
    lr = float(config.get("lr", 1e-3))
    val_ratio = float(config.get("validation_size", 0.2))

    # Covariance configuration
    cov_estimator = config.get("cov_estimator", "sample")
    denoise = config.get("denoise", True)
    lookback_days = config.get("lookback_days", 252)
    half_life = config.get("half_life", 63)

    # LR Scheduler config
    scheduler_config = config.get("scheduler", {
        "type": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 8,
        "min_lr": 1e-5,
    })

    # Cross-validation config
    cv_config = config.get("cross_validation", {})
    cv_enabled = cv_config.get("enabled", True)
    n_splits = cv_config.get("n_splits", 5)
    cv_epochs = cv_config.get("cv_epochs", 40)

    # Backbone selection
    configured_backbones = config.get("backbones", list(BACKBONE_REGISTRY.keys()))

    logger.info(
        "Covariance Estimator: %s | Denoise: %s | Lookback: %d | Batch Size: %d",
        cov_estimator,
        denoise,
        lookback_days,
        batch_size,
    )

    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)

    cv_results = {}

    for name in configured_backbones:
        if name not in BACKBONE_REGISTRY:
            logger.warning("Backbone '%s' not recognized, skipping.", name)
            continue

        # ── 1. Walk-Forward Cross-Validation ──────────────────────────────
        if cv_enabled:
            cv_res = time_series_cross_validation(
                backbone_name=name,
                config=config,
                features_path=features_path,
                prices_path=prices_path,
                n_splits=n_splits,
                cv_epochs=cv_epochs,
                batch_size=batch_size,
                alpha=alpha,
                lr=lr,
                scheduler_config=scheduler_config,
                cov_estimator=cov_estimator,
                denoise=denoise,
                lookback_days=lookback_days,
                half_life=half_life,
                shuffle_train=True,
            )
            cv_results[name] = cv_res

        # ── 2. Full Train on Train/Val split with Scheduler & Early Stopping
        logger.info("=== Full Training & Test Evaluation: %s ===", name)
        backbone_instance = BACKBONE_REGISTRY[name]()
        model = DenoisedPortfolioWrapper(config=config, architecture=backbone_instance)

        model = train_model(
            model=model,
            features_path=features_path,
            prices_path=prices_path,
            epochs=epochs,
            batch_size=batch_size,
            alpha=alpha,
            lr=lr,
            scheduler_config=scheduler_config,
            cov_estimator=cov_estimator,
            denoise=denoise,
            lookback_days=lookback_days,
            half_life=half_life,
            val_ratio=val_ratio,
            shuffle_train=True,
        )

        save_path = save_dir / f"{name}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info("Saved trained model %s → %s", name, save_path)

        # ── 3. Holdout Test Evaluation ────────────────────────────────────
        test_combined, test_sharpe, test_ic = evaluate_model(
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

    if cv_results:
        logger.info("================ CV Summary ================")
        for model_name, res in cv_results.items():
            logger.info(
                "%s | Mean Combined: %.4f (± %.4f) | Mean Sharpe: %.4f | Mean IC: %.4f",
                model_name,
                res["mean_val_loss"],
                res["std_val_loss"],
                res["mean_sharpe_loss"],
                res["mean_ic_loss"],
            )
