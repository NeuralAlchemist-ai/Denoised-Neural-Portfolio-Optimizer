import logging
from pathlib import Path
from typing import Callable, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from portfolio.data.features import load_features
from portfolio.data.preprocess import compute_simple_returns
from portfolio.covariance.estimators import (
    sample_covariance,
    ledoit_wolf_covariance,
    ewma_covariance,
)
from portfolio.covariance.denoise import denoise_covariance

logger = logging.getLogger(__name__)

COVARIANCE_ESTIMATORS: dict[str, Callable[..., np.ndarray]] = {
    "sample": sample_covariance,
    "ledoit_wolf": ledoit_wolf_covariance,
    "ewma": ewma_covariance,
}


class Features_Dataset(Dataset):
    """
    One sample = (features at day t, covariance at day t, simple returns at day t+1).

    The model sees today's features + covariance matrix and is trained to allocate
    weights that perform well on *tomorrow's* returns. This strictly prevents
    look-ahead data leakage.

    Args:
        features_path: Path to the saved features tensor (.pt or .npz).
        prices_path:   Path to raw prices parquet, used to compute returns & covariance.
        split:         'train' | 'test' | None (uses full dataset).
        train_end:     Last date (inclusive) for the training split, e.g. '2022-12-31'.
        test_start:    First date (inclusive) for the test split, e.g. '2023-01-01'.
        lookback_days: Rolling window for covariance estimation (default 252).
        cov_estimator: Covariance estimator to use: 'ledoit_wolf', 'sample', or 'ewma'.
        denoise:       If True, apply Marchenko-Pastur spectral denoising to the covariance matrix.
        half_life:     Decay half-life in trading days if cov_estimator is 'ewma'.
    """

    def __init__(
        self,
        features_path: str = "data/processed/features_tensor.pt",
        prices_path: str = "data/raw/prices.parquet",
        split: str | None = None,
        train_end: str = "2022-12-31",
        test_start: str = "2023-01-01",
        lookback_days: int = 252,
        cov_estimator: str = "ledoit_wolf",
        denoise: bool = False,
        half_life: int = 63,
    ):
        # ── 1. Load feature tensor (T, N, F) ──────────────────────────────
        data = load_features(features_path)
        features: torch.Tensor = data["features"]          # (T, N, F)
        dates: pd.DatetimeIndex = data["dates"]            # length T

        # ── 2. Load prices → simple returns ───────────────────────────────
        prices_df = pd.read_parquet(prices_path)
        returns_df = compute_simple_returns(prices_df)     # (T_raw-1, N)

        # ── 3. Align returns to the feature date index ────────────────────
        # features[t] uses prices up to dates[t].
        # Label for features[t] = NEXT day's return → shift by 1.
        # We drop the LAST feature row because there is no t+1 return for it.
        returns_aligned = returns_df.reindex(dates).to_numpy(dtype=np.float32)
        self.X = features[:-1]                                      # (T-1, N, F)
        self.y = torch.from_numpy(returns_aligned[1:])              # (T-1, N)
        self.dates = dates[:-1]                                     # (T-1,)

        # ── 4. Rolling covariance matrices (T-1, N, N) using estimators.py ─
        if cov_estimator not in COVARIANCE_ESTIMATORS:
            raise ValueError(
                f"Unknown covariance estimator '{cov_estimator}'. "
                f"Supported options: {list(COVARIANCE_ESTIMATORS.keys())}"
            )
        estimator_fn = COVARIANCE_ESTIMATORS[cov_estimator]

        cov_list = []
        full_dates = returns_df.index

        for date in self.dates:
            pos = full_dates.get_loc(date)
            start = max(0, pos - lookback_days + 1)
            window_df = returns_df.iloc[start : pos + 1]

            if cov_estimator == "ewma":
                cov = estimator_fn(window_df, half_life=half_life)
            else:
                cov = estimator_fn(window_df)

            if denoise and len(window_df) >= window_df.shape[1]:
                cov = denoise_covariance(cov, n_obs=len(window_df))

            cov_list.append(cov.astype(np.float32))

        self.cov = torch.from_numpy(np.stack(cov_list, axis=0))     # (T-1, N, N)

        # ── 5. Optional train/test split ──────────────────────────────────
        if split is not None:
            mask = self._split_mask(split, train_end, test_start)
            self.X = self.X[mask]
            self.y = self.y[mask]
            self.cov = self.cov[mask]
            self.dates = self.dates[mask]

    # ------------------------------------------------------------------
    def _split_mask(self, split: str, train_end: str, test_start: str) -> torch.Tensor:
        dates_series = pd.Series(self.dates)
        if split == "train":
            mask = dates_series <= pd.Timestamp(train_end)
        elif split == "test":
            mask = dates_series >= pd.Timestamp(test_start)
        else:
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")
        return torch.tensor(mask.to_numpy())

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x:   Feature tensor  (N, F)   — today's signals.
            cov: Covariance      (N, N)   — covariance estimated via estimators.py.
            y:   Returns         (N,)     — tomorrow's asset returns (label).
        """
        return self.X[idx], self.cov[idx], self.y[idx]


def return_dataloader(
    features_path: str,
    prices_path: str,
    batch_size: int = 32,
    split: str = "train",
    lookback_days: int = 252,
    cov_estimator: str = "ledoit_wolf",
    denoise: bool = False,
    half_life: int = 63,
) -> DataLoader:
    dataset = Features_Dataset(
        features_path=features_path,
        prices_path=prices_path,
        split=split,
        lookback_days=lookback_days,
        cov_estimator=cov_estimator,
        denoise=denoise,
        half_life=half_life,
    )
    # shuffle=False is correct: time-series order must be preserved.
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
