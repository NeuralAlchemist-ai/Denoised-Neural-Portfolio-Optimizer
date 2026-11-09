import logging
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

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
        split:         'train' | 'val' | 'validation' | 'test' | None (uses full dataset).
        train_end:     Last date (inclusive) for the training split, e.g. '2022-12-31'.
        val_end:       Last date (inclusive) for validation split. If None, val_ratio is used.
        test_start:    First date (inclusive) for the test split, e.g. '2023-01-01'.
        val_ratio:     Fraction of pre-test period used for validation if val_end is None.
        indices:       Explicit sequence of integer indices to select (for cross-validation).
        lookback_days: Rolling window for covariance estimation (default 252).
        cov_estimator: Covariance estimator to use: 'ledoit_wolf', 'sample', or 'ewma'.
        denoise:       If True, apply Marchenko-Pastur spectral denoising to the covariance matrix.
        half_life:     Decay half-life in trading days if cov_estimator is 'ewma'.
    """

    def __init__(
        self,
        features_path: str = "data/processed/features_tensor.pt",
        prices_path: str = "data/raw/prices.parquet",
        split: Optional[str] = None,
        train_end: str = "2022-12-31",
        val_end: Optional[str] = None,
        test_start: str = "2023-01-01",
        val_ratio: float = 0.2,
        indices: Optional[Sequence[int]] = None,
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
        returns_aligned = returns_df.reindex(dates).to_numpy(dtype=np.float32, copy=True)
        self.X = features[:-1].clone()
        self.y = torch.from_numpy(returns_aligned[1:].copy())       # (T-1, N)
        self.dates: pd.DatetimeIndex = dates[:-1]                   # (T-1,)

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

        # ── 5. Index-based selection (for cross-validation) ───────────────
        if indices is not None:
            idx_arr = np.asarray(indices, dtype=int)
            idx_tensor = torch.from_numpy(idx_arr)
            self.X = self.X[idx_tensor]
            self.y = self.y[idx_tensor]
            self.cov = self.cov[idx_tensor]
            self.dates = self.dates[idx_arr]

        # ── 6. Split selection ('train' | 'val' | 'test') ─────────────────
        elif split is not None:
            mask_np = self._split_mask(
                split=split,
                train_end=train_end,
                val_end=val_end,
                test_start=test_start,
                val_ratio=val_ratio,
            )
            mask_tensor = torch.from_numpy(mask_np)
            self.X = self.X[mask_tensor]
            self.y = self.y[mask_tensor]
            self.cov = self.cov[mask_tensor]
            self.dates = self.dates[mask_np]

    # ------------------------------------------------------------------
    def _split_mask(
        self,
        split: str,
        train_end: str,
        val_end: Optional[str],
        test_start: str,
        val_ratio: float,
    ) -> np.ndarray:
        """Compute chronological boolean mask for train, validation, or test."""
        split_key = split.lower()
        if split_key == "test":
            mask = self.dates >= pd.Timestamp(test_start)
            return np.asarray(mask, dtype=bool)

        if val_end is not None:
            # Explicit date-based split
            if split_key == "train":
                mask = self.dates <= pd.Timestamp(train_end)
            elif split_key in ("val", "validation"):
                mask = (self.dates > pd.Timestamp(train_end)) & (self.dates <= pd.Timestamp(val_end))
            else:
                raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val', or 'test'.")
            return np.asarray(mask, dtype=bool)

        # Ratio-based train/val split within the pre-test sample
        pre_test_mask = (self.dates <= pd.Timestamp(train_end))
        pre_test_indices = np.where(pre_test_mask)[0]
        if len(pre_test_indices) == 0:
            raise ValueError(f"No dates found before train_end='{train_end}'.")

        n_total = len(pre_test_indices)
        n_val = max(1, int(np.floor(n_total * val_ratio)))
        n_train = n_total - n_val

        mask = np.zeros(len(self.dates), dtype=bool)
        if split_key == "train":
            mask[pre_test_indices[:n_train]] = True
        elif split_key in ("val", "validation"):
            mask[pre_test_indices[n_train:]] = True
        else:
            raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val', or 'test'.")

        return mask

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x:   Feature tensor  (N, F)   — today's signals.
            cov: Covariance      (N, N)   — covariance estimated via estimators.py.
            y:   Returns         (N,)     — tomorrow's asset returns (label).
        """
        idx_int = int(idx)
        return self.X[idx_int], self.cov[idx_int], self.y[idx_int]


def return_dataloader(
    features_path: str,
    prices_path: str,
    batch_size: int = 32,
    split: Optional[str] = "train",
    train_end: str = "2022-12-31",
    val_end: Optional[str] = None,
    test_start: str = "2023-01-01",
    val_ratio: float = 0.2,
    indices: Optional[Sequence[int]] = None,
    lookback_days: int = 252,
    cov_estimator: str = "ledoit_wolf",
    denoise: bool = False,
    half_life: int = 63,
    shuffle: bool = False
) -> DataLoader:
    """Return a DataLoader with optional train/val/test or index-based splitting."""
    dataset = Features_Dataset(
        features_path=features_path,
        prices_path=prices_path,
        split=split,
        train_end=train_end,
        val_end=val_end,
        test_start=test_start,
        val_ratio=val_ratio,
        indices=indices,
        lookback_days=lookback_days,
        cov_estimator=cov_estimator,
        denoise=denoise,
        half_life=half_life,
    )
    # shuffle=False is strictly preserved to prevent time-series data leakage
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
