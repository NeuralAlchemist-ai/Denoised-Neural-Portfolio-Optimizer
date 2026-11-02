import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_EPSILON: float = 1e-8
_FEATURE_LOOKBACK_DAYS: dict[str, int] = {
    "mom_5d": 5,
    "mom_21d": 21,
    "mom_63d": 63,
    "mom_126d": 126,
    "mom_252d": 252,
    "vol_21d": 21,
    "vol_63d": 63,
    "downside_vol_21d": 21,
    "sharpe_63d": 63,
    "price_to_sma252": 252,
}


def _load_feature_config() -> dict[str, Any]:
    """Load and validate the feature-pipeline configuration."""
    config_path = Path(__file__).resolve().parents[3] / "configs" / "data.yaml"
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as error:
        raise RuntimeError(f"Unable to read feature configuration: {config_path}") from error

    feature_config = config.get("feature_pipeline") if isinstance(config, dict) else None
    if not isinstance(feature_config, dict):
        raise ValueError("data.yaml must define a 'feature_pipeline' mapping.")

    names = feature_config.get("feature_names")
    clip_value = feature_config.get("clip_value")
    warmup_days = feature_config.get("warmup_days")
    trading_days = feature_config.get("trading_days_per_year")

    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("feature_pipeline.feature_names must be a non-empty unique string list.")
    unknown_names = set(names).difference(_FEATURE_LOOKBACK_DAYS)
    if unknown_names:
        raise ValueError(f"Unsupported configured feature names: {sorted(unknown_names)}")
    if not isinstance(clip_value, (int, float)) or not np.isfinite(clip_value) or clip_value <= 0:
        raise ValueError("feature_pipeline.clip_value must be a positive finite number.")
    if not isinstance(warmup_days, int) or isinstance(warmup_days, bool) or warmup_days < 1:
        raise ValueError("feature_pipeline.warmup_days must be a positive integer.")
    required_warmup = max(_FEATURE_LOOKBACK_DAYS[name] for name in names)
    if warmup_days < required_warmup:
        raise ValueError(
            "feature_pipeline.warmup_days must be at least the longest configured "
            f"feature lookback ({required_warmup})."
        )
    if not isinstance(trading_days, int) or isinstance(trading_days, bool) or trading_days < 1:
        raise ValueError("feature_pipeline.trading_days_per_year must be a positive integer.")

    return feature_config


_FEATURE_CONFIG = _load_feature_config()
FEATURE_NAMES: list[str] = _FEATURE_CONFIG["feature_names"]
_DEFAULT_CLIP_VALUE: float = float(_FEATURE_CONFIG["clip_value"])
_DEFAULT_WARMUP_DAYS: int = _FEATURE_CONFIG["warmup_days"]
_TRADING_DAYS_PER_YEAR: int = _FEATURE_CONFIG["trading_days_per_year"]


def _validate_prices(prices: pd.DataFrame) -> None:
    """Validate the complete, point-in-time price panel required by this module."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Prices must be a non-empty pandas DataFrame.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices must use a DatetimeIndex.")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("Prices index must be sorted and contain unique timestamps.")
    if prices.columns.empty or not prices.columns.is_unique:
        raise ValueError("Prices columns must be non-empty and unique.")

    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Prices must not contain missing or non-finite values.")
    if (values <= 0.0).any():
        raise ValueError("Prices must be strictly positive.")


def _align_returns(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and align supplied simple returns to the price index."""
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a pandas DataFrame.")
    if list(returns.columns) != list(prices.columns):
        raise ValueError("Returns columns must match prices columns in the same order.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns must use a DatetimeIndex.")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise ValueError("Returns index must be sorted and contain unique timestamps.")

    expected_indices = (prices.index, prices.index[1:])
    if not any(returns.index.equals(index) for index in expected_indices):
        raise ValueError(
            "Returns index must equal the price index or the price index without its first date."
        )

    aligned = returns.reindex(prices.index)
    values = aligned.iloc[1:].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Returns must not contain missing or non-finite values after alignment.")
    return aligned


def _validate_feature_frames(features: dict[str, pd.DataFrame]) -> None:
    """Ensure feature frames share a non-empty, aligned panel."""
    if not features:
        raise ValueError("Features cannot be empty.")

    first = next(iter(features.values()))
    if not isinstance(first, pd.DataFrame) or first.empty:
        raise ValueError("Each feature must be a non-empty pandas DataFrame.")
    for name, frame in features.items():
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"Feature '{name}' must be a pandas DataFrame.")
        if not frame.index.equals(first.index) or not frame.columns.equals(first.columns):
            raise ValueError("All feature DataFrames must have identical indexes and columns.")


def compute_raw_features(
    prices: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    """Compute dictionary of raw feature DataFrames, each of shape (T, N).

    Args:
        prices: Asset price series of shape (T, N) with DatetimeIndex.
        returns: Optional arithmetic simple returns of shape (T, N). If None,
            computed from `prices.pct_change()`.

    Returns:
        Dictionary mapping feature name to DataFrame of shape (T, N).

    Raises:
        ValueError: If the price panel or supplied returns violate the data contract.
    """
    _validate_prices(prices)

    if returns is None:
        returns = prices.pct_change(fill_method=None)
    else:
        returns = _align_returns(prices, returns)

    # 1. Multi-horizon Momentum (Returns)
    mom_5d = prices.pct_change(5)
    mom_21d = prices.pct_change(21)
    mom_63d = prices.pct_change(63)
    mom_126d = prices.pct_change(126)
    mom_252d = prices.pct_change(252)

    # 2. Volatility & Downside Risk (annualized by configured trading calendar)
    annualization_factor = np.sqrt(_TRADING_DAYS_PER_YEAR)
    vol_21d = returns.rolling(window=21).std() * annualization_factor
    vol_63d = returns.rolling(window=63).std() * annualization_factor

    neg_returns = returns.clip(upper=0.0)
    downside_vol_21d = (
        neg_returns.pow(2).rolling(window=21).mean().pow(0.5) * annualization_factor
    )

    # 3. Quality & Trend Deviation
    rolling_mean_63d = returns.rolling(window=63).mean()
    rolling_std_63d = returns.rolling(window=63).std()
    sharpe_63d = (rolling_mean_63d / (rolling_std_63d + _EPSILON)) * annualization_factor

    price_to_sma252 = (prices / prices.rolling(window=252).mean()) - 1.0

    raw_features: dict[str, pd.DataFrame] = {
        "mom_5d": mom_5d,
        "mom_21d": mom_21d,
        "mom_63d": mom_63d,
        "mom_126d": mom_126d,
        "mom_252d": mom_252d,
        "vol_21d": vol_21d,
        "vol_63d": vol_63d,
        "downside_vol_21d": downside_vol_21d,
        "sharpe_63d": sharpe_63d,
        "price_to_sma252": price_to_sma252,
    }

    return raw_features


def normalize_features(
    features: dict[str, pd.DataFrame],
    clip_val: float = _DEFAULT_CLIP_VALUE,
) -> dict[str, pd.DataFrame]:
    """Cross-sectionally standardize and clip features per timestamp.

    Computes Z-score across assets (axis=1) for each date:
        z = (x - mean) / (std + eps)
    clipped to [-clip_val, clip_val] to eliminate extreme outliers.

    Args:
        features: Dictionary mapping feature name to DataFrame of shape (T, N).
        clip_val: Bound for symmetric winsorization/clipping.

    Returns:
        Dictionary mapping feature name to standardized DataFrame of shape (T, N).
    """
    if not np.isfinite(clip_val) or clip_val <= 0.0:
        raise ValueError("clip_val must be a positive finite number.")
    _validate_feature_frames(features)

    normalized: dict[str, pd.DataFrame] = {}

    for name, df in features.items():
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        z = df.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
        # A complete cross-section with no dispersion is a valid neutral signal.
        # Missing values deliberately remain missing and are rejected before tensorization.
        z.loc[std.eq(0.0)] = 0.0
        z_clipped = z.clip(lower=-clip_val, upper=clip_val)
        normalized[name] = z_clipped

    return normalized


def features_to_tensor(
    features: dict[str, pd.DataFrame],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    """Stack feature DataFrames into a dense 3D numpy array.

    Args:
        features: Dictionary mapping feature name to DataFrame of shape (T, N).
        feature_names: List specifying the exact ordering along the feature axis.

    Returns:
        3D float32 numpy array of shape (T, N, F).
    """
    _validate_feature_frames(features)
    if not feature_names:
        raise ValueError("feature_names cannot be empty.")
    missing_names = [name for name in feature_names if name not in features]
    if missing_names:
        raise ValueError(f"Requested features are missing: {missing_names}")

    ordered_arrays = [features[name].to_numpy(dtype=np.float32) for name in feature_names]
    tensor = np.stack(ordered_arrays, axis=-1)
    if not np.isfinite(tensor).all():
        raise ValueError(
            "Features contain missing or non-finite values. Increase warmup_days or fix input data."
        )
    return tensor


def extract_feature_tensor(
    prices: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None,
    clip_val: float = _DEFAULT_CLIP_VALUE,
    warmup_days: int = _DEFAULT_WARMUP_DAYS,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> tuple[np.ndarray, pd.DatetimeIndex, pd.Index]:
    """End-to-end extraction from prices to aligned, warmup-trimmed feature tensor.

    Args:
        prices: Asset price series of shape (T, N) with DatetimeIndex.
        returns: Optional arithmetic returns aligned to prices, either with the
            full price index or with the first price date omitted.
        clip_val: Winsorization threshold for cross-sectional Z-scores.
        warmup_days: Number of initial days to discard to remove NaN warmup period.
        feature_names: Ordered list of feature names.

    Execution timing:
        Features dated ``t`` use the close at ``t`` and must only be used for
        orders executed from ``t + 1`` onward. Do not use them to trade at the
        same close.

    Returns:
        Tuple containing:
            - features_tensor: np.ndarray of shape (T - warmup_days, N, F)
            - dates: DatetimeIndex of length T - warmup_days
            - tickers: Asset ticker Index of length N
    """
    if not isinstance(warmup_days, int) or isinstance(warmup_days, bool):
        raise ValueError("warmup_days must be an integer.")
    if warmup_days < _DEFAULT_WARMUP_DAYS:
        raise ValueError(
            f"warmup_days must be at least {_DEFAULT_WARMUP_DAYS}, the longest feature lookback."
        )
    if warmup_days >= len(prices):
        raise ValueError("warmup_days must leave at least one feature observation.")

    raw = compute_raw_features(prices, returns=returns)
    norm = normalize_features(raw, clip_val=clip_val)

    trimmed = {k: v.iloc[warmup_days:] for k, v in norm.items()}

    dates = prices.index[warmup_days:]
    tickers = prices.columns
    tensor = features_to_tensor(trimmed, feature_names=feature_names)

    return tensor, dates, tickers


def save_features(
    features_tensor: np.ndarray,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    feature_names: Sequence[str] = FEATURE_NAMES,
    path: Union[str, Path] = "data/processed/features_tensor.pt",
) -> None:
    """Save processed feature tensor and metadata to disk.

    Supports .pt (PyTorch dictionary) and .npz (compressed NumPy archive).

    Args:
        features_tensor: 3D array of shape (T, N, F).
        dates: DatetimeIndex of length T.
        tickers: Index or list of asset tickers of length N.
        feature_names: List of feature names of length F.
        path: Destination path (.pt or .npz).

    Raises:
        ValueError: If the tensor, metadata, or output format is invalid.
    """
    path = Path(path)
    if path.suffix not in {".pt", ".npz"}:
        raise ValueError("Unsupported file format. Expected .pt or .npz")
    if not isinstance(features_tensor, np.ndarray) or features_tensor.ndim != 3:
        raise ValueError("features_tensor must be a three-dimensional numpy array.")
    if not np.isfinite(features_tensor).all():
        raise ValueError("features_tensor must not contain missing or non-finite values.")
    path.parent.mkdir(parents=True, exist_ok=True)

    t, n, f = features_tensor.shape
    if len(dates) != t or len(tickers) != n or len(feature_names) != f:
        raise ValueError(
            f"Shape mismatch: tensor ({t}, {n}, {f}) vs "
            f"metadata lengths (dates={len(dates)}, tickers={len(tickers)}, features={len(feature_names)})"
        )

    payload = {
        "features": features_tensor,
        "dates": pd.DatetimeIndex(dates),
        "tickers": np.array(tickers, dtype=str),
        "feature_names": np.array(feature_names, dtype=str),
    }

    if path.suffix == ".pt":
        try:
            import torch

            payload["features"] = torch.from_numpy(features_tensor).float()
            payload["dates"] = [timestamp.isoformat() for timestamp in dates]
            payload["tickers"] = payload["tickers"].tolist()
            payload["feature_names"] = payload["feature_names"].tolist()
            torch.save(payload, path)
            logger.info("Saved feature tensor to %s with torch", path)
            return
        except ImportError:
            raise RuntimeError("Saving .pt files requires PyTorch to be installed.") from None

    np.savez_compressed(
        path,
        features=payload["features"],
        dates=np.asarray([timestamp.isoformat() for timestamp in dates], dtype=str),
        tickers=payload["tickers"],
        feature_names=payload["feature_names"],
    )
    logger.info("Saved features to %s with numpy", path)


def load_features(
    path: Union[str, Path] = "data/processed/features_tensor.pt",
) -> dict[str, Any]:
    """Load feature tensor and metadata from disk.

    Args:
        path: Path to the saved .pt or .npz file.

    Returns:
        Dictionary containing:
            - 'features': Tensor or array of shape (T, N, F)
            - 'dates': pd.DatetimeIndex of length T
            - 'tickers': list of asset ticker strings of length N
            - 'feature_names': list of feature name strings of length F

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file extension is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found at: {path}")

    if path.suffix == ".pt":
        import torch

        data = torch.load(path, weights_only=True)
        return {
            "features": data["features"],
            "dates": pd.DatetimeIndex(data["dates"]),
            "tickers": list(data["tickers"]),
            "feature_names": list(data["feature_names"]),
        }

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {
                "features": data["features"],
                "dates": pd.DatetimeIndex(data["dates"]),
                "tickers": list(data["tickers"]),
                "feature_names": list(data["feature_names"]),
            }

    raise ValueError(f"Unsupported file format: {path.suffix}. Expected .pt or .npz")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parents[3]
    prices_path = project_root / "data" / "raw" / "prices.parquet"

    if prices_path.exists():
        prices_df = pd.read_parquet(prices_path)
        logger.info("Loaded prices data with shape: %s", prices_df.shape)

        tensor, dates_idx, tickers_idx = extract_feature_tensor(prices_df)
        logger.info(
            "Extracted feature tensor with shape: %s (T=%d, N=%d, F=%d)",
            tensor.shape,
            tensor.shape[0],
            tensor.shape[1],
            tensor.shape[2],
        )

        output_path = project_root / "data" / "processed" / "features_tensor.pt"
        save_features(
            features_tensor=tensor,
            dates=dates_idx,
            tickers=tickers_idx,
            path=output_path,
        )
        logger.info("Features pipeline complete. File saved to: %s", output_path)
    else:
        logger.warning("Prices file not found at %s. Skipping standalone run.", prices_path)