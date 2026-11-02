import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _validate_prices(prices: pd.DataFrame) -> None:
    """Validate the complete price panel required for return calculations."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Prices must be a non-empty pandas DataFrame.")
    if len(prices) < 2:
        raise ValueError("Prices must contain at least two observations.")
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


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple arithmetic returns from price series.

    Formula: r_t = (P_t - P_{t-1}) / P_{t-1}

    Args:
        prices: DataFrame of asset prices with dates as index and tickers as columns.

    Returns:
        DataFrame of simple returns with the initial NaN row removed.

    Raises:
        ValueError: If prices violate the complete-panel data contract.
    """
    _validate_prices(prices)

    return prices.pct_change(fill_method=None).iloc[1:]


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily logarithmic (continuously compounded) returns from price series.

    Formula: r_t = ln(P_t / P_{t-1})

    Args:
        prices: DataFrame of asset prices with dates as index and tickers as columns.

    Returns:
        DataFrame of log returns with the initial NaN row removed.

    Raises:
        ValueError: If prices violate the complete-panel data contract.
    """
    _validate_prices(prices)

    return np.log(prices).diff().iloc[1:]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    project_root = Path(__file__).resolve().parents[3]
    raw_data_path = project_root / "data" / "raw" / "prices.parquet"
    processed_data_path = project_root / "data" / "processed" / "log_returns.parquet"

    logger.info("Loading raw prices from %s", raw_data_path)
    prices_df = pd.read_parquet(raw_data_path)

    logger.info("Computing logarithmic returns...")
    returns = compute_log_returns(prices_df)

    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(processed_data_path)

    logger.info(
        "Saved returns of shape %s (%d assets, %d trading days) -> %s",
        returns.shape,
        len(returns.columns),
        len(returns),
        processed_data_path,
    )