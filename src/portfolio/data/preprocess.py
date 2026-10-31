import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple arithmetic returns from price series.

    Formula: r_t = (P_t - P_{t-1}) / P_{t-1}

    Args:
        prices: DataFrame of asset prices with dates as index and tickers as columns.

    Returns:
        DataFrame of simple returns with the initial NaN row removed.

    Raises:
        ValueError: If the input prices DataFrame is empty.
    """
    if prices.empty:
        error_msg = "Input prices DataFrame is empty. Cannot compute simple returns."
        logger.error(error_msg)
        raise ValueError(error_msg)

    return prices.pct_change().dropna()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily logarithmic (continuously compounded) returns from price series.

    Formula: r_t = ln(P_t / P_{t-1})

    Args:
        prices: DataFrame of asset prices with dates as index and tickers as columns.

    Returns:
        DataFrame of log returns with the initial NaN row removed.

    Raises:
        ValueError: If the input prices DataFrame is empty.
    """
    if prices.empty:
        error_msg = "Input prices DataFrame is empty. Cannot compute log returns."
        logger.error(error_msg)
        raise ValueError(error_msg)

    return np.log(prices / prices.shift(1)).dropna()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    project_root = Path(__file__).resolve().parents[3]
    raw_data_path = project_root / "data" / "raw" / "prices.parquet"
    processed_data_path = project_root / "data" / "processed" / "returns.parquet"

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