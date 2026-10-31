import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    """Load and parse a YAML configuration file from disk.

    Args:
        path: Filesystem path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the configuration file is empty.
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Configuration file at {path} is empty.")

    return config


def _fetch_adj_close(
    tickers: list[str],
    start: str,
    end: str,
    threads: bool = True,
) -> pd.DataFrame:
    """Fetch raw Adjusted Close price data from Yahoo Finance.

    Args:
        tickers: List of ticker symbols to fetch.
        start: Start date string formatted as 'YYYY-MM-DD'.
        end: End date string formatted as 'YYYY-MM-DD'.
        threads: Whether to download concurrently using threads. Set to False
            for sequential downloads to avoid API rate limiting.

    Returns:
        DataFrame containing Adjusted Close prices, or empty DataFrame on failure.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, threads=threads)
    if raw.empty or "Adj Close" not in raw.columns:
        return pd.DataFrame()

    adj_close = raw["Adj Close"]
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame(name=tickers[0])
    return adj_close


def _find_missing_tickers(
    df: pd.DataFrame,
    expected_tickers: list[str],
) -> list[str]:
    """Identify tickers that are missing from DataFrame columns or contain only NaNs.

    Args:
        df: Price DataFrame to validate.
        expected_tickers: Complete list of expected ticker symbols.

    Returns:
        List of tickers with missing or all-NaN price series.
    """
    return [
        ticker
        for ticker in expected_tickers
        if ticker not in df.columns or df[ticker].isna().all()
    ]


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    path: Path,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> pd.DataFrame:
    """Download historical adjusted close prices from Yahoo Finance and save to Parquet.

    Fetches price data in bulk, detects dropped or all-NaN ticker series caused
    by API rate limits, retries missing tickers with exponential backoff, and
    persists the resulting dataset to disk.

    Args:
        tickers: List of ticker symbols to fetch.
        start: Start date string formatted as 'YYYY-MM-DD'.
        end: End date string formatted as 'YYYY-MM-DD'.
        path: Destination path where the Parquet file will be written.
        max_retries: Maximum number of retry attempts for dropped tickers.
        backoff_factor: Multiplier for exponential backoff sleep delay in seconds.

    Returns:
        DataFrame of adjusted close prices with tickers as columns and dates as index.

    Raises:
        ValueError: If the initial download returns empty data or lacks 'Adj Close'.
        RuntimeError: If any ticker still has all-NaN values after max_retries.
    """
    logger.info("Fetching price data for %d tickers from %s to %s", len(tickers), start, end)
    df = _fetch_adj_close(tickers, start=start, end=end, threads=True)

    if df.empty:
        error_msg = "Initial price download returned empty dataset or lacks 'Adj Close'."
        logger.error(error_msg)
        raise ValueError(error_msg)

    missing = _find_missing_tickers(df, tickers)

    for attempt in range(1, max_retries + 1):
        if not missing:
            break

        sleep_time = backoff_factor * attempt
        logger.warning(
            "[%d/%d] Missing data for %d tickers (%s). Retrying in %.1fs...",
            attempt,
            max_retries,
            len(missing),
            missing,
            sleep_time,
        )
        time.sleep(sleep_time)

        retry_df = _fetch_adj_close(missing, start=start, end=end, threads=False)
        for ticker in list(missing):
            if ticker in retry_df.columns and not retry_df[ticker].isna().all():
                df[ticker] = retry_df[ticker]
                missing.remove(ticker)

    if missing:
        error_msg = (
            f"Failed to fetch valid price series for {len(missing)} tickers "
            f"after {max_retries} retries: {missing}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    df = df[tickers]

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Saved %d assets (%d trading days) -> %s", len(df.columns), len(df), path)
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "configs" / "data.yaml"
    raw_data_path = project_root / "data" / "raw" / "prices.parquet"

    config = load_config(config_path)
    start_date = config["date_range"]["start"]
    end_date = config["date_range"]["end"]
    ticker_list = config["ticker_list"]

    download_prices(ticker_list, start_date, end_date, raw_data_path)