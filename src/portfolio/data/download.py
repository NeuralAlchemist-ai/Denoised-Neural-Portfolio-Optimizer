import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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


def _validate_download_inputs(
    tickers: list[str],
    start: str,
    end: str,
    path: Path,
    max_retries: int,
    backoff_factor: float,
) -> None:
    """Validate download inputs before contacting the data provider."""
    if (
        not isinstance(tickers, list)
        or not tickers
        or not all(isinstance(ticker, str) and ticker.strip() for ticker in tickers)
    ):
        raise ValueError("tickers must be a non-empty list of ticker symbols.")
    if len(set(tickers)) != len(tickers):
        raise ValueError("tickers must not contain duplicates.")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("start must be earlier than end.")
    if path.suffix != ".parquet":
        raise ValueError("Price data path must use the .parquet extension.")
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer.")
    if not np.isfinite(backoff_factor) or backoff_factor < 0.0:
        raise ValueError("backoff_factor must be a non-negative finite number.")


def validate_data_config(config: dict[str, Any]) -> None:
    """Validate the download-relevant portion of the project configuration."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping.")
    date_range = config.get("date_range")
    tickers = config.get("ticker_list")
    asset_count = config.get("number_of_assets")
    if not isinstance(date_range, dict) or "start" not in date_range or "end" not in date_range:
        raise ValueError("Configuration must define date_range.start and date_range.end.")
    if not isinstance(asset_count, int) or isinstance(asset_count, bool) or asset_count < 1:
        raise ValueError("number_of_assets must be a positive integer.")

    _validate_download_inputs(
        tickers,
        date_range["start"],
        date_range["end"],
        Path("prices.parquet"),
        max_retries=0,
        backoff_factor=0.0,
    )
    if len(tickers) != asset_count:
        raise ValueError("number_of_assets must match the number of configured tickers.")


def _invalid_tickers(df: pd.DataFrame, expected_tickers: list[str]) -> list[str]:
    """Return tickers without a complete, finite, positive price history."""
    invalid: list[str] = []
    for ticker in expected_tickers:
        if ticker not in df.columns:
            invalid.append(ticker)
            continue
        try:
            values = df[ticker].to_numpy(dtype=float)
        except (TypeError, ValueError):
            invalid.append(ticker)
            continue
        if not np.isfinite(values).all() or (values <= 0.0).any():
            invalid.append(ticker)
    return invalid


def _validate_price_panel(df: pd.DataFrame, expected_tickers: list[str]) -> None:
    """Ensure persisted prices satisfy the shared downstream data contract."""
    if df.empty:
        raise RuntimeError("Downloaded price panel is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("Downloaded prices must use a DatetimeIndex.")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise RuntimeError("Downloaded price index must be sorted and unique.")
    if list(df.columns) != expected_tickers:
        raise RuntimeError("Downloaded price columns do not match the requested ticker order.")

    invalid = _invalid_tickers(df, expected_tickers)
    if invalid:
        raise RuntimeError(
            "Downloaded prices contain missing, non-finite, or non-positive values for: "
            f"{invalid}"
        )


def _write_metadata(path: Path, tickers: list[str], start: str, end: str) -> None:
    """Persist the provenance required to reproduce a downloaded price panel."""
    metadata_path = path.with_suffix(".metadata.yaml")
    metadata = {
        "source": "Yahoo Finance",
        "field": "Adj Close",
        "auto_adjust": False,
        "requested_start": start,
        "requested_end": end,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        yaml.safe_dump(metadata, metadata_file, sort_keys=False)


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
    """Identify tickers without a complete valid price history.

    Args:
        df: Price DataFrame to validate.
        expected_tickers: Complete list of expected ticker symbols.

    Returns:
        List of tickers with absent, missing, non-finite, or non-positive prices.
    """
    return _invalid_tickers(df, expected_tickers)


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    path: Path,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> pd.DataFrame:
    """Download historical adjusted close prices from Yahoo Finance and save to Parquet.

    Fetches price data in bulk, retries incomplete ticker histories with
    backoff, validates the complete price panel, and persists both prices and
    provenance metadata to disk.

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
        RuntimeError: If any ticker has invalid prices after max_retries.
    """
    path = Path(path)
    _validate_download_inputs(tickers, start, end, path, max_retries, backoff_factor)
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
            if ticker in retry_df.columns and ticker not in _find_missing_tickers(retry_df, [ticker]):
                df[ticker] = retry_df[ticker]
                missing.remove(ticker)

    if missing:
        error_msg = (
            f"Failed to fetch complete valid price series for {len(missing)} tickers "
            f"after {max_retries} retries: {missing}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    df = df.loc[:, tickers]
    _validate_price_panel(df, tickers)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.parquet")
    df.to_parquet(temporary_path)
    temporary_path.replace(path)
    _write_metadata(path, tickers, start, end)
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
    validate_data_config(config)
    start_date = config["date_range"]["start"]
    end_date = config["date_range"]["end"]
    ticker_list = config["ticker_list"]

    download_prices(ticker_list, start_date, end_date, raw_data_path)