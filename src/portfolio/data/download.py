from pathlib import Path
import yaml
from typing import Any

import yfinance as yf


def load_config(
        path: Path
) -> dict[str, Any]:
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


def download_prices(
        tickers: list[str], start: str, end: str, path: Path
) -> None:
    """Download historical adjusted close prices from Yahoo Finance and save to Parquet.

    Args:
        tickers: List of ticker symbols to fetch.
        start: Start date string formatted as 'YYYY-MM-DD'.
        end: End date string formatted as 'YYYY-MM-DD'.
        path: Destination path where the Parquet file will be written.

    Raises:
        ValueError: If no data is returned or 'Adj Close' is missing.
    """
    # auto_adjust=False with ['Adj Close'] preserves exact dividend/split adjusted levels
    raw_df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False
    )

    if raw_df.empty:
        raise ValueError("Downloaded dataset is empty. Check tickers or network.")

    if "Adj Close" not in raw_df.columns:
        raise ValueError("Downloaded dataset does not contain \"Adj Close\".")

    df = raw_df["Adj Close"]

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

    print(f"Downloaded {len(df.columns)} assets from {start} to {end} -> {path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "configs" / "data.yaml"
    raw_data_path = project_root / "data" / "raw" / "prices.parquet"

    config = load_config(config_path)
    start = config["date_range"]["start"]
    end = config["date_range"]["end"]
    ticker_list = config["ticker_list"]

    download_prices(ticker_list, start, end, raw_data_path)