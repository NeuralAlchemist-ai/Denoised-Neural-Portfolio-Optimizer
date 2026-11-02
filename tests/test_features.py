import numpy as np
import pandas as pd
import pytest

from portfolio.data.features import (
    FEATURE_NAMES,
    compute_raw_features,
    extract_feature_tensor,
    features_to_tensor,
    load_features,
    normalize_features,
    save_features,
)
from portfolio.data.preprocess import compute_simple_returns


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Generate deterministic synthetic price data (300 days x 5 assets)."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]

    # Geometric random walk
    log_returns = np.random.normal(0.0005, 0.015, size=(len(dates), len(tickers)))
    price_paths = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(price_paths, index=dates, columns=tickers)


def test_compute_raw_features_shapes_and_keys(sample_prices: pd.DataFrame):
    raw = compute_raw_features(sample_prices)
    assert set(raw.keys()) == set(FEATURE_NAMES)
    for name, df in raw.items():
        assert df.shape == sample_prices.shape
        assert list(df.columns) == list(sample_prices.columns)


def test_compute_raw_features_calculates_known_momentum():
    dates = pd.bdate_range("2024-01-01", periods=260)
    prices = pd.DataFrame({"A": np.arange(100.0, 360.0)}, index=dates)

    raw = compute_raw_features(prices)

    assert raw["mom_5d"].loc[dates[5], "A"] == pytest.approx(5.0 / 100.0)
    assert raw["mom_252d"].loc[dates[252], "A"] == pytest.approx(252.0 / 100.0)


def test_normalize_features_bounds(sample_prices: pd.DataFrame):
    raw = compute_raw_features(sample_prices)
    norm = normalize_features(raw, clip_val=2.5)

    for name, df in norm.items():
        # Valid non-warmup rows should be bounded by [-2.5, 2.5]
        valid_rows = df.iloc[252:]
        assert (valid_rows >= -2.5 - 1e-6).all().all()
        assert (valid_rows <= 2.5 + 1e-6).all().all()


def test_extract_feature_tensor_dimensions(sample_prices: pd.DataFrame):
    tensor, dates, tickers = extract_feature_tensor(sample_prices, warmup_days=252)
    expected_t = len(sample_prices) - 252

    assert tensor.shape == (expected_t, 5, 10)
    assert len(dates) == expected_t
    assert list(tickers) == list(sample_prices.columns)
    assert not np.isnan(tensor).any()


def test_extract_feature_tensor_accepts_project_simple_returns(sample_prices: pd.DataFrame):
    returns = compute_simple_returns(sample_prices)

    tensor, dates, tickers = extract_feature_tensor(sample_prices, returns=returns)

    assert tensor.shape == (48, 5, len(FEATURE_NAMES))
    assert dates.equals(sample_prices.index[252:])
    assert tickers.equals(sample_prices.columns)


def test_extract_feature_tensor_rejects_missing_prices(sample_prices: pd.DataFrame):
    prices = sample_prices.copy()
    prices.loc[prices.index[100], "AAPL"] = np.nan

    with pytest.raises(ValueError, match="missing or non-finite"):
        extract_feature_tensor(prices)


@pytest.mark.parametrize("warmup_days", [251, 300, -1, 252.5])
def test_extract_feature_tensor_rejects_invalid_warmup(
    sample_prices: pd.DataFrame, warmup_days: int
):
    with pytest.raises(ValueError, match="warmup_days"):
        extract_feature_tensor(sample_prices, warmup_days=warmup_days)


def test_features_to_tensor_rejects_remaining_missing_values(sample_prices: pd.DataFrame):
    raw = compute_raw_features(sample_prices)

    with pytest.raises(ValueError, match="missing or non-finite"):
        features_to_tensor(raw)


def test_save_and_load_features(sample_prices: pd.DataFrame, tmp_path):
    tensor, dates, tickers = extract_feature_tensor(sample_prices, warmup_days=252)
    save_path = tmp_path / "features_test.npz"

    save_features(
        features_tensor=tensor,
        dates=dates,
        tickers=tickers,
        path=save_path,
    )

    loaded = load_features(save_path)
    np.testing.assert_allclose(loaded["features"], tensor, rtol=1e-5)
    assert list(loaded["tickers"]) == list(tickers)
    assert list(loaded["feature_names"]) == FEATURE_NAMES
    assert len(loaded["dates"]) == len(dates)


def test_save_and_load_pt_features(sample_prices: pd.DataFrame, tmp_path):
    tensor, dates, tickers = extract_feature_tensor(sample_prices)
    save_path = tmp_path / "features_test.pt"

    save_features(tensor, dates, tickers, path=save_path)
    loaded = load_features(save_path)

    np.testing.assert_allclose(loaded["features"].numpy(), tensor, rtol=1e-5)
    assert loaded["dates"].equals(dates)


def test_save_features_rejects_invalid_tensor_and_suffix(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=2)

    with pytest.raises(ValueError, match="three-dimensional"):
        save_features(np.ones((2, 2)), dates, pd.Index(["A", "B"]), path=tmp_path / "x.npz")
    with pytest.raises(ValueError, match="Expected .pt or .npz"):
        save_features(
            np.ones((2, 2, 1)), dates, pd.Index(["A", "B"]), path=tmp_path / "x.csv"
        )