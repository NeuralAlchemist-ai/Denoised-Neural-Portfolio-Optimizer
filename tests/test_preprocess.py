import numpy as np
import pandas as pd
import pytest

from portfolio.data.preprocess import compute_log_returns, compute_simple_returns


@pytest.fixture
def prices() -> pd.DataFrame:
    return pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [200.0, 190.0, 209.0]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )


def test_compute_simple_returns_uses_price_changes(prices: pd.DataFrame):
    returns = compute_simple_returns(prices)

    expected = pd.DataFrame(
        {"A": [0.1, 0.1], "B": [-0.05, 0.1]}, index=prices.index[1:]
    )
    pd.testing.assert_frame_equal(returns, expected)


def test_compute_log_returns_uses_log_price_differences(prices: pd.DataFrame):
    returns = compute_log_returns(prices)

    np.testing.assert_allclose(returns["A"], [np.log(1.1), np.log(1.1)])
    np.testing.assert_allclose(returns["B"], [np.log(0.95), np.log(1.1)])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, 0.0, -1.0])
def test_return_calculations_reject_invalid_prices(prices: pd.DataFrame, bad_value: float):
    invalid = prices.copy()
    invalid.iloc[1, 0] = bad_value

    with pytest.raises(ValueError, match="missing|non-finite|strictly positive"):
        compute_simple_returns(invalid)
    with pytest.raises(ValueError, match="missing|non-finite|strictly positive"):
        compute_log_returns(invalid)


def test_return_calculations_require_two_observations():
    prices = pd.DataFrame({"A": [100.0]}, index=pd.bdate_range("2024-01-01", periods=1))

    with pytest.raises(ValueError, match="at least two"):
        compute_simple_returns(prices)