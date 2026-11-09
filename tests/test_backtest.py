import pytest

import numpy as np
import pandas as pd

from portfolio.strategies.classical import EqualWeightStrategy
from portfolio.backtest.engine import BacktestConfig, BacktestResult, BacktestEngine


@pytest.fixture
def mock_strategy() -> EqualWeightStrategy:
    return EqualWeightStrategy()

@pytest.fixture
def sample_config() -> BacktestConfig:
    return BacktestConfig(
        lookback_days=5,
        rebalance_freq=2,
        transaction_cost_bps=10.0,
        initial_capital=100.0,
    )

@pytest.fixture
def constant_zero_returns() -> pd.DataFrame:
    """10 business days for 2 assets where returns are all 0.0."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    return pd.DataFrame(0.0, index=dates, columns=["A", "B"])

def test_engine_rejects_insufficient_data(mock_strategy, sample_config):
    # Only 5 days, but lookback is 5, so min required is 5 + 1 = 6
    short_returns = pd.DataFrame(
        0.0, index=pd.bdate_range("2024-01-01", periods=5), columns=["A", "B"]
    )
    engine = BacktestEngine(sample_config)
    with pytest.raises(ValueError, match="Insufficient data"):
        engine.run(mock_strategy, short_returns)

@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -1.05])
def test_engine_rejects_corrupted_returns(mock_strategy, sample_config, bad_val):
    dates = pd.bdate_range("2024-01-01", periods=10)
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    returns.iloc[2, 0] = bad_val
    engine = BacktestEngine(sample_config)
    with pytest.raises(ValueError):
        engine.run(mock_strategy, returns)

def test_engine_flat_returns_preserves_capital(mock_strategy, constant_zero_returns):
    config = BacktestConfig(
        lookback_days=5,
        rebalance_freq=2,
        transaction_cost_bps=0.0,  # no cost
        initial_capital=100.0,
    )
    engine = BacktestEngine(config)
    result = engine.run(mock_strategy, constant_zero_returns)

    # 10 days total, lookback=5 => simulated days = 5 (indices 5, 6, 7, 8, 9)
    assert len(result.returns) == 5
    assert len(result.portfolio_values) == 5
    assert np.allclose(result.returns.to_numpy(), 0.0)
    assert np.allclose(result.portfolio_values.to_numpy(), 100.0)

def test_engine_drift_and_turnover_exact_math(mock_strategy):
    dates = pd.bdate_range("2024-01-01", periods=7)
    # Day 0-4: 0.0. Day 5: [0.10, -0.10]. Day 6: [0.0, 0.0]
    data = np.zeros((7, 2))
    data[5] = [0.10, -0.10]
    returns = pd.DataFrame(data, index=dates, columns=["A", "B"])

    config = BacktestConfig(
        lookback_days=5,
        rebalance_freq=1,
        transaction_cost_bps=10.0,  # 0.0010
        initial_capital=1.0,
    )
    engine = BacktestEngine(config)
    result = engine.run(mock_strategy, returns)

    # Check Day 5 turnover & cost
    assert np.isclose(result.turnover.iloc[0], 1.0)
    assert np.isclose(result.transaction_costs.iloc[0], 0.001)

    # Check Day 6 turnover (should be exactly 0.10 due to drift)
    assert np.isclose(result.turnover.iloc[1], 0.10)
    assert np.isclose(result.transaction_costs.iloc[1], 0.0001)