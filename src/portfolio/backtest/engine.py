import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from portfolio.data.download import load_config, validate_data_config, download_prices
from portfolio.data.preprocess import compute_simple_returns
from portfolio.strategies.base import BaseStrategy
from portfolio.strategies import (EqualWeightStrategy,
                                  MinimumVarianceStrategy,
                                  MaxSharpeStrategy)

logger = logging.getLogger(__name__)


_ABSOLUTE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class BacktestConfig:
    lookback_days: int = 252
    rebalance_freq: int = 21
    transaction_cost_bps: float = 10.0
    initial_capital: float = 1.0
    risk_free_rate: float = 0.02


    def __post_init__(self):
        self._validate()


    def _validate(self) -> None:
        if (
            isinstance(self.lookback_days, bool) or
            not isinstance(self.lookback_days, int) or
            self.lookback_days <= 0
        ):
            raise ValueError("lookback_days must be positive integer.")

        if (
            isinstance(self.rebalance_freq, bool) or
            not isinstance(self.rebalance_freq, int) or
            self.rebalance_freq <= 0
        ):
            raise ValueError("rebalance_freq must be positive integer.")

        if (
            not np.isfinite(self.transaction_cost_bps) or
            self.transaction_cost_bps < 0.0
        ):
            raise ValueError("transaction_cost_bps must be positive finite number.")

        if (
            not np.isfinite(self.initial_capital) or
            self.initial_capital <= 0.0
        ):
            raise ValueError("initial_capital must be positive finite number.")

        if not np.isfinite(self.risk_free_rate):
            raise ValueError("initial_capital must be positive finite number.")


    @classmethod
    def from_yaml(cls, path: str | Path) -> "BacktestConfig":
        raw = load_config(path)
        return cls(
            lookback_days=raw.get("lookback_days", 252),
            rebalance_freq=raw.get("rebalance_freq", 21),
            transaction_cost_bps=float(raw.get("transaction_cost_bps", 10.0)),
            initial_capital=float(raw.get("initial_capital", 1.0)),
            risk_free_rate=float(raw.get("risk_free_rate", 0.02)),
        )


@dataclass
class BacktestResult:
    strategy_name: str
    returns: pd.Series            # Daily net portfolio returns (index: Timestamp)
    portfolio_values: pd.Series   # Wealth trajectory starting at initial_capital
    weights: pd.DataFrame         # Target weights at each rebalance (index: rebalance dates, cols: tickers)
    turnover: pd.Series           # Turnover at each rebalance date (index: rebalance dates)
    transaction_costs: pd.Series  # Daily transaction costs paid (0.0 on non-rebalance days)


def _validate_returns(returns: pd.DataFrame, min_period_days: int) -> None:
    """Validate returns panel.
       Args:
           returns: DataFrame of simple asset returns.
       Raises:
           ValueError: If returns DataFrame violates the data contract.
       """
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("Returns must be a non-empty pandas DataFrame.")

    if len(returns.columns) < 2:
        raise ValueError("Returns panel must contain at least two assets.")

    if not returns.columns.is_unique:
        raise ValueError("Returns columns must be unique.")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns must use a DatetimeIndex.")

    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise ValueError("Returns index must be sorted and contain unique timestamps.")

    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Returns must not contain NaN or infinite values.")

    if (values <= -1.0).any():
        raise ValueError("Simple returns cannot be less than or equal to -1.0 (-100%).")

    if len(returns) < min_period_days:
        raise ValueError(
            f"Insufficient data: returns has {len(returns)} rows, but lookback_days "
            f"requires at least {min_period_days} rows."
        )


class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(self, strategy: BaseStrategy, returns: pd.DataFrame) -> BacktestResult:
        _validate_returns(returns, self.config.lookback_days + 1)

        n_samples, n_assets = returns.shape
        lookback_days = self.config.lookback_days
        cost_rate = self.config.transaction_cost_bps / 10_000.0

        returns_arr = returns.to_numpy(dtype=np.float64)

        dates = returns.index

        current_wealth: float = float(self.config.initial_capital)
        current_weights: np.ndarray = np.zeros(n_assets, dtype=np.float64)

        daily_returns: list[float] = []
        portfolio_values: list[float] = []
        transaction_costs: list[float] = []
        sim_dates: list[datetime] = []
        rebalance_dates: list[datetime] = []
        rebalance_weights: list[np.ndarray] = []
        rebalance_turnover: list[float] = []

        for t in range(lookback_days, n_samples):
            if (t - lookback_days) % self.config.rebalance_freq == 0:
                window = returns.iloc[t - lookback_days: t]

                target_weights = strategy.compute_weights(window)
                turnover = float(np.sum(np.abs(target_weights - current_weights)))
                cost = turnover * cost_rate

                rebalance_dates.append(dates[t])
                rebalance_weights.append(target_weights)
                rebalance_turnover.append(turnover)

                current_weights = target_weights

            else:
                cost = 0.0

            r_t = returns_arr[t]
            r_gross = np.dot(current_weights, r_t)
            r_net = r_gross - cost
            current_wealth = current_wealth * (1 + r_net)

            daily_returns.append(r_net)
            portfolio_values.append(current_wealth)
            transaction_costs.append(cost)
            sim_dates.append(dates[t])

            if np.isclose(1 + r_gross, 0, atol=_ABSOLUTE_TOLERANCE):
                r_gross += _ABSOLUTE_TOLERANCE * 2
            current_weights = (current_weights * (1 + r_t)) / (1 + r_gross)

        returns_series = pd.Series(daily_returns, index=sim_dates, name="returns")
        values_series = pd.Series(portfolio_values, index=sim_dates, name="portfolio_value")
        costs_series = pd.Series(transaction_costs, index=sim_dates, name="transaction_cost")

        turnover_series = pd.Series(rebalance_turnover, index=rebalance_dates, name="turnover")
        weights_df = pd.DataFrame(rebalance_weights, index=rebalance_dates, columns=returns.columns)

        return BacktestResult(
            strategy_name=repr(strategy),
            returns=returns_series,
            portfolio_values=values_series,
            weights=weights_df,
            turnover=turnover_series,
            transaction_costs=costs_series,
        )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    data_config_path = project_root / "configs" / "data.yaml"
    backtest_config_path = project_root / "configs" / "backtest.yaml"
    prices_path = project_root / "data" / "raw" / "prices.parquet"
    returns_path = project_root / "data" / "processed" / "returns.parquet"

    data_config = load_config(data_config_path)
    validate_data_config(data_config)
    start_date = data_config["date_range"]["start"]
    end_date = data_config["date_range"]["end"]
    ticker_list = data_config["ticker_list"]

    prices = download_prices(ticker_list, start_date, end_date, prices_path)
    returns = compute_simple_returns(prices)

    backtest_config = BacktestConfig.from_yaml(backtest_config_path)
    engine = BacktestEngine(backtest_config)

    strategy = MaxSharpeStrategy()
    result = engine.run(strategy, returns)

    total_return = (result.portfolio_values.iloc[-1] / result.portfolio_values.iloc[0] - 1) * 100
    annualized_vol = result.returns.std() * (252 ** 0.5) * 100
    avg_turnover = result.turnover.iloc[1:].mean() * 100
    print(f"=== {result.strategy_name} Backtest Summary ===")
    print(f"Simulation Period:    {result.returns.index[0].date()} -> {result.returns.index[-1].date()}")
    print(f"Initial Capital:      {backtest_config.initial_capital:.2f}")
    print(f"Final Capital:        {result.portfolio_values.iloc[-1]:.2f}")
    print(f"Total Cumulative Ret: {total_return:.2f}%")
    print(f"Annualized Vol:       {annualized_vol:.2f}%")
    print(f"Average Turnover:     {avg_turnover:.2f}% per rebalance")
    print(f"Total Rebalances:     {len(result.weights)}")