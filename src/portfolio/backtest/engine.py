import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from portfolio.data.download import load_config
from portfolio.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


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
            self.transaction_cost_bps <= 0.0
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

        returns_arr = returns.to_numpy(dtype=np.float32)

        dates = returns.index

        current_wealth = float(self.config.initial_capital)
        current_weights = np.zeros(n_assets, dtype=np.float32)

        daily_returns: list[float] = []
        portfolio_values: list[float] = []
        transaction_costs: list[float] = []
        sim_dates: list[datetime] = []
        rebalance_dates: list[datetime] = []
        rebalance_turnover: list[datetime] = []