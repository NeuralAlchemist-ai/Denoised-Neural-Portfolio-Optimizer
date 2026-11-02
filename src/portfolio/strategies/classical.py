from collections.abc import Callable
import logging
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from portfolio.covariance.denoise import denoise_covariance
from portfolio.covariance.estimators import (
    ewma_covariance,
    ledoit_wolf_covariance,
    sample_covariance,
)
from portfolio.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Registry of supported covariance estimator functions
_ESTIMATOR_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "sample": sample_covariance,
    "ledoit_wolf": ledoit_wolf_covariance,
    "ewma": ewma_covariance,
}


class EqualWeightStrategy(BaseStrategy):
    """Equal Weight (1/N) benchmark strategy.

    Allocates an identical fraction 1/N of capital across all assets.
    Does not require parameter estimation or optimization, serving as
    the standard de-facto baseline in portfolio literature.
    """

    def compute_weights(self, returns_window: pd.DataFrame) -> np.ndarray:
        """Calculate equal weights across all assets.

        Args:
            returns_window: DataFrame of shape (T, N) containing historical
                returns with assets as columns.

        Returns:
            1-D numpy array of shape (N,) containing 1/N weights.

        Raises:
            ValueError: If returns_window has no columns (no assets).
        """
        n_assets = returns_window.shape[1]

        if n_assets == 0:
            error_msg = "Cannot compute weights for an empty asset universe."
            logger.error(error_msg)
            raise ValueError(error_msg)

        weights = np.full(shape=n_assets, fill_value=1.0 / n_assets, dtype=np.float64)
        logger.debug("EqualWeightStrategy allocated 1/%d across %d assets", n_assets, n_assets)
        return weights


class MinimumVarianceStrategy(BaseStrategy):
    """Global Minimum Variance (GMV) portfolio strategy.

    Finds the long-only portfolio weights that minimize portfolio risk:
        min_w  w^T Sigma w
        s.t.   sum(w) = 1,  w >= 0

    Can be combined with any supported covariance estimator and optionally
    enhanced with Marchenko-Pastur spectral denoising.
    """

    def __init__(
        self,
        estimator: str | Callable[..., np.ndarray] = "sample",
        denoise: bool = False,
        **estimator_kwargs: Any,
    ) -> None:
        """Initialize the Minimum Variance Strategy.

        Args:
            estimator: Covariance estimator to use. Either a string identifier
                ('sample', 'ledoit_wolf', 'ewma') or a callable taking a
                DataFrame of returns and returning an (N, N) covariance matrix.
            denoise: If True, applies Marchenko-Pastur spectral denoising
                to the estimated covariance matrix before optimization.
            **estimator_kwargs: Additional keyword arguments forwarded to the
                estimator function (e.g., half_life for EWMA).

        Raises:
            ValueError: If estimator is a string but not in the supported registry.
        """
        if isinstance(estimator, str):
            if estimator not in _ESTIMATOR_REGISTRY:
                error_msg = (
                    f"Unknown estimator '{estimator}'. "
                    f"Supported estimators: {list(_ESTIMATOR_REGISTRY.keys())}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            self._estimator_fn = _ESTIMATOR_REGISTRY[estimator]
            self.estimator_name = estimator
        elif callable(estimator):
            self._estimator_fn = estimator
            self.estimator_name = getattr(estimator, "__name__", "custom_estimator")
        else:
            error_msg = "estimator must be a string or a callable."
            logger.error(error_msg)
            raise TypeError(error_msg)

        self.denoise = denoise
        self.estimator_kwargs = estimator_kwargs


    def __repr__(self) -> str:
        """Return a descriptive string representation."""
        denoise_suffix = "+denoise" if self.denoise else ""
        return f"MinimumVariance({self.estimator_name}{denoise_suffix})"


    def compute_weights(self, returns_window: pd.DataFrame) -> np.ndarray:
        """Optimize portfolio weights to minimize variance over the historical window.

        Args:
            returns_window: DataFrame of shape (T, N) containing historical returns.

        Returns:
            1-D numpy array of shape (N,) containing optimal weights summing to 1.0.

        Raises:
            ValueError: If returns_window is empty or optimization fails.
        """
        if returns_window.empty or returns_window.shape[1] == 0:
            error_msg = "returns_window must not be empty."
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_obs, n_assets = returns_window.shape

        # 1. Estimate Covariance
        cov_matrix = self._estimator_fn(returns_window, **self.estimator_kwargs)

        # 2. Optionally Denoise
        if self.denoise:
            cov_matrix = denoise_covariance(cov_matrix, n_obs=n_obs)

        # 3. Ensure numerical symmetry
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)

        # 4. Formulate Convex Optimization problem via CVXPY
        w = cp.Variable(n_assets)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov_matrix)))
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
        ]
        problem = cp.Problem(objective, constraints)

        # 5. Solve Quadratic Program
        try:
            problem.solve()
        except Exception as exc:
            error_msg = f"CVXPY solver failed during optimization: {exc}"
            logger.error(error_msg)
            raise ValueError(error_msg) from exc

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
            error_msg = f"Optimization failed to converge. Status: {problem.status}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        raw_weights = np.array(w.value).flatten()

        # Clean tiny numerical negative values resulting from solver tolerance
        raw_weights = np.maximum(raw_weights, 0.0)

        # 6. Normalise weights to sum to 1.0 using base class helper
        weights = self._normalise_weights(raw_weights)

        logger.debug(
            "%s: weights computed successfully for %d assets",
            self.__repr__(),
            n_assets,
        )
        return weights


class MaxSharpeStrategy(BaseStrategy):
    """Maximum Sharpe Ratio (Tangency) portfolio strategy.

    Finds the long-only portfolio weights that maximize the Sharpe ratio:
        max_w  (mu^T w - r_f) / sqrt(w^T Sigma w)
        s.t.   sum(w) = 1,  w >= 0

    Solved as an equivalent strictly convex Quadratic Program using the
    Stutzer / Korn transformation:
        min_y  y^T Sigma y
        s.t.   (mu - r_f * 1)^T y = 1,  y >= 0
    with weights recovered via w = y / sum(y).
    """

    def __init__(
        self,
        estimator: str | Callable[..., np.ndarray] = "sample",
        denoise: bool = False,
        risk_free_rate: float = 0.0,
        **estimator_kwargs: Any,
    ) -> None:
        """Initialize the Maximum Sharpe Strategy.

        Args:
            estimator: Covariance estimator to use ('sample', 'ledoit_wolf', 'ewma')
                or a callable.
            denoise: If True, applies Marchenko-Pastur spectral denoising.
            risk_free_rate: Daily risk-free return rate (default 0.0).
            **estimator_kwargs: Additional keyword arguments for the estimator.

        Raises:
            ValueError: If estimator is an unsupported string name.
            TypeError: If estimator is neither string nor callable.
        """
        if isinstance(estimator, str):
            if estimator not in _ESTIMATOR_REGISTRY:
                error_msg = (
                    f"Unknown estimator '{estimator}'. "
                    f"Supported estimators: {list(_ESTIMATOR_REGISTRY.keys())}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            self._estimator_fn = _ESTIMATOR_REGISTRY[estimator]
            self.estimator_name = estimator
        elif callable(estimator):
            self._estimator_fn = estimator
            self.estimator_name = getattr(estimator, "__name__", "custom_estimator")
        else:
            error_msg = "estimator must be a string or a callable."
            logger.error(error_msg)
            raise TypeError(error_msg)

        self.denoise = denoise
        self.risk_free_rate = risk_free_rate
        self.estimator_kwargs = estimator_kwargs


    def __repr__(self) -> str:
        """Return a descriptive string representation."""
        denoise_suffix = "+denoise" if self.denoise else ""
        return f"MaxSharpe({self.estimator_name}{denoise_suffix})"


    def compute_weights(self, returns_window: pd.DataFrame) -> np.ndarray:
        """Optimize portfolio weights to maximize the Sharpe ratio over historical window.

        Args:
            returns_window: DataFrame of shape (T, N) containing historical returns.

        Returns:
            1-D numpy array of shape (N,) containing optimal weights summing to 1.0.

        Raises:
            ValueError: If returns_window is empty or optimization fails.
        """
        if returns_window.empty or returns_window.shape[1] == 0:
            error_msg = "returns_window must not be empty."
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_obs, n_assets = returns_window.shape

        # 1. Estimate Expected Returns (historical arithmetic mean)
        mu = returns_window.mean().to_numpy()
        excess_mu = mu - self.risk_free_rate

        # Guard: check if at least one asset has positive excess return
        if np.all(excess_mu <= 0.0):
            logger.warning(
                "%s: all asset excess returns <= 0. "
                "Max Sharpe formulation unfeasible; falling back to equal weights.",
                self.__repr__(),
            )
            return np.full(shape=n_assets, fill_value=1.0 / n_assets, dtype=np.float64)

        # 2. Estimate Covariance
        cov_matrix = self._estimator_fn(returns_window, **self.estimator_kwargs)

        # 3. Optionally Denoise
        if self.denoise:
            cov_matrix = denoise_covariance(cov_matrix, n_obs=n_obs)

        # 4. Symmetrize
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)

        # 5. Formulate QP via Korn / Stutzer transformation
        y = cp.Variable(n_assets)
        objective = cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov_matrix)))
        constraints = [
            excess_mu @ y == 1.0,
            y >= 0.0,
        ]
        problem = cp.Problem(objective, constraints)

        # 6. Solve Quadratic Program
        try:
            problem.solve()
        except Exception as exc:
            error_msg = f"CVXPY solver failed during Max Sharpe optimization: {exc}"
            logger.error(error_msg)
            raise ValueError(error_msg) from exc

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or y.value is None:
            logger.warning(
                "%s: solver status '%s'. Falling back to equal weights.",
                self.__repr__(),
                problem.status,
            )
            return np.full(shape=n_assets, fill_value=1.0 / n_assets, dtype=np.float64)

        raw_weights = np.array(y.value).flatten()
        raw_weights = np.maximum(raw_weights, 0.0)

        # 7. Normalise: w = y / sum(y)
        weights = self._normalise_weights(raw_weights)

        logger.debug(
            "%s: weights computed successfully for %d assets",
            self.__repr__(),
            n_assets,
        )
        return weights