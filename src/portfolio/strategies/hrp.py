from collections.abc import Callable
import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy
import scipy.cluster.hierarchy as sch

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

# List of supported linkage methods for hierarchical clustering
_LINKAGE_METHODS_LIST: list[str] = [
    "single",
    "complete",
    "average",
    "ward"
]


class HRPStrategy(BaseStrategy):
    """Hierarchical Risk Parity (HRP) portfolio strategy.

        Uses machine learning graph theory (hierarchical tree clustering) and
        recursive bisection to distribute risk equally across uncorrelated clusters
        without matrix inversion. Compatible with the framework's custom covariance
        estimators and Marchenko-Pastur spectral filters.
        """
    def __init__(
        self,
        estimator: str | Callable[..., np.ndarray] = "sample",
        denoise: bool = False,
        linkage_method: str = "single",
        **estimator_kwargs: Any
    ) -> None:
        """Initialize the Hierarchical Risk Parity Strategy.

        Args:
            estimator: Covariance estimator to use ('sample', 'ledoit_wolf', 'ewma')
                or a callable.
            denoise: If True, applies Marchenko-Pastur spectral denoising.
            linkage_method: Linkage algorithm for clustering ('single', 'complete', 'average', 'ward').
            **estimator_kwargs: Additional keyword arguments for the estimator.

        Raises:
            ValueError: If estimator is an unsupported string name.
            TypeError: If estimator is neither string nor callable.
            ValueError: If linkage method is an unsupported string name.
            TypeError: If linkage method is not a string.
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

        if isinstance(linkage_method, str):
            if linkage_method not in _LINKAGE_METHODS_LIST:
                error_msg = (
                    f"Unknown linkage method '{linkage_method}'. "
                    f"Supported linkage methods: {_LINKAGE_METHODS_LIST}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            self.linkage_method = linkage_method
        else:
            error_msg = "linkage method must be a string."
            logger.error(error_msg)
            raise TypeError(error_msg)

        self.estimator_kwargs = estimator_kwargs


    def __repr__(self) -> str:
        """Return a descriptive string representation."""
        denoise_suffix = "+denoise" if self.denoise else ""
        return f"HRP({self.estimator_name}{denoise_suffix}, linkage={self.linkage_method})"


    def compute_weights(self, returns_window: pd.DataFrame) -> np.ndarray:
        """Optimize portfolio weights using HRP over a historical returns window.

        Args:
            returns_window: DataFrame of shape (T, N) containing historical returns.

        Returns:
            1-D numpy array of shape (N,) containing optimal weights summing to 1.0.

        Raises:
            ValueError: If returns_window is empty or optimization processing fails.
        """
        if returns_window.empty or returns_window.shape[1] == 0:
            error_msg = "returns_window must not be empty."
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_obs, n_assets = returns_window.shape
        tickers = returns_window.columns.tolist()

        if n_obs < 2 or n_assets < 2:
            error_msg = f"Insufficient data/assets for clustering. Shape: {returns_window.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 1. Estimate Covariance using specified method
        cov_matrix = self._estimator_fn(returns_window, **self.estimator_kwargs)

        # 2. Optionally Denoise using Marchenko-Pastur filter
        if self.denoise:
            cov_matrix = denoise_covariance(cov_matrix, n_obs=n_obs)

        # 3. Symmetrize to handle numeric floating precision issues
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
        variances = np.diag(cov_matrix)
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
            error_msg = "HRP requires every asset to have a finite, strictly positive variance."
            logger.error(error_msg)
            raise ValueError(error_msg)

        cov_df = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)

        # 4. Derive Correlation Matrix from the processed Covariance Matrix
        std_devs = np.sqrt(variances)
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        corr_df = pd.DataFrame(corr_matrix, index=tickers, columns=tickers)

        # 5. Compute Angular Distance Matrix for clustering
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr_df), 0.0, 1.0))

        # 6. Hierarchical Tree Clustering & Quasi-Diagonalization (Seriation)
        try:
            condensed_dist = scipy.spatial.distance.squareform(
                dist.to_numpy(),
                checks=False
            )
            linkage = sch.linkage(condensed_dist, method=self.linkage_method)
            sorted_indices = self._get_quasi_diag(linkage)
            sorted_tickers = corr_df.columns[sorted_indices].tolist()
        except Exception as exc:
            error_msg = f"Hierarchical clustering failed in HRP strategy: {exc}"
            logger.error(error_msg)
            raise ValueError(error_msg) from exc

        # 7. Recursive Bisection top-down weight allocation
        raw_weights_series = self._recursive_bisection(cov_df, sorted_tickers)

        # 8. Align back to the original returns window ticker sorting order
        ordered_weights = raw_weights_series.reindex(tickers).to_numpy()

        # 9. Verify and normalise weights via BaseStrategy interface
        weights = self._normalise_weights(ordered_weights)

        logger.debug(
            "%s: weights computed successfully for %d assets",
            self.__repr__(),
            n_assets,
        )
        return weights


    def _get_quasi_diag(self, linkage: np.ndarray) -> list[int]:
        """Sorts clustering linkage to find adjacent assets (Matrix Seriation)."""
        link = linkage.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix = self._index_motion(sort_ix, link, num_items)
        return sort_ix.tolist()


    def _index_motion(self, sort_ix: pd.Series, link: np.ndarray, num_items: int) -> pd.Series:
        """Helper function to trace clusters recursively down the tree structure."""
        new_sort_ix = pd.Series(dtype=int)
        for i in sort_ix:
            if i >= num_items:
                idx = i - num_items
                new_sort_ix = pd.concat([new_sort_ix, pd.Series([link[idx, 0], link[idx, 1]])])
            else:
                new_sort_ix = pd.concat([new_sort_ix, pd.Series([i])])
        return new_sort_ix.reset_index(drop=True)


    def _get_inverse_var_weights(self, cov_slice: pd.DataFrame) -> np.ndarray:
        """Computes traditional inverse-variance weights for a sub-cluster slice."""
        inv_var = 1.0 / np.diag(cov_slice)
        return inv_var / np.sum(inv_var)


    def _get_cluster_var(self, cov: pd.DataFrame, cluster_indices: list[str]) -> float:
        """Calculates the combined pooled variance of a specific cluster asset group."""
        cov_slice = cov.loc[cluster_indices, cluster_indices]
        w = self._get_inverse_var_weights(cov_slice)
        return float(np.dot(w, np.dot(cov_slice, w)))


    def _recursive_bisection(self, cov: pd.DataFrame, sort_ix: list[str]) -> pd.Series:
        """Recursively assigns weights top-down using tree bisection splits."""
        weights = pd.Series(1.0, index=sort_ix)
        cluster_list = [sort_ix]

        while len(cluster_list) > 0:
            cluster_list = [
                c[i:j] for c in cluster_list
                for i, j in ((0, len(c) // 2), (len(c) // 2, len(c)))
                if len(c) > 1
            ]

            for i in range(0, len(cluster_list), 2):
                c_left = cluster_list[i]
                c_right = cluster_list[i + 1]

                v_left = self._get_cluster_var(cov, c_left)
                v_right = self._get_cluster_var(cov, c_right)

                # Split weight allocation factor inversely proportional to sub-cluster variance
                alpha = 1.0 - v_left / (v_left + v_right)

                weights.loc[c_left] *= alpha
                weights.loc[c_right] *= (1.0 - alpha)

        return weights