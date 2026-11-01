import logging

import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Compute the sample covariance matrix from a window of asset returns.

    Uses the unbiased estimator (denominator T-1) via numpy. This is the
    simplest estimator and serves as the baseline for comparison. It becomes
    unreliable when the number of assets N is not negligible relative to the
    number of observations T (i.e. N/T is not close to zero).

    Args:
        returns: DataFrame of shape (T, N) containing asset log returns,
            with dates as index and ticker symbols as columns.

    Returns:
        Covariance matrix of shape (N, N) as a numpy array.

    Raises:
        ValueError: If the returns DataFrame is empty or has fewer than 2 rows.
    """
    if returns.empty or len(returns) < 2:
        error_msg = "returns must have at least 2 observations."
        logger.error(error_msg)
        raise ValueError(error_msg)

    covariance_matrix = np.cov(returns, rowvar=False)
    logger.debug("Sample covariance computed: shape %s", covariance_matrix.shape)
    return covariance_matrix


def ledoit_wolf_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Compute the Ledoit-Wolf shrinkage covariance matrix estimator.

    Blends the sample covariance matrix toward a scaled identity target using
    an analytically optimal shrinkage intensity. This reduces the eigenvalue
    spreading caused by estimation noise, producing a better-conditioned matrix
    that is always positive definite. Preferred over the sample covariance when
    the N/T ratio is non-negligible.

    Args:
        returns: DataFrame of shape (T, N) containing asset log returns,
            with dates as index and ticker symbols as columns.

    Returns:
        Shrinkage covariance matrix of shape (N, N) as a numpy array.

    Raises:
        ValueError: If the returns DataFrame is empty or has fewer than 2 rows.
    """
    if returns.empty or len(returns) < 2:
        error_msg = "returns must have at least 2 observations."
        logger.error(error_msg)
        raise ValueError(error_msg)

    lw = LedoitWolf()
    lw.fit(returns)

    logger.debug(
        "Ledoit-Wolf covariance computed: shape %s, shrinkage=%.4f",
        lw.covariance_.shape,
        lw.shrinkage_,
    )
    return lw.covariance_


def ewma_covariance(returns: pd.DataFrame, half_life: int) -> np.ndarray:
    """Compute the Exponentially Weighted Moving Average (EWMA) covariance matrix.

    Assigns exponentially decaying weights to observations, giving more
    importance to recent returns. The decay rate is controlled via the half-life:
    an observation half_life days in the past receives half the weight of today's
    observation. Uses adjust=True to apply the mathematically correct weighted
    sum across the full finite window.

    Args:
        returns: DataFrame of shape (T, N) containing asset log returns,
            with dates as index and ticker symbols as columns.
        half_life: Number of trading days after which a past observation's
            weight decays to half. Common values: 21 (monthly), 63 (quarterly).

    Returns:
        EWMA covariance matrix of shape (N, N) as a numpy array, estimated
        at the last date in the returns window.

    Raises:
        ValueError: If the returns DataFrame is empty or has fewer than 2 rows.
    """
    if returns.empty or len(returns) < 2:
        error_msg = "returns must have at least 2 observations."
        logger.error(error_msg)
        raise ValueError(error_msg)

    ewm_df = returns.ewm(halflife=half_life, adjust=True).cov()
    last_date = ewm_df.index.get_level_values(0)[-1]
    ewma_covariance_matrix = ewm_df.loc[last_date].to_numpy()

    logger.debug(
        "EWMA covariance computed: shape %s, half_life=%d",
        ewma_covariance_matrix.shape,
        half_life,
    )
    return ewma_covariance_matrix