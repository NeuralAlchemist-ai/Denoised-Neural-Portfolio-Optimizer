from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for all portfolio allocation strategies.

    Defines the standard interface required by the backtest simulation engine.
    Every concrete strategy must inherit from this class and implement
    `compute_weights`.
    """
    def __repr__(self) -> str:
        """Return the class name as the string representation."""
        return self.__class__.__name__


    @abstractmethod
    def compute_weights(self, returns_window: pd.DataFrame) -> np.ndarray:
        """Calculate optimal asset weights based on a window of historical returns.

        Args:
            returns_window: DataFrame of shape (T, N) containing historical log
                or simple returns for N assets across T trading periods, with
                dates as index and tickers as columns.

        Returns:
            1-D numpy array of shape (N,) containing portfolio weights summing to 1.0.

        Raises:
            ValueError: If input window data is invalid or insufficient for optimization.
        """
        ...


    def _normalise_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalise an array of weights to sum to 1.0.

        Args:
            weights: 1-D numpy array of shape (N,) containing raw portfolio weights.

        Returns:
            1-D numpy array of shape (N,) with weights normalised such that sum(w) = 1.0.

        Raises:
            ValueError: If weights array is empty or the sum of weights is zero.
        """
        total_weight = float(np.sum(weights))
        if np.isclose(total_weight, 0.0):
            error_msg = "Cannot normalise weights that sum to zero."
            logger.error(error_msg)
            raise ValueError(error_msg)

        normalised = weights / total_weight
        logger.debug(
            "%s: weights normalised (original sum: %.4f, target sum: 1.0)",
            self.__class__.__name__,
            total_weight,
        )
        return normalised