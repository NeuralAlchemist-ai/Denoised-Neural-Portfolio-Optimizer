import logging

import numpy as np
from scipy.linalg import eigh

logger = logging.getLogger(__name__)

# Minimum eigenvalue floor to guarantee positive definiteness.
# Below this threshold values are numerically indistinguishable from zero.
_PD_EIGENVALUE_FLOOR = 1e-8


def _compute_mp_threshold(q: float, sigma_sq: float = 1.0) -> float:
    """Compute the Marchenko-Pastur upper spectral edge (noise threshold).

    Any eigenvalue of the correlation matrix at or below this threshold is
    considered indistinguishable from noise under Random Matrix Theory.

    Args:
        q: Ratio of number of assets N to number of observations T (N/T).
            Must be in the range (0, 1].
        sigma_sq: Variance of the assumed random component. For a properly
            standardized correlation matrix this is always 1.0.

    Returns:
        Upper edge of the Marchenko-Pastur distribution λ₊ = σ²(1 + √q)².
    """
    return sigma_sq * (1 + np.sqrt(q)) ** 2


def denoise_covariance(cov_matrix: np.ndarray, n_obs: int) -> np.ndarray:
    """Denoise a covariance matrix using the Marchenko-Pastur spectral filter.

    Converts the covariance matrix to a correlation matrix, identifies noise
    eigenvalues via the Marchenko-Pastur upper spectral edge, replaces them
    with their mean (preserving the total variance), reconstructs a clean
    correlation matrix, enforces positive definiteness, and rescales back to
    a covariance matrix.

    Args:
        cov_matrix: Square covariance matrix of shape (N, N) as a numpy array.
            Must be symmetric and positive semi-definite.
        n_obs: Number of observations T used to estimate the covariance matrix
            (i.e. the rolling window length, typically 252 for daily returns).

    Returns:
        Denoised covariance matrix of shape (N, N) as a numpy array. Guaranteed
        to be symmetric and positive definite.

    Raises:
        ValueError: If cov_matrix is not 2-D, not square, or n_obs is less
            than the number of assets N.
    """
    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        error_msg = "cov_matrix must be a 2-D square matrix."
        logger.error(error_msg)
        raise ValueError(error_msg)

    n_assets = cov_matrix.shape[0]

    if n_obs < n_assets:
        error_msg = (
            f"n_obs ({n_obs}) must be >= n_assets ({n_assets}); "
            "otherwise the covariance matrix is singular."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    std_devs = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

    eigenvalues, eigenvectors = eigh(corr_matrix)

    q = n_assets / n_obs
    mp_upper_threshold = _compute_mp_threshold(q)

    noise_mask = eigenvalues <= mp_upper_threshold
    n_noise = int(noise_mask.sum())
    n_signal = n_assets - n_noise
    logger.debug(
        "MP threshold λ₊=%.4f | signal eigenvalues: %d | noise eigenvalues: %d",
        mp_upper_threshold,
        n_signal,
        n_noise,
    )

    noise_mean = np.mean(eigenvalues[noise_mask])
    eigenvalues[noise_mask] = noise_mean

    eigenvalues_clean = np.maximum(eigenvalues, _PD_EIGENVALUE_FLOOR)

    corr_matrix_clean = eigenvectors @ np.diag(eigenvalues_clean) @ eigenvectors.T
    diag_inv = 1.0 / np.sqrt(np.diag(corr_matrix_clean))
    corr_matrix_clean = corr_matrix_clean * np.outer(diag_inv, diag_inv)

    cov_matrix_clean = np.diag(std_devs) @ corr_matrix_clean @ np.diag(std_devs)

    logger.debug("Denoised covariance matrix computed: shape %s", cov_matrix_clean.shape)
    return cov_matrix_clean