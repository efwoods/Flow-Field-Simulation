"""Mixture model for matrix completion"""

from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K, _ = mixture.mu.shape

    log_post = np.zeros((n, K))

    for j in range(K):
        mu_j = mixture.mu[j]
        var_j = mixture.var[j]
        log_p_j = np.log(mixture.p[j] + 1e-16)

        for i in range(n):
            observed = X[i] != 0  # boolean mask
            x_obs = X[i, observed]
            mu_obs = mu_j[observed]

            # log N(x | mu, var) for observed dimensions
            diff = x_obs - mu_obs
            log_prob = -0.5 * np.sum(np.log(2 * np.pi * var_j) + (diff**2) / var_j)

            log_post[i, j] = log_p_j + log_prob

    # Normalize responsibilities
    log_sum = logsumexp(log_post, axis=1, keepdims=True)
    log_post -= log_sum
    post = np.exp(log_post)

    # Total log-likelihood
    ll = np.sum(log_sum)

    return post, ll


def mstep(
    X: np.ndarray,
    post: np.ndarray,
    mixture: GaussianMixture,
    min_variance: float = 0.25,
) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    mu = np.zeros((K, d))
    var = np.zeros(K)
    p = np.zeros(K)

    # Binary mask indicating observed (non-zero) entries
    mask = X != 0

    # Effective sample counts per component
    nk = np.sum(post, axis=0)  # shape: (K,)
    p = nk / n

    for k in range(K):
        for j in range(d):
            observed = mask[:, j]
            weighted_sum = np.sum(post[observed, k] * X[observed, j])
            weight_total = np.sum(post[observed, k])
            mu[k, j] = weighted_sum / weight_total if weight_total > 0 else 0

        total = 0
        denom = 0
        for i in range(n):
            obs = mask[i]
            if np.any(obs):
                diff = X[i, obs] - mu[k, obs]
                total += post[i, k] * np.sum(diff**2)
                denom += post[i, k] * np.sum(obs)
        var[k] = total / (denom + 1e-10)
        var[k] = max(var[k], min_variance)

    return GaussianMixture(mu, var, p)


def run(
    X: np.ndarray, mixture: GaussianMixture, post: np.ndarray
) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """

    prev_ll = None
    ll = None
    while prev_ll is None or np.abs(ll - prev_ll) > 1e-6 * np.abs(ll):
        prev_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post, mixture)
    return mixture, post, ll


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries = 0)
        mixture: a mixture of gaussians

    Returns:
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    K = mixture.mu.shape[0]

    mu = mixture.mu  # (K, d)
    var = mixture.var  # (K,)
    p = mixture.p  # (K,)

    X_pred = X.copy()

    for i in range(n):
        x = X[i]
        observed = x != 0
        missing = ~observed

        if np.all(observed):
            continue  # no missing values

        log_prob = np.zeros(K)
        for k in range(K):
            # Compute log-likelihood of observed entries under component k
            diff = x[observed] - mu[k, observed]
            sq_dist = np.sum(diff**2)
            d_obs = np.sum(observed)

            log_prob[k] = np.log(p[k]) - 0.5 * (
                d_obs * np.log(2 * np.pi * var[k]) + sq_dist / var[k]
            )

        # Convert to responsibilities
        log_prob_norm = logsumexp(log_prob)
        weights = np.exp(log_prob - log_prob_norm)  # shape (K,)

        # Fill in missing values with weighted expected values
        x_filled = x.copy()
        for j in np.where(missing)[0]:
            # Expectation over mixture components
            x_filled[j] = np.sum(weights[k] * mu[k, j] for k in range(K))

        X_pred[i] = x_filled

    return X_pred
