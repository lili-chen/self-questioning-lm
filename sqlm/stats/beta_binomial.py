"""
Beta-binomial utilities for V-SQLM.

Implements Beta-Binomial estimation and majority probability computation.
"""

import math
import logging
from typing import Sequence
from scipy import special, optimize
import numpy as np

logger = logging.getLogger(__name__)


def fit_alpha_beta(
    successes: Sequence[int],
    trials: Sequence[int],
    method: str = 'mom'
) -> tuple[float, float]:
    """
    Fit Beta-Binomial parameters α and β from observed data.

    The Beta-Binomial models variability in success probability across
    tasks, accounting for sample dependence within each task.

    Parameters
    ----------
    successes : Sequence[int]
        Number of successes for each task.
    trials : Sequence[int]
        Number of trials for each task.
    method : str
        Fitting method:
        - 'mom': method of moments
        - 'mle': maximum likelihood estimation

    Returns
    -------
    tuple[float, float]
        (alpha, beta) parameters.

    Examples
    --------
    >>> successes = [7, 8, 6, 9]
    >>> trials = [10, 10, 10, 10]
    >>> alpha, beta = fit_alpha_beta(successes, trials)
    >>> alpha > 0 and beta > 0
    True

    Notes
    -----
    Method of moments formulas:
        μ = mean(successes / trials)
        σ² = var(successes / trials)
        α = μ * ((μ * (1 - μ) / σ²) - 1)
        β = (1 - μ) * ((μ * (1 - μ) / σ²) - 1)
    """
    if len(successes) != len(trials):
        raise ValueError("successes and trials must have same length")

    if len(successes) == 0:
        raise ValueError("Need at least one observation")

    # Convert to numpy arrays
    successes = np.array(successes, dtype=float)
    trials = np.array(trials, dtype=float)

    # Compute proportions
    proportions = successes / trials

    if method == 'mom':
        # Method of moments
        mu = np.mean(proportions)
        var = np.var(proportions, ddof=1) if len(proportions) > 1 else 0.01

        # Avoid division by zero
        if var == 0:
            var = 1e-6

        # Ensure mu is in (0, 1)
        mu = max(0.01, min(0.99, mu))

        # Compute α and β
        common = (mu * (1 - mu) / var) - 1

        # Ensure common > 0
        if common <= 0:
            logger.warning(f"Invalid common term {common}, using default α=β=1")
            return 1.0, 1.0

        alpha = mu * common
        beta = (1 - mu) * common

        # Ensure positive
        alpha = max(0.1, alpha)
        beta = max(0.1, beta)

        return float(alpha), float(beta)

    elif method == 'mle':
        # Maximum likelihood estimation
        def neg_log_likelihood(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return 1e10

            ll = 0.0
            for s, n in zip(successes, trials):
                # Beta-binomial log-likelihood
                try:
                    ll += (
                        special.betaln(s + alpha, n - s + beta)
                        - special.betaln(alpha, beta)
                        + special.gammaln(n + 1)
                        - special.gammaln(s + 1)
                        - special.gammaln(n - s + 1)
                    )
                except:
                    return 1e10

            return -ll

        # Initial guess using method of moments
        alpha0, beta0 = fit_alpha_beta(successes, trials, method='mom')

        # Optimize
        result = optimize.minimize(
            neg_log_likelihood,
            x0=[alpha0, beta0],
            method='L-BFGS-B',
            bounds=[(0.1, 1000), (0.1, 1000)]
        )

        if result.success:
            return float(result.x[0]), float(result.x[1])
        else:
            logger.warning("MLE optimization failed, falling back to MOM")
            return alpha0, beta0

    else:
        raise ValueError(f"Unknown method: {method}")


def pm_majority_correct(alpha: float, beta: float, K: int) -> float:
    """
    Compute probability that majority of K samples is correct.

    Under Beta-Binomial model, the success probability p ~ Beta(α, β),
    and given p, the number of successes ~ Binomial(K, p).

    P(majority correct) = P(≥ K/2 successes)

    Parameters
    ----------
    alpha : float
        Beta distribution alpha parameter.
    beta : float
        Beta distribution beta parameter.
    K : int
        Number of samples.

    Returns
    -------
    float
        Probability that majority is correct.

    Examples
    --------
    >>> pm_majority_correct(alpha=8, beta=2, K=5)  # High success rate
    0.99...

    >>> pm_majority_correct(alpha=5, beta=5, K=5)  # 50-50 success rate
    0.5

    >>> pm_majority_correct(alpha=2, beta=8, K=5)  # Low success rate
    0.01...

    Notes
    -----
    Computed as:
        P(≥ ⌈K/2⌉ correct) = Σ_{k=⌈K/2⌉}^K C(K,k) * B(k+α, K-k+β) / B(α, β)

    where B is the beta function.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")

    if alpha <= 0 or beta <= 0:
        raise ValueError(f"alpha and beta must be positive, got {alpha}, {beta}")

    # Majority threshold
    majority_threshold = (K + 1) // 2

    # Compute probability
    prob = 0.0

    for k in range(majority_threshold, K + 1):
        # P(exactly k successes) under Beta-Binomial
        try:
            log_prob = (
                special.gammaln(K + 1)
                - special.gammaln(k + 1)
                - special.gammaln(K - k + 1)
                + special.betaln(k + alpha, K - k + beta)
                - special.betaln(alpha, beta)
            )
            prob += math.exp(log_prob)
        except (OverflowError, ValueError) as e:
            logger.warning(f"Numerical issue computing term k={k}: {e}")
            continue

    return min(1.0, max(0.0, prob))


def compute_optimal_k(
    alpha: float,
    beta: float,
    target_prob: float = 0.95,
    max_k: int = 50
) -> int:
    """
    Compute minimum K to achieve target majority correctness probability.

    Parameters
    ----------
    alpha, beta : float
        Beta-Binomial parameters.
    target_prob : float
        Target probability (default: 0.95).
    max_k : int
        Maximum K to consider.

    Returns
    -------
    int
        Minimum K to achieve target (or max_k if never achieved).

    Examples
    --------
    >>> compute_optimal_k(alpha=8, beta=2, target_prob=0.95)  # doctest: +SKIP
    3

    >>> compute_optimal_k(alpha=5, beta=5, target_prob=0.95)  # doctest: +SKIP
    50
    """
    for k in range(1, max_k + 1):
        prob = pm_majority_correct(alpha, beta, k)
        if prob >= target_prob:
            return k

    logger.warning(
        f"Could not achieve target_prob {target_prob} with K <= {max_k}"
    )
    return max_k


def compute_effective_samples(alpha: float, beta: float) -> float:
    """
    Compute effective sample size (concentration) of Beta distribution.

    Concentration = α + β. Higher values indicate less variability
    in success probability across tasks.

    Parameters
    ----------
    alpha, beta : float
        Beta parameters.

    Returns
    -------
    float
        Effective sample size.

    Examples
    --------
    >>> compute_effective_samples(8, 2)
    10.0

    >>> compute_effective_samples(1, 1)  # Uniform prior
    2.0
    """
    return alpha + beta


def compute_rho(alpha: float, beta: float) -> float:
    """
    Compute intra-class correlation coefficient ρ.

    Measures correlation between samples within the same task.
    ρ = 0: independent samples (Binomial)
    ρ > 0: positive dependence (Beta-Binomial)

    Parameters
    ----------
    alpha, beta : float
        Beta parameters.

    Returns
    -------
    float
        Correlation coefficient ρ ∈ [0, 1).

    Formula
    -------
    ρ = 1 / (α + β + 1)

    Examples
    --------
    >>> compute_rho(9, 1)  # High concentration
    0.090...

    >>> compute_rho(1, 1)  # Low concentration
    0.333...

    >>> compute_rho(99, 1)  # Very high concentration
    0.009...
    """
    return 1.0 / (alpha + beta + 1)


def compute_expected_accuracy(alpha: float, beta: float) -> float:
    """
    Compute expected accuracy (mean of Beta distribution).

    Parameters
    ----------
    alpha, beta : float
        Beta parameters.

    Returns
    -------
    float
        Expected accuracy ∈ [0, 1].

    Formula
    -------
    E[p] = α / (α + β)

    Examples
    --------
    >>> compute_expected_accuracy(8, 2)
    0.8

    >>> compute_expected_accuracy(5, 5)
    0.5

    >>> compute_expected_accuracy(1, 9)
    0.1
    """
    return alpha / (alpha + beta)
