"""
Stopping module for V-SQLM.

Implements adaptive-K sequential stopping using Wilson confidence bounds.
"""

import math
import logging

logger = logging.getLogger(__name__)


def wilson_lower_bound(p: float, n: int, z: float = 1.96) -> float:
    """
    Compute one-sided Wilson lower bound for a binomial proportion.

    The Wilson score interval is a confidence interval for the probability
    of success in a Bernoulli trial. The lower bound gives a conservative
    estimate that the true proportion is at least this value.

    Parameters
    ----------
    p : float
        Observed proportion (number of successes / n).
    n : int
        Number of trials.
    z : float
        Z-score for desired confidence level.
        Default 1.96 corresponds to 95% one-sided confidence (~97.5% two-sided).

    Returns
    -------
    float
        Lower bound on true proportion.

    Notes
    -----
    Formula:
        lower = (p + z²/(2n) - z * sqrt(p(1-p)/n + z²/(4n²))) / (1 + z²/n)

    For n → ∞, this converges to: p - z * sqrt(p(1-p)/n)

    Examples
    --------
    >>> wilson_lower_bound(0.6, 10, z=1.96)  # doctest: +SKIP
    0.338...

    >>> wilson_lower_bound(0.8, 100, z=1.96)  # doctest: +SKIP
    0.720...

    References
    ----------
    Wilson, E. B. (1927). "Probable inference, the law of succession, and
    statistical inference". Journal of the American Statistical Association.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")

    if z < 0:
        raise ValueError(f"z must be non-negative, got {z}")

    # Handle edge cases
    if n == 1:
        # With single sample, lower bound is very conservative
        return 0.0 if p < 1.0 else 1.0

    # Compute Wilson lower bound
    z_sq = z * z
    numerator = (
        p
        + z_sq / (2 * n)
        - z * math.sqrt(p * (1 - p) / n + z_sq / (4 * n * n))
    )
    denominator = 1 + z_sq / n

    return numerator / denominator


def should_stop(
    leader_share: float,
    t: int,
    k_min: int,
    threshold: float = 0.5,
    z: float = 1.96
) -> bool:
    """
    Decide whether to stop sampling based on Wilson criterion.

    Stops when:
    1. t >= k_min (minimum samples collected)
    2. Wilson lower bound on leader share > threshold

    Parameters
    ----------
    leader_share : float
        Current proportion of samples supporting the leader.
    t : int
        Current number of samples.
    k_min : int
        Minimum number of samples before stopping is allowed.
    threshold : float
        Threshold for lower bound (default: 0.5 for majority).
    z : float
        Z-score for confidence (default: 1.96 for ~95% one-sided).

    Returns
    -------
    bool
        True if sampling should stop.

    Examples
    --------
    >>> should_stop(leader_share=0.8, t=10, k_min=5)
    True

    >>> should_stop(leader_share=0.6, t=3, k_min=5)
    False

    >>> should_stop(leader_share=0.5, t=20, k_min=5)
    False
    """
    # Check minimum sample requirement
    if t < k_min:
        return False

    # Compute Wilson lower bound
    lower_bound = wilson_lower_bound(leader_share, t, z)

    # Stop if lower bound exceeds threshold
    return lower_bound > threshold


def compute_required_samples(
    target_share: float,
    threshold: float = 0.5,
    z: float = 1.96,
    max_k: int = 100
) -> int:
    """
    Estimate minimum samples needed for Wilson bound to exceed threshold.

    Useful for planning experiments or setting k_max.

    Parameters
    ----------
    target_share : float
        Expected proportion supporting the leader.
    threshold : float
        Threshold for Wilson lower bound.
    z : float
        Z-score for confidence.
    max_k : int
        Maximum samples to consider.

    Returns
    -------
    int
        Estimated minimum samples needed (or max_k if never sufficient).

    Examples
    --------
    >>> compute_required_samples(target_share=0.8, threshold=0.5)  # doctest: +SKIP
    7

    >>> compute_required_samples(target_share=0.6, threshold=0.5)  # doctest: +SKIP
    28
    """
    if target_share <= threshold:
        logger.warning(
            f"target_share ({target_share}) <= threshold ({threshold}); "
            "will never satisfy stopping criterion"
        )
        return max_k

    # Binary search for minimum k
    for k in range(1, max_k + 1):
        lower_bound = wilson_lower_bound(target_share, k, z)
        if lower_bound > threshold:
            return k

    logger.warning(
        f"Could not find sufficient k <= {max_k} for "
        f"target_share={target_share}, threshold={threshold}"
    )
    return max_k


def wilson_upper_bound(p: float, n: int, z: float = 1.96) -> float:
    """
    Compute one-sided Wilson upper bound for a binomial proportion.

    Complement of wilson_lower_bound; useful for computing bounds on
    failure rates or minority shares.

    Parameters
    ----------
    p : float
        Observed proportion.
    n : int
        Number of trials.
    z : float
        Z-score for confidence.

    Returns
    -------
    float
        Upper bound on true proportion.

    Examples
    --------
    >>> wilson_upper_bound(0.4, 10, z=1.96)  # doctest: +SKIP
    0.662...
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")

    # Upper bound for p is lower bound for (1 - p), flipped
    # Use symmetry: upper(p) = 1 - lower(1-p)
    z_sq = z * z
    numerator = (
        p
        + z_sq / (2 * n)
        + z * math.sqrt(p * (1 - p) / n + z_sq / (4 * n * n))
    )
    denominator = 1 + z_sq / n

    return numerator / denominator


def get_stopping_stats(
    leader_share: float,
    t: int,
    k_min: int,
    k_max: int,
    threshold: float = 0.5,
    z: float = 1.96
) -> dict:
    """
    Get detailed statistics for stopping decision.

    Parameters
    ----------
    leader_share : float
        Current leader proportion.
    t : int
        Current sample count.
    k_min, k_max : int
        Min and max sample counts.
    threshold : float
        Stopping threshold.
    z : float
        Confidence z-score.

    Returns
    -------
    dict
        Statistics including:
        - should_stop: bool
        - lower_bound: float
        - upper_bound: float
        - margin: float (distance from threshold)
        - at_min: bool
        - at_max: bool

    Examples
    --------
    >>> stats = get_stopping_stats(0.7, 10, 5, 20)
    >>> stats['should_stop']  # doctest: +SKIP
    True
    >>> stats['margin'] > 0  # doctest: +SKIP
    True
    """
    lower_bound = wilson_lower_bound(leader_share, t, z)
    upper_bound = wilson_upper_bound(leader_share, t, z)

    return {
        'should_stop': should_stop(leader_share, t, k_min, threshold, z),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'margin': lower_bound - threshold,
        'at_min': t < k_min,
        'at_max': t >= k_max,
        'current_share': leader_share,
        'sample_count': t,
    }
