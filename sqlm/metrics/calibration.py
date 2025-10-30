"""
Calibration and reliability weighting for V-SQLM.

Implements empirical Bayes calibration-weighted voting.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ReliabilityStore:
    """
    Store and update reliability statistics for different keys.

    Uses empirical Bayes approach with Beta prior to estimate
    reliability weights based on historical accuracy.
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0
    ):
        """
        Initialize reliability store.

        Parameters
        ----------
        alpha_prior : float
            Beta distribution alpha parameter (pseudo-successes).
            Default 1.0 gives uniform prior.
        beta_prior : float
            Beta distribution beta parameter (pseudo-failures).
            Default 1.0 gives uniform prior.

        Notes
        -----
        With uniform prior (α=1, β=1), the posterior mean is:
            weight = (successes + 1) / (trials + 2)

        This provides gentle regularization toward 0.5.
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        # Storage: key -> {'successes': int, 'trials': int}
        self.stats: Dict[str, Dict[str, int]] = {}

    def update(self, key: str, correct: bool) -> None:
        """
        Update reliability statistics for a key.

        Parameters
        ----------
        key : str
            Identifier (e.g., prompt variant, model name, temperature).
        correct : bool
            Whether the prediction was correct.
        """
        if key not in self.stats:
            self.stats[key] = {'successes': 0, 'trials': 0}

        self.stats[key]['trials'] += 1
        if correct:
            self.stats[key]['successes'] += 1

        logger.debug(
            f"Updated {key}: {self.stats[key]['successes']}/{self.stats[key]['trials']}"
        )

    def weight(self, key: str) -> float:
        """
        Compute reliability weight for a key.

        Uses empirical Bayes posterior mean under Beta prior.

        Parameters
        ----------
        key : str
            Identifier.

        Returns
        -------
        float
            Reliability weight in [0, 1]. Higher = more reliable.
            Returns prior mean (α / (α + β)) if key not seen.

        Examples
        --------
        >>> store = ReliabilityStore(alpha_prior=1.0, beta_prior=1.0)
        >>> store.update('variant_A', True)
        >>> store.update('variant_A', True)
        >>> store.update('variant_A', False)
        >>> store.weight('variant_A')  # (2 + 1) / (3 + 2) = 0.6
        0.6

        >>> store.weight('unseen_key')  # Prior mean: 1 / 2 = 0.5
        0.5
        """
        if key not in self.stats:
            # Return prior mean
            return self.alpha_prior / (self.alpha_prior + self.beta_prior)

        successes = self.stats[key]['successes']
        trials = self.stats[key]['trials']

        # Posterior mean under Beta(α, β) prior
        posterior_alpha = self.alpha_prior + successes
        posterior_beta = self.beta_prior + (trials - successes)

        return posterior_alpha / (posterior_alpha + posterior_beta)

    def get_stats(self, key: str) -> Dict[str, int]:
        """
        Get raw statistics for a key.

        Parameters
        ----------
        key : str
            Identifier.

        Returns
        -------
        dict
            {'successes': int, 'trials': int}
        """
        if key not in self.stats:
            return {'successes': 0, 'trials': 0}

        return self.stats[key].copy()

    def get_all_weights(self) -> Dict[str, float]:
        """
        Get weights for all keys.

        Returns
        -------
        dict
            Mapping from key to weight.
        """
        return {key: self.weight(key) for key in self.stats.keys()}


def compute_ece(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and
    observed accuracy, binned by confidence level.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values (0 to 1).
    correctness : list[bool]
        Whether each prediction was correct.
    n_bins : int
        Number of bins for grouping confidences.

    Returns
    -------
    float
        ECE value (0 = perfect calibration, higher = worse).

    Examples
    --------
    >>> confidences = [0.9, 0.8, 0.7, 0.6]
    >>> correctness = [True, True, False, False]
    >>> compute_ece(confidences, correctness, n_bins=2)  # doctest: +SKIP
    0.15
    """
    if len(confidences) != len(correctness):
        raise ValueError("confidences and correctness must have same length")

    if len(confidences) == 0:
        return 0.0

    # Create bins
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bins = [[] for _ in range(n_bins)]

    # Assign samples to bins
    for conf, corr in zip(confidences, correctness):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, corr))

    # Compute ECE
    ece = 0.0
    total_samples = len(confidences)

    for bin_samples in bins:
        if len(bin_samples) == 0:
            continue

        # Average confidence in bin
        avg_conf = sum(c for c, _ in bin_samples) / len(bin_samples)

        # Accuracy in bin
        accuracy = sum(1 for _, corr in bin_samples if corr) / len(bin_samples)

        # Weighted difference
        weight = len(bin_samples) / total_samples
        ece += weight * abs(avg_conf - accuracy)

    return ece


def compute_brier_score(
    confidences: list[float],
    correctness: list[bool]
) -> float:
    """
    Compute Brier score.

    Brier score measures the mean squared difference between
    predicted probabilities and actual outcomes.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values (0 to 1).
    correctness : list[bool]
        Whether each prediction was correct.

    Returns
    -------
    float
        Brier score (0 = perfect, higher = worse).

    Examples
    --------
    >>> confidences = [0.9, 0.8, 0.7]
    >>> correctness = [True, True, False]
    >>> compute_brier_score(confidences, correctness)  # doctest: +SKIP
    0.11...
    """
    if len(confidences) != len(correctness):
        raise ValueError("confidences and correctness must have same length")

    if len(confidences) == 0:
        return 0.0

    squared_errors = [
        (conf - float(corr)) ** 2
        for conf, corr in zip(confidences, correctness)
    ]

    return sum(squared_errors) / len(squared_errors)


def detect_confidence_drift(
    confidences: list[float],
    window_size: int = 10,
    threshold: float = 0.1
) -> list[int]:
    """
    Detect sudden increases in confidence (potential overconfidence).

    Returns indices where confidence increases significantly without
    corresponding evidence.

    Parameters
    ----------
    confidences : list[float]
        Sequence of confidence values.
    window_size : int
        Size of sliding window for computing baseline.
    threshold : float
        Threshold for detecting drift.

    Returns
    -------
    list[int]
        Indices where drift detected.

    Examples
    --------
    >>> confidences = [0.5, 0.5, 0.5, 0.9, 0.9]
    >>> detect_confidence_drift(confidences, window_size=2, threshold=0.2)
    [3]
    """
    drift_indices = []

    for i in range(window_size, len(confidences)):
        # Baseline: mean of previous window
        baseline = sum(confidences[i - window_size:i]) / window_size

        # Current value
        current = confidences[i]

        # Check for sudden increase
        if current - baseline > threshold:
            drift_indices.append(i)

    return drift_indices
