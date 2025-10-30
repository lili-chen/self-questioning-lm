"""
Entropy utilities for V-SQLM.

Implements entropy computation for vote distributions.
"""

import math
import logging
from typing import Dict, Sequence

logger = logging.getLogger(__name__)


def compute_entropy(distribution: Sequence[float]) -> float:
    """
    Compute Shannon entropy of a probability distribution.

    Parameters
    ----------
    distribution : Sequence[float]
        Probability distribution (should sum to 1).

    Returns
    -------
    float
        Entropy in nats (natural logarithm base).
        Range: [0, ln(n)] where n is number of categories.

    Formula
    -------
    H = -Σ p_i * ln(p_i)

    Examples
    --------
    >>> compute_entropy([1.0])  # Deterministic
    0.0

    >>> compute_entropy([0.5, 0.5])  # Maximum entropy for 2 categories
    0.693...

    >>> compute_entropy([0.25, 0.25, 0.25, 0.25])  # Uniform over 4
    1.386...

    >>> compute_entropy([0.8, 0.2])  # Skewed
    0.500...
    """
    if len(distribution) == 0:
        return 0.0

    # Normalize distribution
    total = sum(distribution)
    if total == 0:
        return 0.0

    normalized = [p / total for p in distribution]

    # Compute entropy
    entropy = 0.0
    for p in normalized:
        if p > 0:
            entropy -= p * math.log(p)

    return entropy


def compute_vote_entropy(classes: Dict[str, dict]) -> float:
    """
    Compute entropy of vote distribution across equivalence classes.

    Parameters
    ----------
    classes : Dict[str, dict]
        Equivalence classes with 'members' lists.

    Returns
    -------
    float
        Vote entropy (0 = unanimous, higher = more disagreement).

    Examples
    --------
    >>> classes = {
    ...     '42': {'members': [0, 1, 2], 'canonical': '42'},
    ...     '43': {'members': [3], 'canonical': '43'}
    ... }
    >>> compute_vote_entropy(classes)  # doctest: +SKIP
    0.562...

    >>> classes = {'42': {'members': [0, 1, 2, 3], 'canonical': '42'}}
    >>> compute_vote_entropy(classes)  # Unanimous
    0.0
    """
    if not classes:
        return 0.0

    # Get vote counts
    vote_counts = [len(cls['members']) for cls in classes.values()]

    return compute_entropy(vote_counts)


def normalize_entropy(entropy: float, n_categories: int) -> float:
    """
    Normalize entropy to [0, 1] range.

    Divides by maximum possible entropy for n categories.

    Parameters
    ----------
    entropy : float
        Raw entropy value.
    n_categories : int
        Number of categories.

    Returns
    -------
    float
        Normalized entropy in [0, 1].

    Examples
    --------
    >>> max_entropy = math.log(4)  # Maximum for 4 categories
    >>> normalize_entropy(max_entropy, 4)
    1.0

    >>> normalize_entropy(0.0, 4)
    0.0

    >>> normalize_entropy(math.log(2), 4)  # Half of maximum
    0.5
    """
    if n_categories <= 1:
        return 0.0

    max_entropy = math.log(n_categories)

    if max_entropy == 0:
        return 0.0

    return entropy / max_entropy


def compute_normalized_vote_entropy(classes: Dict[str, dict]) -> float:
    """
    Compute normalized vote entropy in [0, 1].

    Parameters
    ----------
    classes : Dict[str, dict]
        Equivalence classes.

    Returns
    -------
    float
        Normalized entropy (0 = unanimous, 1 = maximum disagreement).

    Examples
    --------
    >>> classes = {
    ...     '42': {'members': [0, 1], 'canonical': '42'},
    ...     '43': {'members': [2, 3], 'canonical': '43'}
    ... }
    >>> compute_normalized_vote_entropy(classes)  # Perfect split
    1.0

    >>> classes = {'42': {'members': [0, 1, 2, 3], 'canonical': '42'}}
    >>> compute_normalized_vote_entropy(classes)  # Unanimous
    0.0
    """
    if not classes:
        return 0.0

    entropy = compute_vote_entropy(classes)
    n_categories = len(classes)

    return normalize_entropy(entropy, n_categories)


def compute_kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """
    Compute Kullback-Leibler divergence KL(P || Q).

    Measures how distribution P differs from reference distribution Q.

    Parameters
    ----------
    p : Sequence[float]
        Target distribution.
    q : Sequence[float]
        Reference distribution.

    Returns
    -------
    float
        KL divergence (0 if identical, higher = more different).
        Returns infinity if support mismatch (p > 0 where q = 0).

    Formula
    -------
    KL(P || Q) = Σ p_i * ln(p_i / q_i)

    Examples
    --------
    >>> compute_kl_divergence([0.5, 0.5], [0.5, 0.5])
    0.0

    >>> compute_kl_divergence([0.8, 0.2], [0.5, 0.5])  # doctest: +SKIP
    0.223...

    >>> compute_kl_divergence([1.0, 0.0], [0.5, 0.5])  # doctest: +SKIP
    0.693...
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    # Normalize
    p_total = sum(p)
    q_total = sum(q)

    if p_total == 0 or q_total == 0:
        return 0.0

    p_norm = [pi / p_total for pi in p]
    q_norm = [qi / q_total for qi in q]

    # Compute KL divergence
    kl = 0.0

    for pi, qi in zip(p_norm, q_norm):
        if pi > 0:
            if qi == 0:
                # Support mismatch
                return float('inf')
            kl += pi * math.log(pi / qi)

    return kl


def compute_js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """
    Compute Jensen-Shannon divergence.

    Symmetric version of KL divergence, bounded in [0, ln(2)].

    Parameters
    ----------
    p : Sequence[float]
        First distribution.
    q : Sequence[float]
        Second distribution.

    Returns
    -------
    float
        JS divergence (0 if identical, ln(2) ≈ 0.693 at maximum).

    Formula
    -------
    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Examples
    --------
    >>> compute_js_divergence([0.5, 0.5], [0.5, 0.5])
    0.0

    >>> compute_js_divergence([1.0, 0.0], [0.0, 1.0])  # Maximum divergence
    0.693...
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    # Normalize
    p_total = sum(p)
    q_total = sum(q)

    if p_total == 0 or q_total == 0:
        return 0.0

    p_norm = [pi / p_total for pi in p]
    q_norm = [qi / q_total for qi in q]

    # Compute mixture M
    m = [(pi + qi) / 2 for pi, qi in zip(p_norm, q_norm)]

    # JS = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    kl_pm = compute_kl_divergence(p_norm, m)
    kl_qm = compute_kl_divergence(q_norm, m)

    return 0.5 * kl_pm + 0.5 * kl_qm
