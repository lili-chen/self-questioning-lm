"""
Proposer curriculum reward for V-SQLM.

Implements reward shaping based on vote distributions and peaked difficulty.
"""

import logging

logger = logging.getLogger(__name__)


def peaked_difficulty(p_star: float) -> float:
    """
    Compute peaked difficulty score.

    Difficulty is maximized at p* = 0.5 (50% solver success rate),
    representing problems that are neither too easy nor too hard.

    Parameters
    ----------
    p_star : float
        Solver success probability (0 to 1).

    Returns
    -------
    float
        Difficulty score (0 to 1, peaked at 0.5).

    Formula
    -------
    difficulty = 4 * p* * (1 - p*)

    This is a quadratic function that:
    - Returns 0 when p* = 0 or p* = 1 (trivial problems)
    - Returns 1 when p* = 0.5 (maximally informative)

    Examples
    --------
    >>> peaked_difficulty(0.5)
    1.0

    >>> peaked_difficulty(0.0)
    0.0

    >>> peaked_difficulty(1.0)
    0.0

    >>> peaked_difficulty(0.25)
    0.75
    """
    if not 0 <= p_star <= 1:
        logger.warning(f"p_star out of range [0, 1]: {p_star}")
        p_star = max(0.0, min(1.0, p_star))

    return 4.0 * p_star * (1.0 - p_star)


def proposer_reward(
    p_star: float,
    entropy: float,
    correct: bool,
    ill_posed: bool,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = -2.0,
    delta: float = -5.0
) -> float:
    """
    Compute proposer reward with curriculum shaping.

    The reward encourages the proposer to generate:
    1. Problems at optimal difficulty (peaked_difficulty)
    2. Problems with high vote entropy (disagreement → informative)
    3. Avoid ill-posed problems (ambiguous or unsolvable)

    Parameters
    ----------
    p_star : float
        Solver success probability for this problem.
    entropy : float
        Vote entropy (disagreement measure).
    correct : bool
        Whether the solver ultimately succeeded.
    ill_posed : bool
        Whether the problem was ill-posed or ambiguous.
    alpha : float
        Weight for difficulty term (default: 1.0).
    beta : float
        Weight for entropy term (default: 0.5).
    gamma : float
        Penalty weight for incorrect solutions (default: -2.0).
    delta : float
        Penalty weight for ill-posed problems (default: -5.0).

    Returns
    -------
    float
        Reward value.

    Formula
    -------
    reward = α * peaked_difficulty(p*) + β * entropy
             + γ * (1 if not correct else 0)
             + δ * (1 if ill_posed else 0)

    Examples
    --------
    >>> proposer_reward(p_star=0.5, entropy=0.8, correct=True, ill_posed=False)
    1.4

    >>> proposer_reward(p_star=0.5, entropy=0.8, correct=False, ill_posed=False)
    -0.6

    >>> proposer_reward(p_star=0.9, entropy=0.1, correct=True, ill_posed=False)
    0.41...
    """
    # Difficulty term
    difficulty_score = peaked_difficulty(p_star)

    # Entropy term (higher entropy = more disagreement = more informative)
    entropy_score = entropy

    # Incorrectness penalty
    incorrect_penalty = 0.0 if correct else 1.0

    # Ill-posed penalty
    ill_posed_penalty = 1.0 if ill_posed else 0.0

    # Compute total reward
    reward = (
        alpha * difficulty_score
        + beta * entropy_score
        + gamma * incorrect_penalty
        + delta * ill_posed_penalty
    )

    return reward


def estimate_p_star_from_votes(
    leader_share: float,
    total_samples: int,
    verified: bool
) -> float:
    """
    Estimate solver success probability from vote statistics.

    Heuristic: p* ≈ leader_share if verified, else leader_share * confidence_discount.

    Parameters
    ----------
    leader_share : float
        Fraction of samples voting for leader.
    total_samples : int
        Total number of samples.
    verified : bool
        Whether the answer was verified.

    Returns
    -------
    float
        Estimated p*.

    Examples
    --------
    >>> estimate_p_star_from_votes(0.8, 10, verified=True)
    0.8

    >>> estimate_p_star_from_votes(0.8, 10, verified=False)
    0.56...
    """
    if verified:
        # High confidence: use leader share directly
        return leader_share

    # Discount for lack of verification
    # Discount more with fewer samples
    confidence_discount = 0.7 + 0.3 * min(total_samples / 20.0, 1.0)

    return leader_share * confidence_discount


def proposer_reward_from_vsqlm_output(
    vsqlm_result: dict,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = -2.0,
    delta: float = -5.0
) -> float:
    """
    Compute proposer reward directly from V-SQLM output.

    Convenience wrapper around proposer_reward that extracts
    required fields from V-SQLM result dict.

    Parameters
    ----------
    vsqlm_result : dict
        V-SQLM result with keys: 'leader_share', 't', 'verified',
        'vote_entropy', 'ill_posed' (optional).
    alpha, beta, gamma, delta : float
        Reward weights.

    Returns
    -------
    float
        Reward value.

    Examples
    --------
    >>> result = {
    ...     'leader_share': 0.7,
    ...     't': 10,
    ...     'verified': True,
    ...     'vote_entropy': 0.6,
    ...     'ill_posed': False
    ... }
    >>> proposer_reward_from_vsqlm_output(result)  # doctest: +SKIP
    1.14
    """
    # Extract fields
    leader_share = vsqlm_result.get('leader_share', 0.5)
    total_samples = vsqlm_result.get('t', 1)
    verified = vsqlm_result.get('verified', False)
    entropy = vsqlm_result.get('vote_entropy', 0.0)
    ill_posed = vsqlm_result.get('ill_posed', False)

    # Estimate p*
    p_star = estimate_p_star_from_votes(leader_share, total_samples, verified)

    # Correct = verified in this context
    correct = verified

    return proposer_reward(
        p_star=p_star,
        entropy=entropy,
        correct=correct,
        ill_posed=ill_posed,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta
    )
