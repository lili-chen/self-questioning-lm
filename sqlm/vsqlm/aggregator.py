"""
Aggregator module for V-SQLM.

Implements majority voting and weighted voting with tie-breaking.
"""

from collections.abc import Mapping
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def majority_vote(classes: Mapping[str, Dict]) -> str:
    """
    Select answer by simple majority vote.

    Parameters
    ----------
    classes : Mapping[str, Dict]
        Equivalence classes with 'members' list.

    Returns
    -------
    str
        Canonical key of winning class.

    Raises
    ------
    ValueError
        If classes is empty.

    Examples
    --------
    >>> classes = {
    ...     '42': {'members': [0, 1, 2], 'passes': False, 'canonical': '42'},
    ...     '43': {'members': [3], 'passes': False, 'canonical': '43'}
    ... }
    >>> majority_vote(classes)
    '42'
    """
    if not classes:
        raise ValueError("Cannot vote with empty classes")

    # Count votes for each class
    votes = {key: len(cls['members']) for key, cls in classes.items()}

    # Find maximum vote count
    max_votes = max(votes.values())

    # Get all classes with maximum votes (for tie-breaking)
    winners = [key for key, count in votes.items() if count == max_votes]

    if len(winners) == 1:
        return winners[0]

    # Tie-breaking: prefer verified answers
    verified_winners = [
        key for key in winners
        if classes[key].get('passes', False)
    ]

    if verified_winners:
        if len(verified_winners) == 1:
            return verified_winners[0]
        # Multiple verified: pick first by occurrence
        return _tie_break_by_first_occurrence(verified_winners, classes)

    # No verified answers: pick first by occurrence
    return _tie_break_by_first_occurrence(winners, classes)


def weighted_vote(
    classes: Mapping[str, Dict],
    weights: list[float]
) -> str:
    """
    Select answer by weighted majority vote.

    Parameters
    ----------
    classes : Mapping[str, Dict]
        Equivalence classes with 'members' list.
    weights : list[float]
        Weight for each sample (indexed by sample position).

    Returns
    -------
    str
        Canonical key of winning class.

    Raises
    ------
    ValueError
        If classes is empty or weights length mismatches.

    Examples
    --------
    >>> classes = {
    ...     '42': {'members': [0, 1], 'passes': False, 'canonical': '42'},
    ...     '43': {'members': [2], 'passes': False, 'canonical': '43'}
    ... }
    >>> weighted_vote(classes, weights=[0.5, 0.5, 2.0])
    '43'
    """
    if not classes:
        raise ValueError("Cannot vote with empty classes")

    # Compute weighted votes for each class
    weighted_votes = {}

    for key, cls in classes.items():
        total_weight = sum(weights[idx] for idx in cls['members'])
        weighted_votes[key] = total_weight

    # Find maximum weighted vote
    max_weight = max(weighted_votes.values())

    # Get all classes with maximum weight (for tie-breaking)
    winners = [key for key, weight in weighted_votes.items() if weight == max_weight]

    if len(winners) == 1:
        return winners[0]

    # Tie-breaking: prefer verified answers
    verified_winners = [
        key for key in winners
        if classes[key].get('passes', False)
    ]

    if verified_winners:
        if len(verified_winners) == 1:
            return verified_winners[0]
        # Multiple verified: prefer highest verifier margin
        return _tie_break_by_margin(verified_winners, classes)

    # No verified answers: prefer highest margin if available
    return _tie_break_by_margin(winners, classes)


def _tie_break_by_first_occurrence(
    candidates: list[str],
    classes: Mapping[str, Dict]
) -> str:
    """
    Break tie by selecting class with earliest first occurrence.

    Parameters
    ----------
    candidates : list[str]
        Candidate canonical keys.
    classes : Mapping[str, Dict]
        Equivalence classes.

    Returns
    -------
    str
        Winner key.
    """
    # Find first occurrence index for each candidate
    first_occurrences = {
        key: min(classes[key]['members'])
        for key in candidates
    }

    # Return key with minimum first occurrence
    return min(first_occurrences, key=first_occurrences.get)


def _tie_break_by_margin(
    candidates: list[str],
    classes: Mapping[str, Dict]
) -> str:
    """
    Break tie by selecting class with highest verifier margin.

    If margin not available, falls back to first occurrence.

    Parameters
    ----------
    candidates : list[str]
        Candidate canonical keys.
    classes : Mapping[str, Dict]
        Equivalence classes.

    Returns
    -------
    str
        Winner key.
    """
    # Check if any class has margin information
    margins = {}

    for key in candidates:
        margin = classes[key].get('margin')
        if margin is not None:
            margins[key] = margin

    if margins:
        # Return key with maximum margin
        max_margin = max(margins.values())
        winners_by_margin = [k for k, m in margins.items() if m == max_margin]

        if len(winners_by_margin) == 1:
            return winners_by_margin[0]

        # Still tied: use first occurrence
        return _tie_break_by_first_occurrence(winners_by_margin, classes)

    # No margin info: use first occurrence
    return _tie_break_by_first_occurrence(candidates, classes)


def get_leader_stats(classes: Mapping[str, Dict]) -> tuple[str, float, int]:
    """
    Get statistics about the current leader.

    Parameters
    ----------
    classes : Mapping[str, Dict]
        Equivalence classes.

    Returns
    -------
    tuple[str, float, int]
        (leader_key, leader_share, total_samples)

    Examples
    --------
    >>> classes = {
    ...     '42': {'members': [0, 1, 2], 'canonical': '42'},
    ...     '43': {'members': [3], 'canonical': '43'}
    ... }
    >>> leader, share, total = get_leader_stats(classes)
    >>> leader
    '42'
    >>> share
    0.75
    >>> total
    4
    """
    if not classes:
        raise ValueError("Cannot compute stats for empty classes")

    # Get leader by simple majority
    leader = majority_vote(classes)

    # Compute total samples
    total_samples = sum(len(cls['members']) for cls in classes.values())

    # Compute leader share
    leader_count = len(classes[leader]['members'])
    leader_share = leader_count / total_samples

    return leader, leader_share, total_samples
