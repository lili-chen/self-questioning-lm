"""Unit tests for voting aggregator."""

import pytest
from sqlm.vsqlm.aggregator import (
    majority_vote,
    weighted_vote,
    get_leader_stats
)


class TestMajorityVote:
    """Test majority voting."""

    def test_clear_majority(self):
        """Test voting with clear majority."""
        classes = {
            '42': {'members': [0, 1, 2], 'canonical': '42', 'passes': False},
            '43': {'members': [3], 'canonical': '43', 'passes': False}
        }

        winner = majority_vote(classes)
        assert winner == '42'

    def test_tie_break_by_verification(self):
        """Test tie-breaking by verification."""
        classes = {
            '42': {'members': [0, 1], 'canonical': '42', 'passes': False},
            '43': {'members': [2, 3], 'canonical': '43', 'passes': True}
        }

        winner = majority_vote(classes)
        # Should prefer verified answer
        assert winner == '43'

    def test_tie_break_by_first_occurrence(self):
        """Test tie-breaking by first occurrence."""
        classes = {
            '42': {'members': [1, 2], 'canonical': '42', 'passes': False},
            '43': {'members': [0, 3], 'canonical': '43', 'passes': False}
        }

        winner = majority_vote(classes)
        # Should prefer answer that appeared first (index 0)
        assert winner == '43'

    def test_single_class(self):
        """Test with single class (unanimous)."""
        classes = {
            '42': {'members': [0, 1, 2, 3], 'canonical': '42', 'passes': False}
        }

        winner = majority_vote(classes)
        assert winner == '42'

    def test_empty_classes(self):
        """Test error handling for empty classes."""
        with pytest.raises(ValueError):
            majority_vote({})


class TestWeightedVote:
    """Test weighted voting."""

    def test_weighted_voting(self):
        """Test that weights affect outcome."""
        classes = {
            '42': {'members': [0, 1], 'canonical': '42', 'passes': False},
            '43': {'members': [2], 'canonical': '43', 'passes': False}
        }

        # Equal weights: 42 wins (2 votes vs 1)
        winner = weighted_vote(classes, weights=[1.0, 1.0, 1.0])
        assert winner == '42'

        # Give 43 high weight: 43 wins (2.0 vs 2.0 → tie-break)
        # Actually with weights [1, 1, 2], 43 has weight 2.0 vs 42's 2.0
        # Tie-breaking will pick by first occurrence
        winner = weighted_vote(classes, weights=[1.0, 1.0, 2.0])
        # 42: members [0,1] → weights 1.0 + 1.0 = 2.0
        # 43: members [2] → weight 2.0
        # Tie! First occurrence: 42 appears at index 0
        assert winner == '42'

        # Give 43 higher weight to win clearly
        winner = weighted_vote(classes, weights=[1.0, 1.0, 3.0])
        assert winner == '43'

    def test_empty_classes(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            weighted_vote({}, weights=[])


class TestGetLeaderStats:
    """Test leader statistics computation."""

    def test_leader_stats(self):
        """Test leader statistics."""
        classes = {
            '42': {'members': [0, 1, 2], 'canonical': '42'},
            '43': {'members': [3], 'canonical': '43'}
        }

        leader, share, total = get_leader_stats(classes)

        assert leader == '42'
        assert share == 0.75  # 3 out of 4
        assert total == 4

    def test_unanimous(self):
        """Test unanimous vote."""
        classes = {
            '42': {'members': [0, 1, 2, 3], 'canonical': '42'}
        }

        leader, share, total = get_leader_stats(classes)

        assert leader == '42'
        assert share == 1.0
        assert total == 4

    def test_even_split(self):
        """Test even split."""
        classes = {
            '42': {'members': [0, 1], 'canonical': '42'},
            '43': {'members': [2, 3], 'canonical': '43'}
        }

        leader, share, total = get_leader_stats(classes)

        # Tie-breaking will pick one
        assert leader in ['42', '43']
        assert share == 0.5
        assert total == 4
