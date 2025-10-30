"""Unit tests for Wilson stopping criterion."""

import pytest
import math
from sqlm.vsqlm.stopping import (
    wilson_lower_bound,
    should_stop,
    compute_required_samples,
    wilson_upper_bound,
    get_stopping_stats
)


class TestWilsonLowerBound:
    """Test Wilson lower bound computation."""

    def test_edge_cases(self):
        """Test edge cases."""
        # Perfect success
        assert wilson_lower_bound(1.0, 10, z=1.96) < 1.0
        assert wilson_lower_bound(1.0, 10, z=1.96) > 0.7

        # Perfect failure
        assert wilson_lower_bound(0.0, 10, z=1.96) >= 0.0
        assert wilson_lower_bound(0.0, 10, z=1.96) < 0.3

        # 50-50
        bound = wilson_lower_bound(0.5, 10, z=1.96)
        assert 0.2 < bound < 0.5

    def test_monotonicity_with_n(self):
        """Wilson bound should tighten (increase) as n increases."""
        p = 0.7
        bounds = [wilson_lower_bound(p, n, z=1.96) for n in [5, 10, 20, 50]]

        # Bounds should be monotonically increasing
        for i in range(len(bounds) - 1):
            assert bounds[i] <= bounds[i + 1]

    def test_convergence_to_p(self):
        """Wilson bound should converge to p as n → ∞."""
        p = 0.75

        # At large n, bound should be close to p
        bound_large_n = wilson_lower_bound(p, 1000, z=1.96)
        assert abs(bound_large_n - p) < 0.05

    def test_known_values(self):
        """Test against known values."""
        # p=0.8, n=100, z=1.96 should give ~0.72
        bound = wilson_lower_bound(0.8, 100, z=1.96)
        assert 0.70 < bound < 0.75

    def test_invalid_inputs(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            wilson_lower_bound(0.5, 0, z=1.96)  # n <= 0

        with pytest.raises(ValueError):
            wilson_lower_bound(-0.1, 10, z=1.96)  # p < 0

        with pytest.raises(ValueError):
            wilson_lower_bound(1.1, 10, z=1.96)  # p > 1


class TestShouldStop:
    """Test stopping criterion."""

    def test_basic_stopping(self):
        """Test basic stopping logic."""
        # High leader share, enough samples
        assert should_stop(leader_share=0.8, t=10, k_min=5, z=1.96) is True

        # High leader share, too few samples
        assert should_stop(leader_share=0.8, t=3, k_min=5, z=1.96) is False

        # Low leader share, enough samples
        assert should_stop(leader_share=0.55, t=10, k_min=5, z=1.96) is False

    def test_borderline_cases(self):
        """Test borderline cases."""
        # Just at threshold
        # p=0.6, n=20 gives lower bound ~0.42, below 0.5
        assert should_stop(leader_share=0.6, t=20, k_min=5, z=1.96) is False

        # p=0.7, n=20 gives lower bound ~0.53, above 0.5
        assert should_stop(leader_share=0.7, t=20, k_min=5, z=1.96) is True


class TestComputeRequiredSamples:
    """Test required samples computation."""

    def test_high_target_share(self):
        """High target share should need fewer samples."""
        k_high = compute_required_samples(target_share=0.9, threshold=0.5, max_k=50)
        k_low = compute_required_samples(target_share=0.6, threshold=0.5, max_k=50)

        assert k_high < k_low

    def test_impossible_target(self):
        """Target below threshold should return max_k."""
        k = compute_required_samples(target_share=0.4, threshold=0.5, max_k=50)
        assert k == 50


class TestWilsonUpperBound:
    """Test Wilson upper bound."""

    def test_upper_bound_properties(self):
        """Upper bound should be >= p."""
        p = 0.6
        n = 20

        lower = wilson_lower_bound(p, n, z=1.96)
        upper = wilson_upper_bound(p, n, z=1.96)

        assert lower <= p <= upper


class TestGetStoppingStats:
    """Test stopping stats aggregation."""

    def test_stats_format(self):
        """Test that stats have expected keys."""
        stats = get_stopping_stats(
            leader_share=0.7,
            t=10,
            k_min=5,
            k_max=20,
            threshold=0.5,
            z=1.96
        )

        expected_keys = [
            'should_stop', 'lower_bound', 'upper_bound', 'margin',
            'at_min', 'at_max', 'current_share', 'sample_count'
        ]

        for key in expected_keys:
            assert key in stats

    def test_stats_consistency(self):
        """Test that stats are internally consistent."""
        stats = get_stopping_stats(
            leader_share=0.7,
            t=10,
            k_min=5,
            k_max=20
        )

        # Margin should equal lower_bound - threshold
        expected_margin = stats['lower_bound'] - 0.5
        assert abs(stats['margin'] - expected_margin) < 1e-6

        # Should not be at min or max
        assert stats['at_min'] is False
        assert stats['at_max'] is False

        # Current share should match input
        assert stats['current_share'] == 0.7
        assert stats['sample_count'] == 10
