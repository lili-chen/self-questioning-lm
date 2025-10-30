"""Unit tests for entropy utilities."""

import pytest
import math
from sqlm.stats.entropy import (
    compute_entropy,
    compute_vote_entropy,
    normalize_entropy,
    compute_kl_divergence,
    compute_js_divergence
)


class TestComputeEntropy:
    """Test Shannon entropy computation."""

    def test_deterministic(self):
        """Deterministic distribution should have zero entropy."""
        assert compute_entropy([1.0]) == 0.0
        assert compute_entropy([1.0, 0.0, 0.0]) == 0.0

    def test_uniform_maximum(self):
        """Uniform distribution should have maximum entropy."""
        # For 2 categories: ln(2) ≈ 0.693
        entropy_2 = compute_entropy([0.5, 0.5])
        assert abs(entropy_2 - math.log(2)) < 1e-6

        # For 4 categories: ln(4) ≈ 1.386
        entropy_4 = compute_entropy([0.25, 0.25, 0.25, 0.25])
        assert abs(entropy_4 - math.log(4)) < 1e-6

    def test_skewed_distribution(self):
        """Skewed distribution should have lower entropy."""
        entropy_skewed = compute_entropy([0.8, 0.2])
        entropy_uniform = compute_entropy([0.5, 0.5])

        assert entropy_skewed < entropy_uniform

    def test_normalization(self):
        """Should handle unnormalized distributions."""
        # [2, 2] should normalize to [0.5, 0.5]
        entropy_unnorm = compute_entropy([2.0, 2.0])
        entropy_norm = compute_entropy([0.5, 0.5])

        assert abs(entropy_unnorm - entropy_norm) < 1e-6


class TestComputeVoteEntropy:
    """Test vote entropy from equivalence classes."""

    def test_unanimous_vote(self):
        """Unanimous vote should have zero entropy."""
        classes = {
            '42': {'members': [0, 1, 2, 3], 'canonical': '42'}
        }

        entropy = compute_vote_entropy(classes)
        assert entropy == 0.0

    def test_split_vote(self):
        """Split vote should have high entropy."""
        classes = {
            '42': {'members': [0, 1], 'canonical': '42'},
            '43': {'members': [2, 3], 'canonical': '43'}
        }

        entropy = compute_vote_entropy(classes)

        # Should be ln(2) ≈ 0.693
        assert abs(entropy - math.log(2)) < 1e-6

    def test_skewed_vote(self):
        """Skewed vote should have lower entropy than uniform."""
        classes_skewed = {
            '42': {'members': [0, 1, 2], 'canonical': '42'},
            '43': {'members': [3], 'canonical': '43'}
        }

        classes_uniform = {
            '42': {'members': [0, 1], 'canonical': '42'},
            '43': {'members': [2, 3], 'canonical': '43'}
        }

        entropy_skewed = compute_vote_entropy(classes_skewed)
        entropy_uniform = compute_vote_entropy(classes_uniform)

        assert entropy_skewed < entropy_uniform


class TestNormalizeEntropy:
    """Test entropy normalization."""

    def test_normalization_range(self):
        """Normalized entropy should be in [0, 1]."""
        # Maximum entropy for 4 categories
        max_entropy = math.log(4)
        normalized = normalize_entropy(max_entropy, 4)

        assert normalized == 1.0

        # Zero entropy
        normalized_zero = normalize_entropy(0.0, 4)
        assert normalized_zero == 0.0

        # Half of maximum
        normalized_half = normalize_entropy(max_entropy / 2, 4)
        assert abs(normalized_half - 0.5) < 1e-6


class TestKLDivergence:
    """Test KL divergence."""

    def test_identical_distributions(self):
        """KL(P || P) should be zero."""
        p = [0.5, 0.3, 0.2]
        kl = compute_kl_divergence(p, p)

        assert abs(kl) < 1e-6

    def test_different_distributions(self):
        """KL(P || Q) should be positive for P ≠ Q."""
        p = [0.7, 0.3]
        q = [0.5, 0.5]

        kl = compute_kl_divergence(p, q)
        assert kl > 0

    def test_support_mismatch(self):
        """KL should be infinity if P has support where Q doesn't."""
        p = [1.0, 0.0]
        q = [0.5, 0.5]

        kl = compute_kl_divergence(p, q)
        # Should be finite (p=1.0, q=0.5)
        assert kl < float('inf')

        p = [0.5, 0.5]
        q = [1.0, 0.0]

        kl = compute_kl_divergence(p, q)
        # Should be infinite (p=0.5 where q=0)
        assert kl == float('inf')


class TestJSDivergence:
    """Test Jensen-Shannon divergence."""

    def test_identical_distributions(self):
        """JS(P, P) should be zero."""
        p = [0.5, 0.3, 0.2]
        js = compute_js_divergence(p, p)

        assert abs(js) < 1e-6

    def test_symmetry(self):
        """JS should be symmetric."""
        p = [0.7, 0.3]
        q = [0.4, 0.6]

        js_pq = compute_js_divergence(p, q)
        js_qp = compute_js_divergence(q, p)

        assert abs(js_pq - js_qp) < 1e-6

    def test_bounded(self):
        """JS should be bounded by ln(2)."""
        # Maximum divergence
        p = [1.0, 0.0]
        q = [0.0, 1.0]

        js = compute_js_divergence(p, q)

        # Should be ln(2) ≈ 0.693
        assert abs(js - math.log(2)) < 1e-6

    def test_different_distributions(self):
        """JS should be positive for different distributions."""
        p = [0.7, 0.3]
        q = [0.5, 0.5]

        js = compute_js_divergence(p, q)

        assert js > 0
        assert js < math.log(2)
