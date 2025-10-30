"""Unit tests for curriculum reward."""

import pytest
from sqlm.curriculum.proposer_reward import (
    peaked_difficulty,
    proposer_reward,
    estimate_p_star_from_votes
)


class TestPeakedDifficulty:
    """Test peaked difficulty function."""

    def test_maximum_at_half(self):
        """Difficulty should be maximum at p=0.5."""
        assert peaked_difficulty(0.5) == 1.0

    def test_zero_at_extremes(self):
        """Difficulty should be zero at p=0 and p=1."""
        assert peaked_difficulty(0.0) == 0.0
        assert peaked_difficulty(1.0) == 0.0

    def test_symmetry(self):
        """Difficulty should be symmetric around 0.5."""
        assert abs(peaked_difficulty(0.3) - peaked_difficulty(0.7)) < 1e-6
        assert abs(peaked_difficulty(0.2) - peaked_difficulty(0.8)) < 1e-6

    def test_known_values(self):
        """Test known values."""
        # p=0.25: 4 * 0.25 * 0.75 = 0.75
        assert abs(peaked_difficulty(0.25) - 0.75) < 1e-6

        # p=0.75: 4 * 0.75 * 0.25 = 0.75
        assert abs(peaked_difficulty(0.75) - 0.75) < 1e-6

    def test_monotonicity(self):
        """Difficulty should increase from 0 to 0.5."""
        values = [peaked_difficulty(p) for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]


class TestProposerReward:
    """Test proposer reward computation."""

    def test_basic_reward(self):
        """Test basic reward computation."""
        # Good problem: p*=0.5, high entropy, correct, not ill-posed
        reward = proposer_reward(
            p_star=0.5,
            entropy=0.8,
            correct=True,
            ill_posed=False,
            alpha=1.0,
            beta=0.5,
            gamma=-2.0,
            delta=-5.0
        )

        # Should be positive
        assert reward > 0

        # Should equal alpha * 1.0 + beta * 0.8 = 1.0 + 0.4 = 1.4
        expected = 1.0 * 1.0 + 0.5 * 0.8
        assert abs(reward - expected) < 1e-6

    def test_incorrect_penalty(self):
        """Test incorrectness penalty."""
        reward_correct = proposer_reward(
            p_star=0.5, entropy=0.5, correct=True, ill_posed=False
        )

        reward_incorrect = proposer_reward(
            p_star=0.5, entropy=0.5, correct=False, ill_posed=False
        )

        # Incorrect should have lower reward
        assert reward_incorrect < reward_correct

        # Difference should be gamma
        assert abs((reward_correct - reward_incorrect) - 2.0) < 1e-6

    def test_ill_posed_penalty(self):
        """Test ill-posed penalty."""
        reward_normal = proposer_reward(
            p_star=0.5, entropy=0.5, correct=True, ill_posed=False
        )

        reward_ill_posed = proposer_reward(
            p_star=0.5, entropy=0.5, correct=True, ill_posed=True
        )

        # Ill-posed should have much lower reward
        assert reward_ill_posed < reward_normal

        # Difference should be delta
        assert abs((reward_normal - reward_ill_posed) - 5.0) < 1e-6

    def test_difficulty_effect(self):
        """Test effect of difficulty."""
        # p*=0.5 should give better reward than p*=0.9
        reward_optimal = proposer_reward(
            p_star=0.5, entropy=0.5, correct=True, ill_posed=False
        )

        reward_easy = proposer_reward(
            p_star=0.9, entropy=0.5, correct=True, ill_posed=False
        )

        assert reward_optimal > reward_easy


class TestEstimatePStar:
    """Test p* estimation from votes."""

    def test_verified_estimate(self):
        """Verified answers should use leader share directly."""
        p_star = estimate_p_star_from_votes(
            leader_share=0.7,
            total_samples=10,
            verified=True
        )

        assert p_star == 0.7

    def test_unverified_discount(self):
        """Unverified answers should be discounted."""
        p_star = estimate_p_star_from_votes(
            leader_share=0.7,
            total_samples=10,
            verified=False
        )

        # Should be less than leader share
        assert p_star < 0.7
        assert p_star > 0.4  # But not too low

    def test_sample_size_effect(self):
        """More samples should reduce discount."""
        p_star_few = estimate_p_star_from_votes(
            leader_share=0.7,
            total_samples=5,
            verified=False
        )

        p_star_many = estimate_p_star_from_votes(
            leader_share=0.7,
            total_samples=50,
            verified=False
        )

        # More samples → higher confidence → less discount
        assert p_star_many > p_star_few
