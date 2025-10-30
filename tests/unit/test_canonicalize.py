"""Unit tests for canonicalization."""

import pytest
from sqlm.vsqlm.canonicalize import (
    canonicalize,
    _normalize_math_answer,
    _levenshtein_distance
)
from sqlm.vsqlm.verifiers.math import MathVerifier


class TestCanonicalizeMath:
    """Test math canonicalization."""

    def test_numeric_equivalence(self):
        """Test that numeric equivalents are grouped."""
        samples = [
            {'answer': '42'},
            {'answer': '42.0'},
            {'answer': '42.00'},
            {'answer': '43'}
        ]

        classes = canonicalize('math', samples)

        # Should have 2 classes
        assert len(classes) == 2

        # Find class for 42
        class_42 = [c for c in classes.values() if '42' in c['canonical']][0]
        assert len(class_42['members']) == 3

    def test_fraction_equivalence(self):
        """Test fraction normalization."""
        samples = [
            {'answer': '1/2'},
            {'answer': '0.5'},
            {'answer': '2/4'}
        ]

        classes = canonicalize('math', samples)

        # All should be in same class
        assert len(classes) == 1

    def test_verify_first(self):
        """Test verify-first with verifier."""
        verifier = MathVerifier(tolerance=1e-6)

        samples = [
            {'answer': '42'},
            {'answer': '42'},
            {'answer': '43'}
        ]

        # Ground truth is 42
        classes = canonicalize('math', samples, verifier=verifier, ground_truth='42')

        # Class for 42 should pass
        class_42 = [c for c in classes.values() if '42' in c['canonical']][0]
        assert class_42['passes'] is True

        # Class for 43 should not pass
        class_43 = [c for c in classes.values() if '43' in c['canonical']][0]
        assert class_43['passes'] is False


class TestCanonicalizeText:
    """Test text canonicalization."""

    def test_case_normalization(self):
        """Test case-insensitive grouping."""
        samples = [
            {'answer': 'Paris'},
            {'answer': 'paris'},
            {'answer': 'PARIS'},
            {'answer': 'London'}
        ]

        classes = canonicalize('text', samples)

        # Should have 2 classes
        assert len(classes) == 2

        # Paris variants should be grouped
        paris_class = [c for c in classes.values() if 'paris' in c['canonical'].lower()][0]
        assert len(paris_class['members']) == 3

    def test_fuzzy_matching(self):
        """Test Levenshtein-based fuzzy matching."""
        samples = [
            {'answer': 'color'},
            {'answer': 'colour'},  # 1 edit distance
            {'answer': 'coloor'}   # 1 edit distance
        ]

        classes = canonicalize('text', samples, levenshtein_threshold=1)

        # With threshold=1, should group similar variants
        # Exact behavior depends on order, but should have <=2 classes
        assert len(classes) <= 2


class TestLevenshteinDistance:
    """Test Levenshtein distance computation."""

    def test_identical_strings(self):
        """Identical strings have distance 0."""
        assert _levenshtein_distance('hello', 'hello') == 0

    def test_single_substitution(self):
        """Single substitution has distance 1."""
        assert _levenshtein_distance('hello', 'hallo') == 1

    def test_single_insertion(self):
        """Single insertion has distance 1."""
        assert _levenshtein_distance('hello', 'helllo') == 1

    def test_single_deletion(self):
        """Single deletion has distance 1."""
        assert _levenshtein_distance('hello', 'helo') == 1

    def test_empty_strings(self):
        """Distance to empty string is length."""
        assert _levenshtein_distance('hello', '') == 5
        assert _levenshtein_distance('', 'world') == 5

    def test_known_values(self):
        """Test against known values."""
        assert _levenshtein_distance('kitten', 'sitting') == 3
        assert _levenshtein_distance('saturday', 'sunday') == 3


class TestNormalizeMathAnswer:
    """Test math answer normalization."""

    def test_integer_normalization(self):
        """Integers should be normalized consistently."""
        assert _normalize_math_answer('42') == '42'
        assert _normalize_math_answer('42.0') == '42'
        assert _normalize_math_answer('42.00') == '42'

    def test_float_normalization(self):
        """Floats should be normalized to fixed precision."""
        result = _normalize_math_answer('3.14159')
        assert result.startswith('3.14')

    def test_fraction_normalization(self):
        """Fractions should be converted to decimals."""
        result = _normalize_math_answer('1/2')
        assert result == '0.5'

    def test_remove_units(self):
        """Units and symbols should be stripped."""
        assert _normalize_math_answer('$42') == '42'
        assert _normalize_math_answer('42%') == '42'
