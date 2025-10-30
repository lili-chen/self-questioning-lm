"""
Text verifiers for V-SQLM.

Implements verification logic for text generation tasks.
"""

import re
import logging

logger = logging.getLogger(__name__)


class TextVerifier:
    """
    Verifier for text generation tasks.

    Checks exact or fuzzy match against ground truth.
    """

    def __init__(self, case_sensitive: bool = False):
        """
        Initialize text verifier.

        Parameters
        ----------
        case_sensitive : bool
            Whether to perform case-sensitive comparison.
        """
        self.case_sensitive = case_sensitive

    def passes(
        self,
        answer: str,
        ground_truth: str | list[str] | None = None,
        **kwargs
    ) -> bool:
        """
        Check if answer matches ground truth.

        Parameters
        ----------
        answer : str
            Proposed answer.
        ground_truth : str | list[str] | None
            Ground truth answer(s). If list, passes if any match.

        Returns
        -------
        bool
            True if answer matches.
        """
        if ground_truth is None:
            logger.warning("TextVerifier.passes called without ground_truth")
            return False

        # Handle multiple acceptable answers
        if isinstance(ground_truth, list):
            return any(self._matches(answer, gt) for gt in ground_truth)

        return self._matches(answer, ground_truth)

    def margin(
        self,
        answer: str,
        ground_truth: str | list[str] | None = None,
        **kwargs
    ) -> float:
        """
        Compute margin (negative edit distance or 0 for exact match).

        Parameters
        ----------
        answer : str
            Proposed answer.
        ground_truth : str | list[str] | None
            Ground truth.

        Returns
        -------
        float
            Margin (higher is better).
        """
        if ground_truth is None:
            return float('-inf')

        if isinstance(ground_truth, list):
            # Best match
            margins = [self._compute_margin(answer, gt) for gt in ground_truth]
            return max(margins)

        return self._compute_margin(answer, ground_truth)

    def _matches(self, answer: str, ground_truth: str) -> bool:
        """Check if answer matches ground truth."""
        if self.case_sensitive:
            return answer.strip() == ground_truth.strip()
        else:
            return answer.strip().lower() == ground_truth.strip().lower()

    def _compute_margin(self, answer: str, ground_truth: str) -> float:
        """Compute margin for single ground truth."""
        if self._matches(answer, ground_truth):
            return 0.0

        # Use negative edit distance
        from sqlm.vsqlm.canonicalize import _levenshtein_distance
        dist = _levenshtein_distance(answer.strip(), ground_truth.strip())
        return -float(dist)


class RegexVerifier:
    """
    Verifier that checks if answer matches a regular expression.

    Useful for constrained generation tasks.
    """

    def __init__(self, pattern: str | None = None):
        """
        Initialize regex verifier.

        Parameters
        ----------
        pattern : str | None
            Regular expression pattern. Can be overridden in passes().
        """
        self.pattern = pattern

    def passes(
        self,
        answer: str,
        pattern: str | None = None,
        **kwargs
    ) -> bool:
        """
        Check if answer matches regex pattern.

        Parameters
        ----------
        answer : str
            Proposed answer.
        pattern : str | None
            Regex pattern (overrides self.pattern if provided).

        Returns
        -------
        bool
            True if answer matches pattern.
        """
        pattern_to_use = pattern or self.pattern

        if pattern_to_use is None:
            logger.warning("RegexVerifier.passes called without pattern")
            return False

        try:
            match = re.search(pattern_to_use, answer)
            return match is not None
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            return False

    def margin(self, answer: str, pattern: str | None = None, **kwargs) -> float:
        """
        Compute margin (0 if matches, -inf otherwise).

        Parameters
        ----------
        answer : str
            Proposed answer.
        pattern : str | None
            Regex pattern.

        Returns
        -------
        float
            Margin.
        """
        return 0.0 if self.passes(answer, pattern) else float('-inf')


class ContainsVerifier:
    """
    Verifier that checks if answer contains specific keywords or phrases.

    Useful for checking if generated text includes required content.
    """

    def __init__(self, case_sensitive: bool = False):
        """
        Initialize contains verifier.

        Parameters
        ----------
        case_sensitive : bool
            Whether to perform case-sensitive search.
        """
        self.case_sensitive = case_sensitive

    def passes(
        self,
        answer: str,
        required_phrases: list[str] | None = None,
        **kwargs
    ) -> bool:
        """
        Check if answer contains all required phrases.

        Parameters
        ----------
        answer : str
            Proposed answer.
        required_phrases : list[str] | None
            List of phrases that must appear in answer.

        Returns
        -------
        bool
            True if all required phrases found.
        """
        if required_phrases is None:
            logger.warning("ContainsVerifier.passes called without required_phrases")
            return False

        if not self.case_sensitive:
            answer = answer.lower()
            required_phrases = [p.lower() for p in required_phrases]

        return all(phrase in answer for phrase in required_phrases)

    def margin(
        self,
        answer: str,
        required_phrases: list[str] | None = None,
        **kwargs
    ) -> float:
        """
        Compute margin (fraction of required phrases found).

        Parameters
        ----------
        answer : str
            Proposed answer.
        required_phrases : list[str] | None
            Required phrases.

        Returns
        -------
        float
            Fraction of phrases found (0.0 to 1.0).
        """
        if required_phrases is None:
            return 0.0

        if not self.case_sensitive:
            answer = answer.lower()
            required_phrases = [p.lower() for p in required_phrases]

        found = sum(1 for phrase in required_phrases if phrase in answer)
        return found / len(required_phrases)


class DummyVerifier:
    """
    Dummy verifier that always returns False (or random).

    Useful for testing vote-only pathways.
    """

    def __init__(self, always_pass: bool = False, pass_rate: float = 0.0):
        """
        Initialize dummy verifier.

        Parameters
        ----------
        always_pass : bool
            If True, always returns True.
        pass_rate : float
            If always_pass is False, returns True with this probability.
        """
        self.always_pass = always_pass
        self.pass_rate = pass_rate

    def passes(self, answer: str, **kwargs) -> bool:
        """Return constant or random result."""
        if self.always_pass:
            return True

        import random
        return random.random() < self.pass_rate

    def margin(self, answer: str, **kwargs) -> float:
        """Return constant margin."""
        return 0.0 if self.passes(answer) else float('-inf')
