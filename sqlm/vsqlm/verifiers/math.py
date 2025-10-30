"""
Math verifiers for V-SQLM.

Implements verification logic for mathematical problems.
"""

from typing import Protocol
import re
import logging

logger = logging.getLogger(__name__)


class Verifier(Protocol):
    """Protocol for answer verifiers."""

    def passes(self, answer: str, **kwargs) -> bool:
        """Check if answer passes verification."""
        ...

    def margin(self, answer: str, **kwargs) -> float:
        """Optional: compute confidence margin for this answer."""
        ...


class MathVerifier:
    """
    Verifier for mathematical answers.

    Checks exact or approximate equality against ground truth.
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize math verifier.

        Parameters
        ----------
        tolerance : float
            Numerical tolerance for approximate equality.
        """
        self.tolerance = tolerance

    def passes(self, answer: str, ground_truth: str | None = None, **kwargs) -> bool:
        """
        Check if answer matches ground truth.

        Parameters
        ----------
        answer : str
            Proposed answer.
        ground_truth : str | None
            Ground truth answer (required).

        Returns
        -------
        bool
            True if answer is correct.
        """
        if ground_truth is None:
            logger.warning("MathVerifier.passes called without ground_truth")
            return False

        answer_val = self._extract_number(answer)
        truth_val = self._extract_number(ground_truth)

        if answer_val is None or truth_val is None:
            # Fallback to string comparison
            return self._normalize(answer) == self._normalize(ground_truth)

        # Numerical comparison
        return abs(answer_val - truth_val) < self.tolerance

    def margin(self, answer: str, ground_truth: str | None = None, **kwargs) -> float:
        """
        Compute confidence margin (negative absolute error).

        Parameters
        ----------
        answer : str
            Proposed answer.
        ground_truth : str | None
            Ground truth answer.

        Returns
        -------
        float
            Margin (higher is better; 0 is perfect match).
        """
        if ground_truth is None:
            return float('-inf')

        answer_val = self._extract_number(answer)
        truth_val = self._extract_number(ground_truth)

        if answer_val is None or truth_val is None:
            return -1.0 if self.passes(answer, ground_truth) == False else 0.0

        return -abs(answer_val - truth_val)

    def _extract_number(self, text: str) -> float | None:
        """
        Extract numeric value from text.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        float | None
            Extracted number, or None if extraction fails.
        """
        # Remove common units and symbols
        text = re.sub(r'[,$%]', '', text)
        text = text.strip()

        # Try direct float parsing
        try:
            return float(text)
        except ValueError:
            pass

        # Try fraction
        if '/' in text:
            try:
                parts = text.split('/')
                if len(parts) == 2:
                    num = float(parts[0].strip())
                    denom = float(parts[1].strip())
                    if denom != 0:
                        return num / denom
            except ValueError:
                pass

        # Try extracting first number
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

        return None

    def _normalize(self, text: str) -> str:
        """Normalize text for string comparison."""
        return text.lower().strip()


class ConstraintMathVerifier:
    """
    Verifier that checks mathematical constraints.

    Useful for problems like "find x such that x > 10 and x < 20".
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize constraint verifier.

        Parameters
        ----------
        tolerance : float
            Numerical tolerance.
        """
        self.tolerance = tolerance

    def passes(
        self,
        answer: str,
        constraints: list[callable] | None = None,
        **kwargs
    ) -> bool:
        """
        Check if answer satisfies all constraints.

        Parameters
        ----------
        answer : str
            Proposed answer.
        constraints : list[callable] | None
            List of constraint functions that take a number and return bool.

        Returns
        -------
        bool
            True if all constraints satisfied.
        """
        if constraints is None:
            logger.warning("ConstraintMathVerifier.passes called without constraints")
            return False

        value = self._extract_number(answer)
        if value is None:
            return False

        return all(constraint(value) for constraint in constraints)

    def margin(
        self,
        answer: str,
        constraints: list[callable] | None = None,
        **kwargs
    ) -> float:
        """
        Compute margin (number of satisfied constraints).

        Parameters
        ----------
        answer : str
            Proposed answer.
        constraints : list[callable] | None
            List of constraints.

        Returns
        -------
        float
            Fraction of constraints satisfied.
        """
        if constraints is None:
            return float('-inf')

        value = self._extract_number(answer)
        if value is None:
            return 0.0

        satisfied = sum(constraint(value) for constraint in constraints)
        return satisfied / len(constraints)

    def _extract_number(self, text: str) -> float | None:
        """Extract numeric value from text."""
        # Remove common units and symbols
        text = re.sub(r'[,$%]', '', text)
        text = text.strip()

        try:
            return float(text)
        except ValueError:
            pass

        # Try extracting first number
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

        return None
