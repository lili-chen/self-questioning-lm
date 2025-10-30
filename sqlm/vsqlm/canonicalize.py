"""
Canonicalization module for V-SQLM.

Implements domain-aware equivalence mapping to group equivalent answers.
"""

from typing import Sequence, Dict, List, TypedDict
import re
import logging

logger = logging.getLogger(__name__)


class CanonicalClass(TypedDict):
    """A class of equivalent answers."""
    canonical: str  # canonical representative
    members: List[int]  # indices into the samples list
    passes: bool  # whether verify-first passed


def canonicalize(
    domain: str,
    samples: Sequence[dict],
    verifier: 'Verifier | None' = None,
    **kwargs
) -> Dict[str, CanonicalClass]:
    """
    Domain-aware equivalence mapping for samples.

    Groups samples into equivalence classes based on domain-specific rules.

    Parameters
    ----------
    domain : str
        Domain type: 'math', 'code', or 'text'.
    samples : Sequence[dict]
        List of samples with 'answer' field.
    verifier : Verifier | None
        Optional verifier for verify-first short-circuiting.
    **kwargs
        Additional domain-specific parameters.

    Returns
    -------
    Dict[str, CanonicalClass]
        Mapping from canonical key to equivalence class.

    Examples
    --------
    >>> samples = [{'answer': '42'}, {'answer': '42.0'}, {'answer': '43'}]
    >>> classes = canonicalize('math', samples)
    >>> len(classes)
    2
    """
    if domain == 'math':
        return _canonicalize_math(samples, verifier, **kwargs)
    elif domain == 'code':
        return _canonicalize_code(samples, verifier, **kwargs)
    elif domain == 'text':
        return _canonicalize_text(samples, verifier, **kwargs)
    else:
        raise ValueError(f"Unknown domain: {domain}")


def _canonicalize_math(
    samples: Sequence[dict],
    verifier: 'Verifier | None' = None,
    tolerance: float = 1e-6,
    **kwargs
) -> Dict[str, CanonicalClass]:
    """
    Canonicalize mathematical answers.

    Handles:
    - Numeric equivalence (42, 42.0, 42.00)
    - Expression simplification via sympy
    - Unit normalization

    Parameters
    ----------
    samples : Sequence[dict]
        Samples with 'answer' field.
    verifier : Verifier | None
        Optional verifier.
    tolerance : float
        Numerical tolerance for comparison.

    Returns
    -------
    Dict[str, CanonicalClass]
        Equivalence classes.
    """
    classes: Dict[str, CanonicalClass] = {}

    for idx, sample in enumerate(samples):
        answer = sample['answer'].strip()

        # Extract numeric value
        canonical = _normalize_math_answer(answer, tolerance)

        # Check if passes verification (verify-first)
        passes = False
        if verifier is not None:
            passes = verifier.passes(answer, **kwargs)

        if canonical in classes:
            classes[canonical]['members'].append(idx)
            # Update passes if any member passes
            if passes:
                classes[canonical]['passes'] = True
        else:
            classes[canonical] = CanonicalClass(
                canonical=canonical,
                members=[idx],
                passes=passes
            )

    return classes


def _normalize_math_answer(answer: str, tolerance: float = 1e-6) -> str:
    """
    Normalize mathematical answer to canonical form.

    Parameters
    ----------
    answer : str
        Raw answer string.
    tolerance : float
        Numerical tolerance.

    Returns
    -------
    str
        Canonical form.
    """
    # Remove common units and punctuation
    answer = re.sub(r'[,$%]', '', answer)
    answer = answer.strip()

    # Try to parse as number
    try:
        # Handle fractions
        if '/' in answer:
            parts = answer.split('/')
            if len(parts) == 2:
                num = float(parts[0].strip())
                denom = float(parts[1].strip())
                if denom != 0:
                    value = num / denom
                else:
                    value = float(answer)
            else:
                value = float(answer)
        else:
            value = float(answer)

        # Round to tolerance
        if abs(value - round(value)) < tolerance:
            return str(int(round(value)))
        else:
            return f"{value:.6f}".rstrip('0').rstrip('.')

    except (ValueError, AttributeError):
        pass

    # Try sympy simplification
    try:
        import sympy
        expr = sympy.sympify(answer, evaluate=True)
        simplified = str(sympy.simplify(expr))
        return simplified
    except:
        pass

    # Fallback: lowercase and strip
    return answer.lower().strip()


def _canonicalize_code(
    samples: Sequence[dict],
    verifier: 'Verifier | None' = None,
    **kwargs
) -> Dict[str, CanonicalClass]:
    """
    Canonicalize code answers.

    Groups by test pass/fail status. Code that passes tests forms
    a single equivalence class; code that fails is grouped by
    failure signature (AST hash or I/O pattern).

    Parameters
    ----------
    samples : Sequence[dict]
        Samples with 'answer' field.
    verifier : Verifier | None
        Code verifier (required for meaningful grouping).

    Returns
    -------
    Dict[str, CanonicalClass]
        Equivalence classes.
    """
    classes: Dict[str, CanonicalClass] = {}

    for idx, sample in enumerate(samples):
        answer = sample['answer'].strip()

        # Check verification
        passes = False
        failure_sig = None

        if verifier is not None:
            passes = verifier.passes(answer, **kwargs)
            if not passes:
                # Get failure signature (e.g., hash of errors or AST)
                failure_sig = _get_code_failure_signature(answer, kwargs.get('test_results'))

        # Canonical key
        if passes:
            canonical = "PASSED"
        else:
            canonical = f"FAILED:{failure_sig or hash(answer) % 10000}"

        if canonical in classes:
            classes[canonical]['members'].append(idx)
            if passes:
                classes[canonical]['passes'] = True
        else:
            classes[canonical] = CanonicalClass(
                canonical=canonical,
                members=[idx],
                passes=passes
            )

    return classes


def _get_code_failure_signature(code: str, test_results: dict | None = None) -> str:
    """
    Compute failure signature for code.

    Parameters
    ----------
    code : str
        Code string.
    test_results : dict | None
        Optional test results with error information.

    Returns
    -------
    str
        Failure signature.
    """
    if test_results and 'error' in test_results:
        # Hash error message
        error_str = str(test_results['error'])
        return str(hash(error_str) % 10000)

    # Try AST-based signature
    try:
        import ast
        tree = ast.parse(code)
        # Simple structural hash
        node_types = [type(node).__name__ for node in ast.walk(tree)]
        return str(hash(tuple(node_types)) % 10000)
    except:
        pass

    # Fallback: hash of code
    return str(hash(code) % 10000)


def _canonicalize_text(
    samples: Sequence[dict],
    verifier: 'Verifier | None' = None,
    levenshtein_threshold: int = 2,
    **kwargs
) -> Dict[str, CanonicalClass]:
    """
    Canonicalize text answers.

    Groups by:
    - Exact match after normalization (lowercase, strip, depunctuate)
    - Fuzzy match with Levenshtein distance ≤ threshold

    Parameters
    ----------
    samples : Sequence[dict]
        Samples with 'answer' field.
    verifier : Verifier | None
        Optional verifier.
    levenshtein_threshold : int
        Max edit distance for fuzzy matching.

    Returns
    -------
    Dict[str, CanonicalClass]
        Equivalence classes.
    """
    classes: Dict[str, CanonicalClass] = {}

    for idx, sample in enumerate(samples):
        answer = sample['answer'].strip()

        # Normalize
        canonical = _normalize_text_answer(answer)

        # Check verification
        passes = False
        if verifier is not None:
            passes = verifier.passes(answer, **kwargs)

        # Try exact match first
        if canonical in classes:
            classes[canonical]['members'].append(idx)
            if passes:
                classes[canonical]['passes'] = True
            continue

        # Try fuzzy match
        matched = False
        for existing_key in list(classes.keys()):
            if _levenshtein_distance(canonical, existing_key) <= levenshtein_threshold:
                classes[existing_key]['members'].append(idx)
                if passes:
                    classes[existing_key]['passes'] = True
                matched = True
                break

        if not matched:
            classes[canonical] = CanonicalClass(
                canonical=canonical,
                members=[idx],
                passes=passes
            )

    return classes


def _normalize_text_answer(answer: str) -> str:
    """
    Normalize text answer to canonical form.

    Parameters
    ----------
    answer : str
        Raw answer string.

    Returns
    -------
    str
        Normalized form.
    """
    # Lowercase
    answer = answer.lower()

    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)

    # Strip whitespace
    answer = ' '.join(answer.split())

    return answer


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Parameters
    ----------
    s1, s2 : str
        Input strings.

    Returns
    -------
    int
        Edit distance.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
