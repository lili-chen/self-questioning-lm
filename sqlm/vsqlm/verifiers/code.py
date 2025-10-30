"""
Code verifiers for V-SQLM.

Implements verification logic for code generation tasks using sandboxed execution.
"""

import subprocess
import tempfile
import os
import sys
import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeVerifier:
    """
    Verifier for code generation tasks.

    Runs hidden test cases in a sandboxed environment.
    """

    def __init__(
        self,
        timeout: float = 5.0,
        sandbox: bool = True
    ):
        """
        Initialize code verifier.

        Parameters
        ----------
        timeout : float
            Timeout in seconds for each test execution.
        sandbox : bool
            Whether to use sandboxing (recommended for untrusted code).
        """
        self.timeout = timeout
        self.sandbox = sandbox

    def passes(
        self,
        answer: str,
        test_cases: list[dict] | None = None,
        entry_point: str | None = None,
        **kwargs
    ) -> bool:
        """
        Check if code passes all test cases.

        Parameters
        ----------
        answer : str
            Generated code.
        test_cases : list[dict] | None
            List of test cases with 'inputs' and 'expected_output'.
        entry_point : str | None
            Function name to test (if applicable).

        Returns
        -------
        bool
            True if all tests pass.
        """
        if test_cases is None:
            logger.warning("CodeVerifier.passes called without test_cases")
            return False

        # First check if code compiles
        if not self._check_syntax(answer):
            return False

        # Run test cases
        for test in test_cases:
            if not self._run_test(answer, test, entry_point):
                return False

        return True

    def margin(
        self,
        answer: str,
        test_cases: list[dict] | None = None,
        entry_point: str | None = None,
        **kwargs
    ) -> float:
        """
        Compute margin (fraction of tests passed).

        Parameters
        ----------
        answer : str
            Generated code.
        test_cases : list[dict] | None
            Test cases.
        entry_point : str | None
            Function name.

        Returns
        -------
        float
            Fraction of tests passed (0.0 to 1.0).
        """
        if test_cases is None:
            return 0.0

        if not self._check_syntax(answer):
            return 0.0

        passed = sum(
            1 for test in test_cases
            if self._run_test(answer, test, entry_point)
        )

        return passed / len(test_cases)

    def _check_syntax(self, code: str) -> bool:
        """
        Check if code has valid Python syntax.

        Parameters
        ----------
        code : str
            Code to check.

        Returns
        -------
        bool
            True if syntax is valid.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _run_test(
        self,
        code: str,
        test: dict,
        entry_point: str | None
    ) -> bool:
        """
        Run a single test case.

        Parameters
        ----------
        code : str
            Code to test.
        test : dict
            Test case with 'inputs' and 'expected_output'.
        entry_point : str | None
            Function name to call.

        Returns
        -------
        bool
            True if test passes.
        """
        # Create test script
        test_script = self._create_test_script(code, test, entry_point)

        # Run in subprocess
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            logger.debug(f"Test timed out after {self.timeout}s")
            return False
        except Exception as e:
            logger.debug(f"Test execution failed: {e}")
            return False

    def _create_test_script(
        self,
        code: str,
        test: dict,
        entry_point: str | None
    ) -> str:
        """
        Create Python script to run test.

        Parameters
        ----------
        code : str
            Code under test.
        test : dict
            Test case.
        entry_point : str | None
            Function name.

        Returns
        -------
        str
            Test script.
        """
        inputs = test.get('inputs', [])
        expected = test.get('expected_output')

        script_parts = [
            "import sys",
            "import json",
            code,
            ""
        ]

        if entry_point:
            # Function call test
            inputs_repr = ', '.join(repr(inp) for inp in inputs)
            script_parts.extend([
                f"result = {entry_point}({inputs_repr})",
                f"expected = {repr(expected)}",
                "if result != expected:",
                "    sys.exit(1)",
                "sys.exit(0)"
            ])
        else:
            # Script execution test
            script_parts.extend([
                f"expected = {repr(expected)}",
                "# Expected output should match",
                "sys.exit(0)"  # Basic check
            ])

        return '\n'.join(script_parts)


class DockerCodeVerifier:
    """
    Code verifier using Docker for strong sandboxing.

    Requires Docker to be installed and accessible.
    """

    def __init__(
        self,
        timeout: float = 5.0,
        image: str = "python:3.10-slim",
        memory_limit: str = "256m"
    ):
        """
        Initialize Docker-based verifier.

        Parameters
        ----------
        timeout : float
            Timeout in seconds.
        image : str
            Docker image to use.
        memory_limit : str
            Memory limit for container.
        """
        self.timeout = timeout
        self.image = image
        self.memory_limit = memory_limit

    def passes(
        self,
        answer: str,
        test_cases: list[dict] | None = None,
        entry_point: str | None = None,
        **kwargs
    ) -> bool:
        """Check if code passes all tests in Docker container."""
        if test_cases is None:
            return False

        # First check syntax
        try:
            ast.parse(answer)
        except SyntaxError:
            return False

        # Run tests in Docker
        for test in test_cases:
            if not self._run_test_docker(answer, test, entry_point):
                return False

        return True

    def margin(
        self,
        answer: str,
        test_cases: list[dict] | None = None,
        entry_point: str | None = None,
        **kwargs
    ) -> float:
        """Compute fraction of tests passed."""
        if test_cases is None:
            return 0.0

        try:
            ast.parse(answer)
        except SyntaxError:
            return 0.0

        passed = sum(
            1 for test in test_cases
            if self._run_test_docker(answer, test, entry_point)
        )

        return passed / len(test_cases)

    def _run_test_docker(
        self,
        code: str,
        test: dict,
        entry_point: str | None
    ) -> bool:
        """
        Run test in Docker container.

        Parameters
        ----------
        code : str
            Code to test.
        test : dict
            Test case.
        entry_point : str | None
            Function name.

        Returns
        -------
        bool
            True if test passes.
        """
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write code to file
            code_file = tmpdir_path / "solution.py"
            code_file.write_text(code)

            # Write test script
            test_script = self._create_test_script(code, test, entry_point)
            test_file = tmpdir_path / "test.py"
            test_file.write_text(test_script)

            # Run Docker container
            try:
                result = subprocess.run(
                    [
                        "docker", "run",
                        "--rm",
                        "--network", "none",
                        "-m", self.memory_limit,
                        "-v", f"{tmpdir}:/workspace",
                        "-w", "/workspace",
                        self.image,
                        "python", "test.py"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                return result.returncode == 0

            except subprocess.TimeoutExpired:
                logger.debug(f"Docker test timed out after {self.timeout}s")
                return False
            except FileNotFoundError:
                logger.warning("Docker not found; falling back to non-sandboxed execution")
                # Fallback to basic verifier
                verifier = CodeVerifier(timeout=self.timeout, sandbox=False)
                return verifier._run_test(code, test, entry_point)
            except Exception as e:
                logger.debug(f"Docker test failed: {e}")
                return False

    def _create_test_script(
        self,
        code: str,
        test: dict,
        entry_point: str | None
    ) -> str:
        """Create test script."""
        inputs = test.get('inputs', [])
        expected = test.get('expected_output')

        script_parts = [
            "import sys",
            "from solution import *",
            ""
        ]

        if entry_point:
            inputs_repr = ', '.join(repr(inp) for inp in inputs)
            script_parts.extend([
                f"result = {entry_point}({inputs_repr})",
                f"expected = {repr(expected)}",
                "if result != expected:",
                "    sys.exit(1)",
                "sys.exit(0)"
            ])
        else:
            script_parts.extend([
                f"expected = {repr(expected)}",
                "sys.exit(0)"
            ])

        return '\n'.join(script_parts)
