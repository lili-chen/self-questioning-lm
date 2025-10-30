"""Integration smoke test for V-SQLM."""

import pytest
from sqlm.vsqlm.sampler import DummySolverBackend
from sqlm.vsqlm.verifiers import MathVerifier, DummyVerifier
from sqlm.vsqlm.solver import solve_task


class TestVSQLMSmoke:
    """Smoke tests for end-to-end V-SQLM pipeline."""

    def test_basic_solve_no_verify(self):
        """Test basic solve without verification."""
        config = {
            'experiment': {'name': 'test', 'seed': 42},
            'sampling': {
                'K_min': 3,
                'K_max': 5,
                'temperatures': [0.7],
                'variants': ['cot']
            },
            'stopping': {'z': 1.96, 'threshold': 0.5},
            'voting': {'weighted': False, 'calibration': {}},
            'debate': {'enabled': False}
        }

        task = {
            'task_id': 'test_1',
            'prompt': 'What is 2+2?',
            'ground_truth': '4'
        }

        solver = DummySolverBackend()
        verifier = DummyVerifier(always_pass=False)

        result = solve_task(
            task=task,
            domain='math',
            solver_backend=solver,
            verifier=verifier,
            config=config
        )

        # Should return a result
        assert 'answer' in result
        assert 'verified' in result
        assert 'method' in result
        assert 't' in result

        # Should have used between K_min and K_max samples
        assert config['sampling']['K_min'] <= result['t'] <= config['sampling']['K_max']

    def test_solve_with_verify_first(self):
        """Test solve with verify-first short-circuit."""
        config = {
            'experiment': {'name': 'test', 'seed': 42},
            'sampling': {
                'K_min': 3,
                'K_max': 10,
                'temperatures': [0.7],
                'variants': ['cot']
            },
            'stopping': {'z': 1.96, 'threshold': 0.5},
            'voting': {'weighted': False, 'calibration': {}},
            'debate': {'enabled': False}
        }

        task = {
            'task_id': 'test_2',
            'prompt': 'What is 2+2?',
            'ground_truth': '4'
        }

        solver = DummySolverBackend()
        verifier = DummyVerifier(always_pass=True, pass_rate=1.0)

        result = solve_task(
            task=task,
            domain='math',
            solver_backend=solver,
            verifier=verifier,
            config=config
        )

        # Should verify
        assert result['verified'] is True

        # Should use verify-first method
        assert 'verify-first' in result['method']

        # Should short-circuit early
        assert result['t'] <= config['sampling']['K_max']

    def test_solve_with_debate(self):
        """Test solve with debate enabled."""
        config = {
            'experiment': {'name': 'test', 'seed': 42},
            'sampling': {
                'K_min': 2,
                'K_max': 5,
                'temperatures': [0.7],
                'variants': ['cot']
            },
            'stopping': {'z': 1.96, 'threshold': 0.5},
            'voting': {'weighted': False, 'calibration': {}},
            'debate': {
                'enabled': True,
                'tau': 0.1,  # Low threshold to always trigger
                'rounds': 2,
                'tit_for_tat_level': 2,
                'early_stop': True
            }
        }

        task = {
            'task_id': 'test_3',
            'prompt': 'What is 2+2?',
            'ground_truth': '4'
        }

        solver = DummySolverBackend()
        verifier = DummyVerifier(always_pass=False)

        # Debate backends
        debater_A = DummySolverBackend(name='debater_A')
        debater_B = DummySolverBackend(name='debater_B')
        judge = DummySolverBackend(name='judge')

        result = solve_task(
            task=task,
            domain='math',
            solver_backend=solver,
            verifier=verifier,
            config=config,
            debater_A=debater_A,
            debater_B=debater_B,
            judge=judge
        )

        # Should have debate in method
        assert 'debate' in result.get('method', '')

        # Should have debate metadata
        assert 'debate_triggered' in result
        assert 'judge_meta' in result

    def test_solve_multiple_tasks(self):
        """Test solving multiple tasks in sequence."""
        config = {
            'experiment': {'name': 'test', 'seed': 42},
            'sampling': {
                'K_min': 3,
                'K_max': 5,
                'temperatures': [0.7],
                'variants': ['cot']
            },
            'stopping': {'z': 1.96, 'threshold': 0.5},
            'voting': {'weighted': False, 'calibration': {}},
            'debate': {'enabled': False}
        }

        tasks = [
            {'task_id': f'test_{i}', 'prompt': f'Problem {i}', 'ground_truth': str(i)}
            for i in range(5)
        ]

        solver = DummySolverBackend()
        verifier = DummyVerifier()

        results = []

        for task in tasks:
            result = solve_task(
                task=task,
                domain='math',
                solver_backend=solver,
                verifier=verifier,
                config=config
            )
            results.append(result)

        # Should have results for all tasks
        assert len(results) == 5

        # All should have required fields
        for result in results:
            assert 'answer' in result
            assert 'method' in result
            assert 't' in result


class TestVSQLMStoppingBehavior:
    """Test stopping behavior under different conditions."""

    def test_early_stop_high_agreement(self):
        """Test early stopping with high agreement."""
        # Configure for potential early stop
        config = {
            'experiment': {'name': 'test', 'seed': 100},  # Seed for determinism
            'sampling': {
                'K_min': 3,
                'K_max': 20,
                'temperatures': [0.0],  # Low temp for consistency
                'variants': ['cot']
            },
            'stopping': {'z': 1.96, 'threshold': 0.5},
            'voting': {'weighted': False, 'calibration': {}},
            'debate': {'enabled': False}
        }

        task = {
            'task_id': 'test_stop',
            'prompt': 'Easy problem',
            'ground_truth': '42'
        }

        solver = DummySolverBackend()
        verifier = DummyVerifier()

        result = solve_task(
            task=task,
            domain='math',
            solver_backend=solver,
            verifier=verifier,
            config=config
        )

        # With deterministic sampling (temp=0), should get high agreement
        # and potentially stop early
        # (Exact behavior depends on DummySolver, but t should be reasonable)
        assert result['t'] >= config['sampling']['K_min']
        assert result['t'] <= config['sampling']['K_max']
