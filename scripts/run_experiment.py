#!/usr/bin/env python3
"""
V-SQLM Experiment Runner.

Runs V-SQLM experiments based on YAML configuration files.
"""

import argparse
import logging
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlm.vsqlm.sampler import create_backend
from sqlm.vsqlm.verifiers import (
    MathVerifier, CodeVerifier, TextVerifier, DummyVerifier
)
from sqlm.vsqlm.solver import solve_task
from sqlm.metrics.logging import JSONLLogger, MetricsAggregator, compute_config_hash
from sqlm.metrics.calibration import ReliabilityStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_verifier(config: dict):
    """Create verifier from config."""
    if not config['verifier']['enabled']:
        return DummyVerifier(always_pass=False, pass_rate=0.0)

    verifier_type = config['verifier']['type']

    if verifier_type == 'math':
        return MathVerifier(tolerance=config['verifier'].get('tolerance', 1e-6))
    elif verifier_type == 'code':
        return CodeVerifier(
            timeout=config['verifier'].get('timeout', 5.0),
            sandbox=config['verifier'].get('sandbox', True)
        )
    elif verifier_type == 'text':
        return TextVerifier(case_sensitive=config['verifier'].get('case_sensitive', False))
    elif verifier_type == 'dummy':
        return DummyVerifier(
            always_pass=config['verifier'].get('always_pass', False),
            pass_rate=config['verifier'].get('pass_rate', 0.0)
        )
    else:
        raise ValueError(f"Unknown verifier type: {verifier_type}")


def load_dataset(config: dict) -> list[dict]:
    """
    Load dataset from config.

    For now, creates synthetic data. In practice, would load from
    GSM8K, MBPP, etc.
    """
    dataset_name = config['dataset']['name']
    max_samples = config['dataset'].get('max_samples')

    logger.info(f"Loading dataset: {dataset_name}")

    # Synthetic data for testing
    tasks = []

    for i in range(max_samples or 10):
        tasks.append({
            'task_id': f"{dataset_name}_{i}",
            'prompt': f"What is 2 + 2? (task {i})",
            'ground_truth': "4",
        })

    logger.info(f"Loaded {len(tasks)} tasks")

    return tasks


def main():
    parser = argparse.ArgumentParser(description='Run V-SQLM experiment')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--override',
        type=str,
        nargs='*',
        help='Override config values (e.g., experiment.name=test)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.override:
        for override in args.override:
            key, value = override.split('=')
            keys = key.split('.')

            # Navigate to nested dict
            d = config
            for k in keys[:-1]:
                d = d[k]

            # Set value (try to infer type)
            try:
                d[keys[-1]] = yaml.safe_load(value)
            except:
                d[keys[-1]] = value

    # Setup
    exp_name = config['experiment']['name']
    domain = config['experiment']['domain']
    output_dir = Path(config['logging']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Domain: {domain}")
    logger.info(f"Config hash: {compute_config_hash(config)}")

    # Create backends
    logger.info("Creating solver backend...")
    solver_backend = create_backend(**config['solver'])

    # Create verifier
    logger.info("Creating verifier...")
    verifier = create_verifier(config)

    # Create reliability store
    reliability_store = None
    if config['voting']['weighted']:
        logger.info("Creating reliability store for weighted voting...")
        reliability_store = ReliabilityStore(
            alpha_prior=config['voting']['calibration']['alpha_prior'],
            beta_prior=config['voting']['calibration']['beta_prior']
        )

    # Create debate backends
    debater_A = debater_B = judge = None
    if config['debate']['enabled']:
        logger.info("Creating debate backends...")
        debater_A = create_backend(**config['debate']['debater_A'])
        debater_B = create_backend(**config['debate']['debater_B'])
        judge = create_backend(**config['debate']['judge'])

    # Load dataset
    tasks = load_dataset(config)

    # Setup logging
    jsonl_path = output_dir / 'results.jsonl'
    logger.info(f"Logging to: {jsonl_path}")

    jsonl_logger = JSONLLogger(jsonl_path)
    metrics_agg = MetricsAggregator()

    # Run experiment
    logger.info(f"Solving {len(tasks)} tasks...")

    for i, task in enumerate(tasks):
        logger.info(f"Task {i+1}/{len(tasks)}: {task['task_id']}")

        try:
            result = solve_task(
                task=task,
                domain=domain,
                solver_backend=solver_backend,
                verifier=verifier,
                config=config,
                reliability_store=reliability_store,
                debater_A=debater_A,
                debater_B=debater_B,
                judge=judge
            )

            # Add config hash
            result['config_hash'] = compute_config_hash(config)

            # Log
            jsonl_logger.log(result)
            metrics_agg.add(result)

            logger.info(
                f"  Answer: {result['answer']}, "
                f"Verified: {result['verified']}, "
                f"Method: {result['method']}, "
                f"t={result['t']}"
            )

        except Exception as e:
            logger.error(f"Error solving task {task['task_id']}: {e}", exc_info=True)

    # Close logger
    jsonl_logger.close()

    # Compute and save metrics
    logger.info("Computing metrics...")
    metrics = metrics_agg.compute_metrics()

    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")

    metrics_agg.save_summary(output_dir / 'metrics.json')
    metrics_agg.save_curve(output_dir / 'budget_curve.csv')

    logger.info(f"Experiment complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
