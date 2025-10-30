"""
Logging utilities for V-SQLM experiments.

Provides JSONL logging and metrics aggregation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class JSONLLogger:
    """
    Logger that writes experiment results to JSONL files.

    Each line is a JSON object representing one task result.
    """

    def __init__(self, output_path: str | Path):
        """
        Initialize JSONL logger.

        Parameters
        ----------
        output_path : str | Path
            Path to output JSONL file.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        self.file = open(self.output_path, 'a')

        logger.info(f"Initialized JSONL logger: {self.output_path}")

    def log(self, record: Dict[str, Any]) -> None:
        """
        Log a single record.

        Parameters
        ----------
        record : dict
            Record to log. Must be JSON-serializable.
        """
        # Add timestamp if not present
        if 'timestamp' not in record:
            record['timestamp'] = datetime.now().isoformat()

        # Write to file
        json_str = json.dumps(record)
        self.file.write(json_str + '\n')
        self.file.flush()

    def close(self) -> None:
        """Close the log file."""
        self.file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class MetricsAggregator:
    """
    Aggregates metrics from experiment results.

    Computes accuracy, token usage, latency, etc.
    """

    def __init__(self):
        """Initialize metrics aggregator."""
        self.records = []

    def add(self, record: Dict[str, Any]) -> None:
        """
        Add a record to aggregate.

        Parameters
        ----------
        record : dict
            Task result record.
        """
        self.records.append(record)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute aggregate metrics.

        Returns
        -------
        dict
            Metrics including:
            - accuracy: fraction of correct answers
            - total_tasks: number of tasks
            - avg_samples: average K used
            - avg_tokens: average tokens (prompt + generated)
            - avg_latency_ms: average latency
            - verify_first_rate: fraction short-circuited by verify-first
            - debate_usage_rate: fraction using debate
            - early_stop_rate: fraction of debates stopped early
        """
        if not self.records:
            return {}

        n = len(self.records)

        # Accuracy
        accuracy = sum(1 for r in self.records if r.get('verified', False)) / n

        # Average samples
        avg_samples = sum(r.get('t', 0) for r in self.records) / n

        # Token usage
        total_tokens = sum(
            r.get('tokens_prompt', 0) + r.get('tokens_gen', 0)
            for r in self.records
        )
        avg_tokens = total_tokens / n

        # Latency
        avg_latency = sum(r.get('latency_ms', 0) for r in self.records) / n

        # Method breakdown
        verify_first = sum(
            1 for r in self.records
            if 'verify-first' in r.get('method', '')
        )
        verify_first_rate = verify_first / n

        debate_used = sum(
            1 for r in self.records
            if 'debate' in r.get('method', '')
        )
        debate_usage_rate = debate_used / n

        # Early stop rate (among debates)
        if debate_used > 0:
            early_stops = sum(
                1 for r in self.records
                if 'debate' in r.get('method', '')
                and r.get('judge_meta', {}).get('early_stopped', False)
            )
            early_stop_rate = early_stops / debate_used
        else:
            early_stop_rate = 0.0

        return {
            'accuracy': accuracy,
            'total_tasks': n,
            'avg_samples': avg_samples,
            'avg_tokens': avg_tokens,
            'total_tokens': total_tokens,
            'avg_latency_ms': avg_latency,
            'verify_first_rate': verify_first_rate,
            'debate_usage_rate': debate_usage_rate,
            'early_stop_rate': early_stop_rate,
        }

    def compute_budget_curve(
        self,
        budget_key: str = 'total_tokens'
    ) -> list[tuple[float, float]]:
        """
        Compute accuracy vs. budget curve.

        Parameters
        ----------
        budget_key : str
            Key to use as budget (e.g., 'total_tokens', 't').

        Returns
        -------
        list[tuple[float, float]]
            List of (budget, accuracy) points, sorted by budget.
        """
        if not self.records:
            return []

        # Sort records by budget
        sorted_records = sorted(
            self.records,
            key=lambda r: r.get(budget_key, 0)
        )

        # Compute cumulative accuracy
        curve = []
        correct_count = 0

        for i, record in enumerate(sorted_records):
            if record.get('verified', False):
                correct_count += 1

            budget = record.get(budget_key, 0)
            accuracy = correct_count / (i + 1)

            curve.append((budget, accuracy))

        return curve

    def save_summary(self, output_path: str | Path) -> None:
        """
        Save metrics summary to JSON file.

        Parameters
        ----------
        output_path : str | Path
            Path to output JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = self.compute_metrics()

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics summary: {output_path}")

    def save_curve(
        self,
        output_path: str | Path,
        budget_key: str = 'total_tokens'
    ) -> None:
        """
        Save budget curve to CSV file.

        Parameters
        ----------
        output_path : str | Path
            Path to output CSV file.
        budget_key : str
            Budget key.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        curve = self.compute_budget_curve(budget_key)

        with open(output_path, 'w') as f:
            f.write(f"{budget_key},accuracy\n")
            for budget, accuracy in curve:
                f.write(f"{budget},{accuracy}\n")

        logger.info(f"Saved budget curve: {output_path}")


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute hash of configuration for reproducibility tracking.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    str
        Hex digest of configuration hash.

    Examples
    --------
    >>> config = {'K_min': 3, 'K_max': 10, 'z': 1.96}
    >>> hash1 = compute_config_hash(config)
    >>> hash2 = compute_config_hash(config)
    >>> hash1 == hash2
    True
    """
    # Sort keys for determinism
    config_str = json.dumps(config, sort_keys=True)

    # Compute SHA256
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))

    return hash_obj.hexdigest()[:16]


def load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    """
    Load records from JSONL file.

    Parameters
    ----------
    path : str | Path
        Path to JSONL file.

    Returns
    -------
    list[dict]
        List of records.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def save_jsonl(records: list[Dict[str, Any]], path: str | Path) -> None:
    """
    Save records to JSONL file.

    Parameters
    ----------
    records : list[dict]
        Records to save.
    path : str | Path
        Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        for record in records:
            json_str = json.dumps(record)
            f.write(json_str + '\n')

    logger.info(f"Saved {len(records)} records to {path}")
