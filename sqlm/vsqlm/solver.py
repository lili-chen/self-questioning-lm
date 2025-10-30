"""
Core V-SQLM solver implementation.

Implements the end-to-end V-SQLM pipeline with verify-first, adaptive-K, and debate.
"""

import logging
from typing import Dict, Any
import time

from sqlm.vsqlm.sampler import SolverBackend
from sqlm.vsqlm.canonicalize import canonicalize
from sqlm.vsqlm.aggregator import majority_vote, weighted_vote, get_leader_stats
from sqlm.vsqlm.stopping import should_stop
from sqlm.stats.entropy import compute_vote_entropy
from sqlm.debate.minidebate import should_trigger_debate, run_debate, select_debater_solutions
from sqlm.metrics.calibration import ReliabilityStore

logger = logging.getLogger(__name__)


def solve_task(
    task: Dict[str, Any],
    domain: str,
    solver_backend: SolverBackend,
    verifier: Any,
    config: Dict[str, Any],
    reliability_store: ReliabilityStore | None = None,
    debater_A: SolverBackend | None = None,
    debater_B: SolverBackend | None = None,
    judge: SolverBackend | None = None
) -> Dict[str, Any]:
    """
    Solve a single task using V-SQLM pipeline.

    Pipeline:
    1. Sample solutions with adaptive K
    2. Canonicalize into equivalence classes
    3. Verify-first short-circuit if any passes
    4. Check Wilson stopping criterion
    5. Vote (weighted or unweighted)
    6. Optionally trigger debate if high entropy
    7. Return final answer and metadata

    Parameters
    ----------
    task : dict
        Task with 'prompt' and optionally 'ground_truth', 'test_cases', etc.
    domain : str
        Domain: 'math', 'code', or 'text'.
    solver_backend : SolverBackend
        Solver for sampling solutions.
    verifier : Verifier
        Verifier for verify-first short-circuiting.
    config : dict
        Configuration with sampling, stopping, debate settings.
    reliability_store : ReliabilityStore | None
        Optional reliability store for weighted voting.
    debater_A, debater_B, judge : SolverBackend | None
        Optional debate participants.

    Returns
    -------
    dict
        Result with keys:
        - answer: final answer
        - verified: bool
        - method: string describing solution method
        - t: number of samples used
        - leader_share: vote share of winner
        - vote_entropy: entropy of vote distribution
        - tokens_prompt, tokens_gen: token counts
        - latency_ms: total latency
        - debate_rounds: number of debate rounds (if used)
        - judge_meta: judge metadata (if debate used)
    """
    start_time = time.time()

    # Extract config
    K_min = config['sampling']['K_min']
    K_max = config['sampling']['K_max']
    temperatures = config['sampling']['temperatures']
    variants = config['sampling']['variants']
    z = config['stopping']['z']
    threshold = config['stopping']['threshold']
    weighted = config['voting']['weighted']
    debate_enabled = config['debate']['enabled']
    debate_tau = config['debate'].get('tau', 0.4)

    prompt = task['prompt']
    seed = config['experiment']['seed']

    # Storage
    samples = []
    weights = []

    # Tokens and latency tracking
    tokens_prompt = 0
    tokens_gen = 0

    # Sampling loop
    chosen = None
    method = None

    for t in range(1, K_max + 1):
        # Select temperature and variant
        temp_idx = (t - 1) % len(temperatures)
        var_idx = (t - 1) % len(variants)

        temperature = temperatures[temp_idx]
        variant = variants[var_idx]

        # Sample
        sample = solver_backend.sample(
            prompt,
            seed=seed + t if seed else None,
            temperature=temperature,
            variant=variant
        )

        samples.append(sample)

        # Track tokens (if available in meta)
        if 'usage' in sample.get('meta', {}):
            usage = sample['meta']['usage']
            tokens_prompt += usage.get('prompt_tokens', 0)
            tokens_gen += usage.get('completion_tokens', 0)

        # Canonicalize
        classes = canonicalize(
            domain,
            samples,
            verifier=verifier,
            **task  # pass ground_truth, test_cases, etc.
        )

        # Verify-first short-circuit
        verified_classes = [k for k, c in classes.items() if c.get('passes', False)]

        if verified_classes:
            chosen = verified_classes[0]  # Pick first verified
            method = "verify-first"
            logger.info(f"Task {task.get('task_id', '?')}: verify-first at t={t}")
            break

        # Check stopping criterion
        leader, leader_share, total_samples = get_leader_stats(classes)

        if t >= K_min and should_stop(leader_share, t, K_min, threshold, z):
            chosen = leader
            method = "vote-wilson"
            logger.info(
                f"Task {task.get('task_id', '?')}: Wilson stop at t={t}, "
                f"leader_share={leader_share:.3f}"
            )
            break

    # Max K reached
    if chosen is None:
        leader, leader_share, total_samples = get_leader_stats(classes)
        chosen = leader
        method = "vote-maxK"
        t = K_max

    # Get final answer
    y_hat = classes[chosen]['canonical']

    # Verify final answer
    verified = verifier.passes(y_hat, **task) if verifier else False

    # Compute vote entropy
    vote_entropy = compute_vote_entropy(classes)

    # Decide on debate
    debate_triggered = False
    debate_rounds = 0
    judge_meta = {}

    any_verified = any(c.get('passes', False) for c in classes.values())

    if debate_enabled and should_trigger_debate(vote_entropy, any_verified, debate_tau):
        logger.info(
            f"Task {task.get('task_id', '?')}: triggering debate "
            f"(entropy={vote_entropy:.3f}, tau={debate_tau})"
        )

        # Select solutions for debate
        solution_A, solution_B = select_debater_solutions(classes, samples, method='top2')

        if debater_A and debater_B and judge:
            debate_result = run_debate(
                problem=prompt,
                debater_A=debater_A,
                debater_B=debater_B,
                judge=judge,
                solution_A=solution_A,
                solution_B=solution_B,
                rounds=config['debate']['rounds'],
                tit_for_tat_level=config['debate']['tit_for_tat_level'],
                early_stop=config['debate']['early_stop'],
                seed=seed + 10000 if seed else None
            )

            y_hat = debate_result['final_answer']
            verified = verifier.passes(y_hat, **task) if verifier else False
            method = method + "+debate"
            debate_triggered = True
            debate_rounds = len([t for t in debate_result['transcript'] if t['role'] != 'judge'])
            judge_meta = debate_result['judge_meta']

    # Update reliability store (for weighted voting)
    if reliability_store and weighted:
        for idx, sample in enumerate(samples):
            variant = sample.get('meta', {}).get('variant', 'unknown')
            # Check if this sample's answer matches ground truth
            sample_correct = verifier.passes(sample['answer'], **task) if verifier else False
            reliability_store.update(variant, sample_correct)

    # Compute latency
    latency_ms = (time.time() - start_time) * 1000

    # Estimate token counts if not available
    if tokens_prompt == 0:
        # Rough estimate: 1 token ≈ 4 chars
        tokens_prompt = len(prompt) // 4
        tokens_gen = sum(len(s['answer']) // 4 for s in samples)

    return {
        'task_id': task.get('task_id', '?'),
        'domain': domain,
        'answer': y_hat,
        'verified': verified,
        'method': method,
        't': t,
        'leader_share': leader_share,
        'vote_entropy': vote_entropy,
        'tokens_prompt': tokens_prompt,
        'tokens_gen': tokens_gen,
        'latency_ms': latency_ms,
        'debate_triggered': debate_triggered,
        'debate_rounds': debate_rounds,
        'judge_meta': judge_meta,
        'seed': seed,
        'n_classes': len(classes),
    }
