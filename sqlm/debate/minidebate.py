"""
Mini-debate module for V-SQLM.

Implements sparse, entropy-gated debate with 2 debaters, ≤2 rounds, and a judge.
"""

from typing import TypedDict
import logging

from sqlm.debate.prompts import (
    format_debate_prompt,
    format_transcript,
    parse_judge_decision
)

logger = logging.getLogger(__name__)


class DebateResult(TypedDict):
    """Result of a debate."""
    final_answer: str
    transcript: list[dict]
    judge_meta: dict


def run_debate(
    problem: str,
    debater_A: 'SolverBackend',
    debater_B: 'SolverBackend',
    judge: 'SolverBackend',
    solution_A: str,
    solution_B: str,
    rounds: int = 2,
    tit_for_tat_level: int = 2,
    early_stop: bool = True,
    seed: int | None = None
) -> DebateResult:
    """
    Run a structured debate between two debaters with a judge.

    Parameters
    ----------
    problem : str
        The problem statement.
    debater_A : SolverBackend
        Affirmative debater backend.
    debater_B : SolverBackend
        Negative debater backend.
    judge : SolverBackend
        Judge backend.
    solution_A : str
        Debater A's initial solution.
    solution_B : str
        Debater B's initial solution.
    rounds : int
        Maximum debate rounds (default: 2).
    tit_for_tat_level : int
        Tit-for-tat awareness level (0-2, default: 2).
    early_stop : bool
        Whether judge can stop early (default: True).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    DebateResult
        Final answer, transcript, and judge metadata.

    Examples
    --------
    >>> from sqlm.vsqlm.sampler import DummySolverBackend
    >>> debater_A = DummySolverBackend(name='A')
    >>> debater_B = DummySolverBackend(name='B')
    >>> judge = DummySolverBackend(name='judge')
    >>> result = run_debate(
    ...     "What is 2+2?",
    ...     debater_A, debater_B, judge,
    ...     "4", "5",
    ...     rounds=2
    ... )  # doctest: +SKIP
    """
    transcript = []
    judge_meta = {
        'early_stopped': False,
        'rounds_used': 0,
        'decision_method': None
    }

    # Track arguments for each debater
    last_argument_A = None
    last_argument_B = None

    for round_num in range(1, rounds + 1):
        # Round start
        logger.info(f"Debate round {round_num}/{rounds}")

        # Affirmative (Debater A) argues
        aff_prompt = format_debate_prompt(
            role='affirmative',
            problem=problem,
            solution=solution_A,
            round_num=round_num,
            max_rounds=rounds,
            tit_for_tat_level=tit_for_tat_level
        )

        aff_response = debater_A.sample(
            aff_prompt,
            seed=seed + round_num if seed else None,
            variant='debate'
        )

        last_argument_A = aff_response['answer']

        transcript.append({
            'role': 'affirmative',
            'content': last_argument_A,
            'round': round_num,
            'solution': solution_A
        })

        # Negative (Debater B) critiques and argues
        neg_prompt = format_debate_prompt(
            role='negative',
            problem=problem,
            solution=solution_B,
            affirmative_solution=solution_A,
            affirmative_argument=last_argument_A,
            round_num=round_num,
            max_rounds=rounds,
            tit_for_tat_level=tit_for_tat_level
        )

        neg_response = debater_B.sample(
            neg_prompt,
            seed=seed + round_num + 1000 if seed else None,
            variant='debate'
        )

        last_argument_B = neg_response['answer']

        transcript.append({
            'role': 'negative',
            'content': last_argument_B,
            'round': round_num,
            'solution': solution_B
        })

        # Judge evaluates
        transcript_str = format_transcript(transcript)

        if early_stop and round_num < rounds:
            # Discriminative mode: can judge stop early?
            judge_prompt = format_debate_prompt(
                role='judge',
                problem=problem,
                transcript=transcript_str,
                round_num=round_num,
                max_rounds=rounds
            )

            judge_response = judge.sample(
                judge_prompt,
                seed=seed + 2000 if seed else None,
                variant='judge'
            )

            decision = parse_judge_decision(judge_response['answer'])

            if decision['action'] == 'select':
                # Judge has selected an answer
                judge_meta['early_stopped'] = True
                judge_meta['rounds_used'] = round_num
                judge_meta['decision_method'] = 'discriminative'

                transcript.append({
                    'role': 'judge',
                    'content': judge_response['answer'],
                    'round': round_num,
                    'decision': decision
                })

                return DebateResult(
                    final_answer=decision['answer'],
                    transcript=transcript,
                    judge_meta=judge_meta
                )

            # Judge says continue
            logger.info(f"Judge requests continue: {decision['justification'][:100]}")

        # Continue to next round

    # Max rounds reached: extractive mode
    logger.info(f"Max rounds ({rounds}) reached, judge must select")

    transcript_str = format_transcript(transcript)

    judge_prompt = format_debate_prompt(
        role='judge',
        problem=problem,
        transcript=transcript_str,
        round_num=rounds,
        max_rounds=rounds
    )

    judge_response = judge.sample(
        judge_prompt,
        seed=seed + 3000 if seed else None,
        variant='judge'
    )

    decision = parse_judge_decision(judge_response['answer'])

    judge_meta['early_stopped'] = False
    judge_meta['rounds_used'] = rounds
    judge_meta['decision_method'] = 'extractive'

    transcript.append({
        'role': 'judge',
        'content': judge_response['answer'],
        'round': rounds,
        'decision': decision
    })

    return DebateResult(
        final_answer=decision['answer'],
        transcript=transcript,
        judge_meta=judge_meta
    )


def should_trigger_debate(
    vote_entropy: float,
    any_verified: bool,
    tau: float = 0.4
) -> bool:
    """
    Determine whether to trigger debate based on vote entropy.

    Debate is triggered when:
    1. Vote entropy exceeds threshold tau (indicating disagreement)
    2. AND no answer has been verified yet

    Parameters
    ----------
    vote_entropy : float
        Entropy of vote distribution (0 = unanimous, higher = more disagreement).
    any_verified : bool
        Whether any answer has been verified.
    tau : float
        Entropy threshold for triggering debate (default: 0.4).

    Returns
    -------
    bool
        True if debate should be triggered.

    Examples
    --------
    >>> should_trigger_debate(vote_entropy=0.6, any_verified=False, tau=0.4)
    True

    >>> should_trigger_debate(vote_entropy=0.6, any_verified=True, tau=0.4)
    False

    >>> should_trigger_debate(vote_entropy=0.2, any_verified=False, tau=0.4)
    False
    """
    if any_verified:
        # If we have verification, don't need debate
        return False

    # Trigger if entropy exceeds threshold
    return vote_entropy > tau


def select_debater_solutions(
    classes: dict,
    samples: list[dict],
    method: str = 'top2'
) -> tuple[str, str]:
    """
    Select two solutions for debate.

    Parameters
    ----------
    classes : dict
        Equivalence classes from canonicalization.
    samples : list[dict]
        Original samples.
    method : str
        Selection method:
        - 'top2': top 2 by vote count
        - 'random': random 2 from different classes

    Returns
    -------
    tuple[str, str]
        (solution_A, solution_B) for debate.

    Examples
    --------
    >>> classes = {
    ...     '42': {'members': [0, 1], 'canonical': '42'},
    ...     '43': {'members': [2], 'canonical': '43'}
    ... }
    >>> samples = [
    ...     {'answer': '42'},
    ...     {'answer': '42'},
    ...     {'answer': '43'}
    ... ]
    >>> select_debater_solutions(classes, samples, method='top2')
    ('42', '43')
    """
    if method == 'top2':
        # Sort classes by vote count
        sorted_classes = sorted(
            classes.items(),
            key=lambda x: len(x[1]['members']),
            reverse=True
        )

        if len(sorted_classes) >= 2:
            solution_A = sorted_classes[0][1]['canonical']
            solution_B = sorted_classes[1][1]['canonical']
        elif len(sorted_classes) == 1:
            # Only one class: use it for both (debate will be trivial)
            solution_A = solution_B = sorted_classes[0][1]['canonical']
        else:
            raise ValueError("No classes available for debate")

        return solution_A, solution_B

    elif method == 'random':
        import random
        class_keys = list(classes.keys())

        if len(class_keys) >= 2:
            selected = random.sample(class_keys, 2)
            return classes[selected[0]]['canonical'], classes[selected[1]]['canonical']
        elif len(class_keys) == 1:
            canonical = classes[class_keys[0]]['canonical']
            return canonical, canonical
        else:
            raise ValueError("No classes available for debate")

    else:
        raise ValueError(f"Unknown method: {method}")
