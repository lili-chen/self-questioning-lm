"""
Debate prompts for V-SQLM sparse debate.

Provides prompt templates for affirmative debater, negative debater, and judge.
"""

# Affirmative debater prompt
AFFIRMATIVE_PROMPT = """You are the affirmative debater. Your task is to argue for your solution to the problem.

Problem: {problem}

Your proposed solution: {solution}

Provide concise, checkable reasoning to support your answer. Focus on:
1. Why your solution is correct
2. Key evidence or steps in your reasoning
3. Potential objections and how you address them

Agreement with the opponent is not required—your objective is to present the strongest case for correctness.

Be specific and evidence-based. Limit your argument to 3-4 key points.

Your argument:"""

# Negative debater prompt
NEGATIVE_PROMPT = """You are the negative debater. Your task is to critique the affirmative's solution and present your own.

Problem: {problem}

Affirmative's solution: {affirmative_solution}
Affirmative's argument: {affirmative_argument}

Your proposed solution: {solution}

Provide a specific critique of errors or gaps in the affirmative's reasoning, then present your own solution with supporting evidence.

Focus on:
1. Specific errors in the affirmative's reasoning
2. Why your solution is correct
3. Key differences between the approaches

Be evidence-based and concise. Limit your response to 3-4 key points.

Your response:"""

# Judge prompt (discriminative mode - early stop)
JUDGE_DISCRIMINATIVE_PROMPT = """You are the judge in a debate. Your task is to determine if a correct solution has already appeared.

Problem: {problem}

Debate transcript:
{transcript}

After reviewing the debate, decide:
1. Has a correct solution already been presented? If YES, identify it and provide a one-sentence justification referencing specific evidence from the debate.
2. If NO correct solution has appeared yet, would one more round be helpful? If so, say "CONTINUE" and briefly explain what remains unresolved.

If you've reached the maximum rounds ({max_rounds}), you MUST select the best answer from the transcript even if uncertain.

Your decision:"""

# Judge prompt (extractive mode - at round cap)
JUDGE_EXTRACTIVE_PROMPT = """You are the judge in a debate. The debate has reached the maximum number of rounds.

Problem: {problem}

Debate transcript:
{transcript}

Your task is to extract the best answer from the transcript above. Consider:
1. Which solution has the strongest supporting evidence?
2. Which reasoning is most sound?
3. Which approach addresses potential objections best?

Provide:
1. The selected answer
2. A brief (2-3 sentence) justification referencing specific evidence from the debate

Your decision:"""

# Tit-for-tat instruction (added to negative debater for level >= 2)
TIT_FOR_TAT_INSTRUCTION = """
Note: The affirmative debater has seen your previous arguments. Anticipate that they may have updated their position or addressed your critiques.
"""

# Meta prompt for role clarification
META_PROMPT = """Role: {role}

You are participating in a structured debate to determine the correct answer to a problem. The debate format is:
- Round 1: Affirmative presents their solution and reasoning
- Round 1: Negative critiques and presents their solution
- Round 2 (if needed): Affirmative responds to critique
- Round 2 (if needed): Negative responds
- Judge: Selects the correct answer or requests another round

Guidelines for {role}:
{role_guidelines}

Maintain professional objectivity. The goal is truth-seeking, not winning.
"""

# Role-specific guidelines
ROLE_GUIDELINES = {
    'affirmative': """
- Present your solution clearly with supporting reasoning
- Address potential objections proactively
- Focus on evidence and logical steps
- Be concise (3-4 key points)
""",
    'negative': """
- Identify specific errors in opponent's reasoning
- Present your own solution with evidence
- Explain key differences in approaches
- Be specific and constructive
""",
    'judge': """
- Evaluate solutions based on evidence and logic
- Determine if a correct answer has appeared
- Use early stopping if confident
- At round cap, extract the best answer from transcript
- Provide clear justification for your decision
"""
}


def format_debate_prompt(
    role: str,
    problem: str,
    solution: str | None = None,
    affirmative_solution: str | None = None,
    affirmative_argument: str | None = None,
    transcript: str | None = None,
    round_num: int | None = None,
    max_rounds: int = 2,
    tit_for_tat_level: int = 0
) -> str:
    """
    Format debate prompt for a specific role.

    Parameters
    ----------
    role : str
        Role: 'affirmative', 'negative', or 'judge'.
    problem : str
        The problem statement.
    solution : str | None
        The debater's proposed solution.
    affirmative_solution : str | None
        The affirmative's solution (for negative debater).
    affirmative_argument : str | None
        The affirmative's argument (for negative debater).
    transcript : str | None
        Full debate transcript (for judge).
    round_num : int | None
        Current round number.
    max_rounds : int
        Maximum debate rounds.
    tit_for_tat_level : int
        Tit-for-tat awareness level (0=none, 1=basic, 2=full).

    Returns
    -------
    str
        Formatted prompt.
    """
    if role == 'affirmative':
        prompt = AFFIRMATIVE_PROMPT.format(
            problem=problem,
            solution=solution or "[Your solution here]"
        )

    elif role == 'negative':
        prompt = NEGATIVE_PROMPT.format(
            problem=problem,
            affirmative_solution=affirmative_solution or "[Affirmative's solution]",
            affirmative_argument=affirmative_argument or "[Affirmative's argument]",
            solution=solution or "[Your solution here]"
        )

        # Add tit-for-tat instruction if level >= 2
        if tit_for_tat_level >= 2 and round_num and round_num > 1:
            prompt += TIT_FOR_TAT_INSTRUCTION

    elif role == 'judge':
        # Choose discriminative or extractive mode based on round
        if round_num and round_num >= max_rounds:
            # Extractive mode at cap
            prompt = JUDGE_EXTRACTIVE_PROMPT.format(
                problem=problem,
                transcript=transcript or "[Empty transcript]"
            )
        else:
            # Discriminative mode with early stop option
            prompt = JUDGE_DISCRIMINATIVE_PROMPT.format(
                problem=problem,
                transcript=transcript or "[Empty transcript]",
                max_rounds=max_rounds
            )

    else:
        raise ValueError(f"Unknown role: {role}")

    return prompt


def format_transcript(debate_history: list[dict]) -> str:
    """
    Format debate history into readable transcript.

    Parameters
    ----------
    debate_history : list[dict]
        List of debate turns with 'role', 'content', 'round' fields.

    Returns
    -------
    str
        Formatted transcript.
    """
    lines = []

    for turn in debate_history:
        role = turn['role'].upper()
        content = turn['content']
        round_num = turn.get('round', '?')

        lines.append(f"[Round {round_num}] {role}:")
        lines.append(content)
        lines.append("")  # Blank line

    return '\n'.join(lines)


def parse_judge_decision(judge_output: str) -> dict:
    """
    Parse judge's output to extract decision.

    Parameters
    ----------
    judge_output : str
        Raw judge output.

    Returns
    -------
    dict
        Parsed decision with keys:
        - action: 'select' or 'continue'
        - answer: selected answer (if action='select')
        - justification: reasoning
    """
    output_lower = judge_output.lower()

    # Check for continue signal
    if 'continue' in output_lower and 'continue' in judge_output.upper():
        return {
            'action': 'continue',
            'answer': None,
            'justification': judge_output
        }

    # Otherwise, assume selection
    # Try to extract answer (heuristic: look for "Answer:" or similar)
    import re

    answer_match = re.search(
        r'(?:answer|solution|select):\s*(.+?)(?:\n|$)',
        judge_output,
        re.IGNORECASE
    )

    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Fallback: use first line as answer
        answer = judge_output.split('\n')[0].strip()

    return {
        'action': 'select',
        'answer': answer,
        'justification': judge_output
    }
