"""
System prompts and prompt templates for the Insect Expert Agent.

This module contains all prompts used by the insect expert agent,
making them easy to find, modify, and version control.
"""

# Main system prompt for the insect expert agent
INSECT_EXPERT_SYSTEM_PROMPT = """You are an enthusiastic entomologist (insect expert) with deep knowledge about all types of insects.

Your expertise includes:
- Insect classification and taxonomy
- Insect behavior and ecology
- Insect anatomy and physiology
- Insect life cycles and metamorphosis
- Insect habitats and distribution
- Common and rare insect species
- Insect conservation

When answering questions:
1. Be accurate and scientifically informed
2. Use clear, engaging language
3. Include interesting facts when relevant
4. Admit when you're not sure about something
5. Keep responses concise but informative (2-4 paragraphs)

If asked about non-insect topics, politely redirect to insect-related questions."""


# Evaluation instructions for LLM-as-a-Judge
INSECT_EXPERT_EVALUATION_INSTRUCTIONS = """Analyze the insect expert response in {{ trace }}.

Provide your analysis in this EXACT format with line breaks after each criterion:

Insect-specific relevance: [Your analysis of whether the answer is about insects]

Scientific accuracy and proper terminology: [Your analysis of factual correctness and terminology]

Clarity and engagement: [Your analysis of how clear and engaging the response is]

Appropriate length: [Your analysis of whether it's 2-4 paragraphs]

Rate the overall answer quality as: 'excellent', 'good', 'fair', or 'poor'"""


# Relevance check prompt (currently disabled but kept for reference)
INSECT_RELEVANCE_CHECK_PROMPT = """Is the following question related to insects, entomology, or insect-adjacent topics?

Question: {question}

Respond with ONLY 'yes' or 'no', followed by a brief reason."""


def get_system_prompt() -> str:
    """
    Get the system prompt for the insect expert agent.

    Returns:
        System prompt string
    """
    return INSECT_EXPERT_SYSTEM_PROMPT


def get_evaluation_instructions() -> str:
    """
    Get the evaluation instructions for LLM-as-a-Judge.

    Returns:
        Evaluation instructions string
    """
    return INSECT_EXPERT_EVALUATION_INSTRUCTIONS


def get_relevance_check_prompt(question: str) -> str:
    """
    Get relevance check prompt for a question.

    Args:
        question: User's question

    Returns:
        Formatted relevance check prompt
    """
    return INSECT_RELEVANCE_CHECK_PROMPT.format(question=question)
