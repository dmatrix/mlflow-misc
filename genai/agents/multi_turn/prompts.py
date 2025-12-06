"""
Prompts for Multi-Turn Customer Support Evaluation Tutorial.

Demonstrates MLflow 3.7 session-level evaluation with {{ conversation }} template.
"""


def get_support_agent_system_prompt() -> str:
    """
    System prompt for customer support agent.

    Returns:
        System prompt defining agent behavior
    """
    return """You are a helpful customer support agent for TechCorp.

Your responsibilities:
- Assist customers with technical issues
- Ask clarifying questions when needed
- Provide step-by-step troubleshooting
- Remember context from earlier in the conversation
- Be polite, professional, and concise

Guidelines:
- Keep responses under 100 words
- Reference previous messages when relevant
- Guide users through solutions systematically
"""


def get_coherence_judge_instructions() -> str:
    """
    TUTORIAL KEY FEATURE: Coherence Judge with {{ conversation }} template.

    The {{ conversation }} template variable makes this judge automatically
    session-level. MLflow will:
    1. Aggregate all traces with the same session ID
    2. Format them as a conversation
    3. Pass to this judge for evaluation

    Returns:
        Judge instructions with {{ conversation }} template
    """
    return """You are evaluating the coherence of a multi-turn customer support conversation.

Conversation to evaluate:
{{ conversation }}

Evaluate whether the conversation flows logically:
- Does the agent maintain context across turns?
- Are responses relevant to previous messages?
- Does the conversation follow a logical progression?
- Are there any contradictions or confusing jumps?

Provide your evaluation as:

- Value: True if the conversation is coherent and flows naturally, False if there are significant coherence issues.

- Rationale: Explain your reasoning in 2-3 sentences. Consider:
  - Context maintenance: Agent remembers what user said earlier
  - Logical flow: Each turn builds on previous turns
  - Relevance: Responses address the actual questions/issues raised
  - Consistency: No contradictions in advice or information
"""


def get_context_retention_judge_instructions() -> str:
    """
    TUTORIAL KEY FEATURE: Context Retention Judge with {{ conversation }} template.

    Similar to coherence judge, but evaluates on a 4-level scale.
    The {{ conversation }} template makes this session-level.

    Returns:
        Judge instructions with {{ conversation }} template
    """
    return """You are evaluating context retention in a multi-turn customer support conversation.

Conversation to evaluate:
{{ conversation }}

Assess how well the agent remembers and uses information from earlier turns:

EXCELLENT:
- Perfectly recalls all relevant prior context
- Seamlessly references earlier information
- Builds on previous exchanges without repetition

GOOD:
- Recalls most relevant context
- Occasionally references earlier info
- Minor lapses but overall maintains context

FAIR:
- Some context retention issues
- Asks redundant questions
- Doesn't fully leverage earlier information

POOR:
- Frequently forgets prior context
- Treats each turn independently
- Significant repetition or contradictions

Provide your evaluation as:

- Value: Rate the conversation as excellent, good, fair, or poor.

- Rationale: Explain your rating in 2-3 sentences. Focus on:
  - Information recall: Does agent remember user's problem details?
  - Progressive assistance: Does troubleshooting build logically?
  - Avoiding repetition: No asking for info already provided
  - Continuity: Conversation feels connected, not fragmented
"""


def map_retention_to_score(retention_level: str) -> int:
    """
    Map context retention level to numeric score.

    Args:
        retention_level: One of [excellent, good, fair, poor]

    Returns:
        Numeric score (1-4)
    """
    mapping = {
        "excellent": 4,
        "good": 3,
        "fair": 2,
        "poor": 1
    }

    if retention_level not in mapping:
        raise ValueError(
            f"Unknown retention level: {retention_level}. "
            f"Expected one of: {list(mapping.keys())}"
        )

    return mapping[retention_level]
