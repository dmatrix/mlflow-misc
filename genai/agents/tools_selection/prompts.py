"""
Tutorial: Prompts for Agent and Judge

This module contains all prompts used in the tutorial:
- Agent prompts: Instructions for the agent to perform tasks
- Judge prompts: Evaluation criteria for the LLM-as-a-Judge

KEY CONCEPT: Centralizing prompts makes them easy to find, modify, and version control.
"""


def get_tool_selection_prompt(user_request: str, available_tools: list) -> str:
    """
    TUTORIAL: Agent Prompt for Tool Selection

    This prompt instructs the agent to select the most appropriate tool
    for a given user request.

    Args:
        user_request: The user's query
        available_tools: List of available tool names

    Returns:
        Formatted prompt for tool selection

    Customization Tips:
        - Add selection criteria (e.g., "prefer faster tools", "prioritize accuracy")
        - Provide examples of good selections
        - Add constraints (e.g., "avoid tool X for query type Y")
    """
    return f"""Given the user request and available tools, select the most appropriate tool.

User Request: {user_request}
Available Tools: {available_tools}

Respond with ONLY the tool name, nothing else."""


def get_judge_instructions() -> str:
    """
    TUTORIAL: Judge Instructions for MLflow's make_judge()

    These instructions define HOW the judge evaluates agent decisions.
    The judge will use these criteria to assess whether the agent
    selected the appropriate tool.

    IMPORTANT: Must include {{ trace }} template variable for MLflow to inject trace data.

    Returns:
        String containing evaluation criteria and output format

    Customization Tips:
        - Add domain-specific criteria (e.g., "prefer official APIs over web search")
        - Add severity levels (e.g., "critical error", "minor issue", "optimal")
        - Include examples of good vs bad decisions
    """
    return """You are an expert evaluator assessing AI agent tool selection decisions.

You will receive a trace showing the agent's decision-making process:
{{ trace }}

Evaluate whether the tool selection was appropriate by considering:
- Does the selected tool match the user's intent?
- Can this tool address the task requirements?
- Are there more suitable tools the agent overlooked?

Provide your evaluation as:
- Value: "correct" if the selection was appropriate, "incorrect" otherwise
- Rationale: Detailed explanation of your assessment, including alternative tools if applicable
"""

