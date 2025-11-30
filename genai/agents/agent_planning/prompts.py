"""
Tutorial: Prompts for Agent Planning and Judge

This module contains all prompts used for multi-step agent planning evaluation:
- Agent prompts: Instructions for the agent to create plans
- Judge prompts: Evaluation criteria for the LLM-as-a-Judge
- Execution prompts: Instructions for tool selection during execution

KEY CONCEPT: Centralizing prompts makes them easy to find, modify, and version control.
"""

from typing import List, Dict, Any
import json


def get_planning_prompt(task_goal: str, available_resources: List[str]) -> str:
    """
    TUTORIAL: Agent Prompt for Multi-Step Planning

    This prompt instructs the agent to create a multi-step plan
    to accomplish a given task using available resources.

    Args:
        task_goal: The goal/task to accomplish
        available_resources: List of available tools/APIs/resources

    Returns:
        Formatted prompt for plan creation

    Customization Tips:
        - Add planning constraints (e.g., "minimize API calls", "prioritize speed")
        - Provide examples of good vs bad plans
        - Add format requirements (e.g., "number each step", "include estimated time")
        - Add domain-specific guidance (e.g., "always validate inputs before processing")
    """
    resources_formatted = ", ".join(available_resources)

    return f"""You are an expert planning agent. Create a detailed, step-by-step plan to accomplish the given task.

Task Goal: {task_goal}

Available Resources: {resources_formatted}

Requirements:
- Create a clear, sequential plan with numbered steps
- Only use resources from the available list
- Each step should be specific and actionable
- Steps should be in logical order
- The plan should be efficient and complete

Provide your plan as a numbered list. Example format:
1. [First step using specific resource]
2. [Second step using specific resource]
3. [Third step using specific resource]

Your plan:"""


def get_judge_instructions() -> str:
    """
    TUTORIAL: Judge Instructions for MLflow's make_judge()

    These instructions define HOW the judge evaluates agent planning decisions.
    The judge will use these criteria to assess the quality of multi-step plans.

    IMPORTANT: Must include {{ trace }} template variable for MLflow to inject trace data.

    Returns:
        String containing evaluation criteria and output format

    Customization Tips:
        - Add domain-specific criteria (e.g., "ensure security checks are included")
        - Add weighted criteria (e.g., "prioritize safety over efficiency")
        - Include examples of excellent vs poor plans
        - Add specific failure patterns to watch for
    """
    return """You are an expert evaluator assessing AI agent planning capabilities.

You will receive a trace showing the agent's planning process:
{{ trace }}

Evaluate the plan's quality by considering these criteria:

1. **Logical Step Ordering**: Are the steps in the correct sequence? Does each step build on previous ones?
2. **Tool Validity**: Does the plan only use valid, available tools/resources? No hallucinated tools?
3. **Task Sufficiency**: Will this plan actually accomplish the stated goal? Are all necessary steps included?
4. **Efficiency**: Is the approach optimal, or are there unnecessary steps or redundant actions?

Assign a quality rating:

- **excellent**: Clear structure, logical flow, all steps necessary and sufficient, optimal efficiency
- **good**: Well-structured with minor inefficiencies or small improvements possible
- **adequate**: Basic plan that works but lacks optimization or has minor logical issues
- **poor**: Significant gaps, illogical ordering, or uses invalid tools
- **very_poor**: Fundamentally flawed, cannot achieve goal, or relies heavily on unavailable resources

Provide your evaluation as:
- Value: One of [excellent, good, adequate, poor, very_poor]
- Rationale: Detailed explanation including:
  - Strengths of the plan
  - Weaknesses or gaps identified
  - Specific suggestions for improvement
  - Assessment of each criterion (ordering, validity, sufficiency, efficiency)
"""


def get_quality_levels() -> List[str]:
    """
    Return the quality level options used by the judge.

    These map to numeric scores:
    - excellent: 5
    - good: 4
    - adequate: 3
    - poor: 2
    - very_poor: 1

    Returns:
        List of quality level strings
    """
    return ["excellent", "good", "adequate", "poor", "very_poor"]


def get_quality_score(quality_level: str) -> int:
    """
    Map quality level to numeric score (1-5).

    Args:
        quality_level: One of [excellent, good, adequate, poor, very_poor]

    Returns:
        Numeric score from 1 to 5

    Raises:
        ValueError: If quality_level is not recognized
    """
    quality_map = {
        "excellent": 5,
        "good": 4,
        "adequate": 3,
        "poor": 2,
        "very_poor": 1
    }

    if quality_level not in quality_map:
        raise ValueError(
            f"Unknown quality level: {quality_level}. "
            f"Expected one of: {list(quality_map.keys())}"
        )

    return quality_map[quality_level]


def get_execution_prompt(
    step_description: str,
    task_goal: str,
    available_tools: List[str],
    context: Dict[str, Any]
) -> str:
    """
    TUTORIAL: Execution Prompt for Tool Selection

    This prompt instructs the agent to select and call the appropriate tool
    for executing a specific step of the plan.

    Args:
        step_description: Description of the current step to execute
        task_goal: Overall task goal for context
        available_tools: List of available tool names
        context: Results from previous steps

    Returns:
        Formatted prompt for tool selection

    Customization Tips:
        - Add tool usage guidelines (e.g., "prefer batch operations")
        - Include error handling instructions
        - Add parameter extraction hints
    """
    # Format context for display
    if context:
        context_str = "Previous Step Results:\n"
        for step_key, step_data in context.items():
            if isinstance(step_data, dict) and "result" in step_data:
                context_str += f"- {step_key}: {json.dumps(step_data['result'], indent=2)[:200]}...\n"
    else:
        context_str = "No previous step results yet."

    tools_formatted = ", ".join(available_tools)

    return f"""You are executing a step in a multi-step plan to accomplish a task.

Overall Task Goal: {task_goal}

Current Step to Execute: {step_description}

{context_str}

Available Tools: {tools_formatted}

Instructions:
1. Determine which tool is most appropriate for this step
2. Extract the necessary parameters (you may use data from previous step results)
3. Call the tool using function calling

Important:
- Only use tools from the available list
- Use previous step results when needed (e.g., use a booking_id from a previous booking)
- Be specific with parameters

Select and call the appropriate tool now."""


def get_execution_judge_instructions() -> str:
    """
    TUTORIAL: Judge Instructions for Plan Execution Evaluation

    These instructions define how the judge evaluates the actual execution
    of a plan, including tool selection and parameter quality.

    Returns:
        Judge instructions for execution evaluation
    """
    return """You are evaluating the execution of a multi-step plan.

You will receive a trace showing how the agent executed the plan:
{{ trace }}

Evaluate the execution quality on these dimensions:

1. **Step Completion**: Were all steps executed successfully? Any failures?
2. **Tool Selection**: Did the agent select appropriate tools for each step?
3. **Parameter Quality**: Were parameters correctly extracted and passed to tools?
4. **Context Usage**: Did the agent properly use results from previous steps?
5. **Error Handling**: Were errors handled appropriately?

Assign a quality rating:

- **excellent**: All steps executed flawlessly with optimal tool selection and parameter usage
- **good**: Mostly successful with minor issues in tool selection or parameters
- **adequate**: Execution completed but with some inefficiencies or suboptimal choices
- **poor**: Significant execution failures or inappropriate tool usage
- **very_poor**: Execution fundamentally failed or used invalid tools

Provide your evaluation as:
- Value: One of [excellent, good, adequate, poor, very_poor]
- Rationale: Detailed explanation including:
  - Which steps succeeded/failed
  - Tool selection quality for each step
  - Parameter extraction accuracy
  - Context utilization effectiveness
  - Overall execution assessment
"""
