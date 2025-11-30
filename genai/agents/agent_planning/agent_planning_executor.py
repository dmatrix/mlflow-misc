"""
Multi-Agent Planning Executor with Tool Calling.

Extends the planning judge to actually EXECUTE plans using real (simulated) tools.
Demonstrates the complete workflow: Plan → Execute → Evaluate.
"""

from genai.common import get_client
from genai.common.config import AgentConfig
from genai.agents.agent_planning.prompts import get_execution_prompt
from genai.agents.agent_planning.tools import get_tool, get_tool_schemas
import mlflow
from mlflow.entities import SpanType
from typing import Dict, Any, List
import json
import os


class AgentPlanningExecutor:
    """
    Execute multi-step plans with actual tool calls.

    This class demonstrates how an agent can:
    1. Parse a plan into individual steps
    2. For each step, use LLM function calling to select the right tool
    3. Execute tools and collect results
    4. Pass results between steps (context management)
    5. Trace everything with MLflow
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize executor.

        Args:
            config: Agent configuration (model, provider, etc.)
        """
        provider_kwargs = config.get_provider_kwargs()
        self.client = get_client(config.provider, **provider_kwargs)
        self.config = config
        self.execution_context = {}  # Stores results from previous steps

    @mlflow.trace(span_type=SpanType.AGENT, name="execute_plan")
    def execute_plan(self, plan: str, task_goal: str) -> Dict[str, Any]:
        """
        Execute a complete multi-step plan.

        Args:
            plan: Multi-step plan as text (numbered list)
            task_goal: Original task goal for context

        Returns:
            Execution results including all step outputs and summary
        """
        # Parse plan into individual steps
        steps = self._parse_plan(plan)

        print(f"\n  Executing {len(steps)} steps...")

        results = []

        for i, step in enumerate(steps):
            print(f"  └─ Step {i+1}/{len(steps)}: {step[:60]}{'...' if len(step) > 60 else ''}")

            step_result = self._execute_step(
                step_number=i + 1,
                step_description=step,
                task_goal=task_goal
            )
            results.append(step_result)

            # Add to context for next steps
            self.execution_context[f"step_{i+1}"] = step_result

            if step_result.get("tool_used"):
                print(f"     ✓ Used {step_result['tool_used']}")

        successful = sum(1 for r in results if r.get("success", False))

        return {
            "total_steps": len(steps),
            "successful_steps": successful,
            "step_results": results,
            "final_context": self.execution_context
        }

    @mlflow.trace(span_type=SpanType.AGENT, name="execute_step")
    def _execute_step(self, step_number: int, step_description: str, task_goal: str) -> Dict[str, Any]:
        """
        Execute a single step using LLM function calling.

        The LLM determines:
        1. Which tool to use for this step
        2. What parameters to pass (may use context from previous steps)

        Args:
            step_number: Step index
            step_description: What this step should do
            task_goal: Overall task goal for context

        Returns:
            Step execution result with tool used and output
        """
        # Get tool schemas for LLM function calling
        tool_schemas = get_tool_schemas()

        # Create prompt for LLM
        prompt = get_execution_prompt(
            step_description=step_description,
            task_goal=task_goal,
            available_tools=list(tool_schemas.keys()),
            context=self.execution_context
        )

        # Prepare API parameters
        api_params = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"type": "function", "function": schema} for schema in tool_schemas.values()],
            "tool_choice": "auto"
        }

        # Add temperature if supported
        if self.config.provider == "openai":
            api_params["temperature"] = 0.0  # Deterministic for execution

        try:
            # Call LLM with function calling capability
            response = self.client.chat.completions.create(**api_params)

            # Check if LLM made a tool call
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_params = json.loads(tool_call.function.arguments)

                # Execute the tool
                tool = get_tool(tool_name)
                result = tool.execute(**tool_params)

                return {
                    "step_number": step_number,
                    "description": step_description,
                    "tool_used": tool_name,
                    "parameters": tool_params,
                    "success": result.success,
                    "result": result.data,
                    "message": result.message,
                    "execution_time_ms": result.execution_time_ms
                }
            else:
                # No tool call - LLM decided this step doesn't need a tool
                # (e.g., informational or commentary step)
                return {
                    "step_number": step_number,
                    "description": step_description,
                    "tool_used": None,
                    "success": True,
                    "result": "No tool execution needed",
                    "message": "Informational step"
                }

        except Exception as e:
            # Handle execution errors gracefully
            return {
                "step_number": step_number,
                "description": step_description,
                "tool_used": None,
                "success": False,
                "result": None,
                "message": f"Execution error: {str(e)}"
            }

    def _parse_plan(self, plan: str) -> List[str]:
        """
        Parse a numbered plan into individual steps.

        Handles various formats:
        - "1. Do something"
        - "1) Do something"
        - "- Do something"
        - "Step 1: Do something"

        Args:
            plan: Plan text with numbered/bulleted steps

        Returns:
            List of step descriptions
        """
        lines = plan.strip().split('\n')
        steps = []

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Match various numbering formats
            # Patterns: "1.", "1)", "Step 1:", "-", "*"
            import re

            # Try to extract step content
            # Remove common prefixes
            patterns = [
                r'^\d+[\.\)]\s*',  # "1. " or "1) "
                r'^Step\s+\d+:\s*',  # "Step 1: "
                r'^[-*]\s*',  # "- " or "* "
            ]

            step = line
            for pattern in patterns:
                step = re.sub(pattern, '', step, count=1)

            # Only add if there's actual content after removing prefix
            if step and len(step) > 3:  # At least a few characters
                steps.append(step.strip())

        return steps
