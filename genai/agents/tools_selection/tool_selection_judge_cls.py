"""
LLM-as-a-Judge Tutorial: Tool Selection Evaluation with MLflow

This module contains the AgentToolSelectionJudge class for evaluating AI agent tool selection decisions.

TUTORIAL GOALS:
1. Use MLflow tracing to capture agent actions
2. Create a judge using mlflow.genai.judges.make_judge()
3. Evaluate agent decisions using the judge
4. Integrate with MLflow experiments for reproducibility

SCENARIO:
An AI agent selects a tool to answer a user query. The judge evaluates whether
the agent chose the appropriate tool.

EVALUATION CRITERIA:
- Does the selected tool match the user's intent?
- Can this tool address the task requirements?
- Are there more suitable tools available?

Based on: https://medium.com/@juanc.olamendy/using-llm-as-a-judge-to-evaluate-agent-outputs-a-comprehensive-tutorial-00b6f1f356cc
And adapted to use the common config and provider classes from the genai package and the prompts module from the tools_selection package.
"""

from genai.common import get_client
from genai.common.config import AgentConfig
from genai.agents.tools_selection.prompts import get_judge_instructions, get_tool_selection_prompt
import mlflow
from mlflow.entities import SpanType
from mlflow.genai.judges import make_judge
from typing import Dict, Any, List
from typing_extensions import Literal
import os


class AgentToolSelectionJudge:
    """
    Tutorial: LLM-as-a-Judge for Tool Selection Evaluation.

    This class demonstrates the complete LLM-as-a-Judge pattern:
    1. Agent performs an action (select_tool) - traced with MLflow
    2. Judge evaluates the action (evaluate) - uses make_judge()

    The judge is a specialized LLM that assesses whether the agent made
    the right decision based on predefined criteria.
    """

    def __init__(self, config: AgentConfig, judge_model: str = None):
        """
        Initialize the agent and judge.

        Args:
            config: Configuration for the agent model
            judge_model: Optional separate model for judging (defaults to agent model)
        """
        # Initialize the agent's LLM client
        provider_kwargs = config.get_provider_kwargs()
        self.client = get_client(config.provider, **provider_kwargs)
        self.config = config
        self.judge_model = judge_model or config.model

        # Initialize the MLflow judge
        self._init_judge()

    def _init_judge(self):
        """
        TUTORIAL STEP 1: Create an MLflow Judge

        The judge is created using mlflow.genai.judges.make_judge() which:
        - Takes evaluation instructions (criteria)
        - Uses an LLM to perform the evaluation
        - Returns structured feedback (value + rationale)
        """
        # Set up environment for Databricks (needed by LiteLLM)
        if self.config.provider == "databricks":
            os.environ["OPENAI_API_KEY"] = os.environ.get("DATABRICKS_TOKEN", "")
            os.environ["OPENAI_API_BASE"] = f"{self.config.databricks_host}/serving-endpoints"
            model_uri = f"openai:/{self.judge_model}"
        else:
            model_uri = self.judge_model

        # Create the judge with:
        # - name: identifier for the judge
        # - instructions: evaluation criteria (from prompts.py)
        # - feedback_value_type: possible evaluation outcomes
        # - model: LLM to use for judging
        self.judge = make_judge(
            name="tool_selection_quality",
            instructions=get_judge_instructions(),
            feedback_value_type=Literal["correct", "incorrect"],
            model=model_uri
        )

    @mlflow.trace(span_type=SpanType.AGENT, name="select_tool")
    def select_tool(self, user_request: str, available_tools: List[str]) -> str:
        """
        TUTORIAL STEP 2: Agent Action with MLflow Tracing

        The @mlflow.trace decorator automatically captures:
        - Input parameters (user_request, available_tools)
        - Output (selected tool)
        - Execution time and metadata

        This trace is what the judge will evaluate.

        Args:
            user_request: The user's query
            available_tools: List of available tool names

        Returns:
            Selected tool name
        """
        # Get the tool selection prompt
        prompt = get_tool_selection_prompt(user_request, available_tools)

        # Call the LLM to select a tool
        api_params = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50
        }

        # Add temperature if supported
        # OpenAI models support 0.0 for deterministic selection
        # Some Databricks models only support default (1.0)
        if self.config.provider == "openai":
            api_params["temperature"] = 0.0

        # Call the LLM to select a tool based on the user request and available tools
        # The response is the tool selected by the LLM
        # The tool selected is the tool that the LLM selected as the most appropriate tool for the user request

        response = self._call_llm(**api_params)
        tool_selected = response.choices[0].message.content.strip()

        # Note: No need to log_param here - the trace automatically captures inputs/outputs
        return tool_selected

    def evaluate(self, trace_id: str) -> Dict[str, Any]:
        """
        TUTORIAL STEP 3: Evaluate with the Judge

        The judge evaluates the trace and returns structured feedback:
        - feedback.value: "correct" or "incorrect"
        - feedback.rationale: Detailed explanation of the evaluation

        Args:
            trace_id: MLflow trace ID to evaluate

        Returns:
            Dictionary with 'is_correct' and 'reasoning' keys

        Example:
            >>> trace_id = mlflow.get_last_active_trace_id()
            >>> result = judge.evaluate(trace_id)
            >>> print(result['is_correct'])  # True/False
            >>> print(result['reasoning'])   # Explanation
        """
        # Fetch the trace from MLflow
        trace = mlflow.get_trace(trace_id)

        # Call the judge to evaluate the trace
        # The judge analyzes the trace and returns feedback
        feedback = self.judge(trace=trace)

        # Return structured result for easy consumption
        return {
            "is_correct": feedback.value == "correct",
            "reasoning": feedback.rationale
        }

    @mlflow.trace(span_type=SpanType.LLM, name="llm_call")
    def _call_llm(self, **api_params):
        """Call LLM with MLflow tracing."""
        return self.client.chat.completions.create(**api_params)
