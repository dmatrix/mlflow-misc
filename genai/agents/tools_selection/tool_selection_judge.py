"""
LLM-as-a-Judge Tutorial: Tool Selection Evaluation with MLflow

This tutorial demonstrates how to use MLflow's LLM-as-a-Judge pattern to evaluate AI agent decisions.

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


def main():
    """
    TUTORIAL: Complete LLM-as-a-Judge Example

    This function demonstrates the complete workflow:
    1. Setup MLflow tracing
    2. Initialize agent and judge
    3. Agent performs an action (tool selection)
    4. Judge evaluates the agent's decision
    5. Display results
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge Tutorial: Tool Selection Evaluation with MLflow",
        epilog="""
Tutorial Examples:

  # Basic usage with Databricks
  export DATABRICKS_TOKEN='your-token'
  export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'
  uv run mlflow-tool-selection-judge

  # Use OpenAI instead
  export OPENAI_API_KEY='sk-...'
  uv run mlflow-tool-selection-judge --provider openai

  # Custom query
  uv run mlflow-tool-selection-judge --query "Send email to John about the meeting"

  # Use different model for judging
  uv run mlflow-tool-selection-judge --judge-model databricks-claude-sonnet-4-5

After running, view traces in MLflow UI:
  mlflow ui
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "databricks"],
        default="databricks",
        help="LLM provider (default: databricks)"
    )
    parser.add_argument(
        "--model",
        help="Model identifier (default: databricks-gpt-5 for databricks, gpt-4o-mini for openai)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature 0.0-2.0 (default: 1.0)"
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="tool-selection-judge",
        help="MLflow experiment name (default: tool-selection-judge)"
    )
    parser.add_argument(
        "--judge-model",
        help="Judge model for evaluation (default: same as main model)"
    )
    parser.add_argument(
        "--query",
        default="What's the weather like in San Francisco?",
        help="User query for tool selection (default: 'What's the weather like in San Francisco?')"
    )

    args = parser.parse_args()

    # Set model default based on provider if not specified
    if args.model is None:
        if args.provider == "databricks":
            args.model = "databricks-gpt-5"
        else:
            args.model = "gpt-4o-mini"

    # ========================================================================
    # TUTORIAL STEP 1: Setup MLflow Tracing
    # ========================================================================
    print("\n" + "=" * 70)
    print("TUTORIAL: LLM-as-a-Judge with MLflow")
    print("=" * 70)

    from genai.common.mlflow_config import setup_mlflow_tracking
    setup_mlflow_tracking(
        experiment_name=args.mlflow_experiment,
        enable_autolog=True
    )
    print("\n[Step 1] MLflow tracing enabled")
    print(f"  └─ Experiment: {args.mlflow_experiment}")
    print(f"  └─ View traces: mlflow ui")

    # ========================================================================
    # TUTORIAL STEP 2: Initialize Agent and Judge
    # ========================================================================
    # For Databricks: DATABRICKS_TOKEN and DATABRICKS_HOST must be set in environment
    # For OpenAI: OPENAI_API_KEY must be set in environment
    config = AgentConfig(
        model=args.model,
        provider=args.provider,
        temperature=args.temperature
    )

    judge_model = args.judge_model or "databricks-gemini-2-5-flash"
    print("\n[Step 2] Initializing Agent and Judge")
    print(f"  └─ Provider: {config.provider}")
    print(f"  └─ Agent Model: {config.model}")
    print(f"  └─ Judge Model: {judge_model}")
    print(f"  └─ Temperature: {config.temperature}")

    judge = AgentToolSelectionJudge(config, judge_model=judge_model)

    # ========================================================================
    # TUTORIAL STEP 3: Define Test Scenario
    # ========================================================================
    user_request = args.query
    available_tools = ["get_weather_api", "search_web", "get_calendar", "send_email"]

    print("\n[Step 3] Test Scenario")
    print(f"  └─ User Query: {user_request}")
    print(f"  └─ Available Tools: {available_tools}")

    # ========================================================================
    # TUTORIAL STEP 4: Agent Performs Action (with tracing)
    # ========================================================================
    print("\n[Step 4] Agent selects a tool...")
    try:
        tool_selected = judge.select_tool(user_request, available_tools)
        print(f"  └─ ✓ Selected: {tool_selected}")

        # ========================================================================
        # TUTORIAL STEP 5: Judge Evaluates the Agent's Decision
        # ========================================================================
        print("\n[Step 5] Judge evaluates the selection...")
        trace_id = mlflow.get_last_active_trace_id()
        result = judge.evaluate(trace_id)

        # ========================================================================
        # TUTORIAL STEP 6: Display Results
        # ========================================================================
        print("\n[Step 6] Evaluation Results")
        print("=" * 70)
        print(f"Decision: {'✓ CORRECT' if result['is_correct'] else '✗ INCORRECT'}")
        print("\nReasoning:")
        print(f"{result['reasoning']}")
        print("=" * 70)

        print("\n✓ Tutorial complete! View detailed traces in MLflow UI:")
        print("  mlflow ui\n")

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()