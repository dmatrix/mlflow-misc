"""
LLM-as-a-Judge Tutorial: Tool Selection Evaluation with MLflow - CLI Entry Point

This script provides a command-line interface for the tool selection evaluation tutorial.
It demonstrates how to use MLflow's LLM-as-a-Judge pattern to evaluate AI agent tool selection decisions.

For the class implementation, see tool_selection_judge_cls.py.

TUTORIAL WORKFLOW:
1. Setup MLflow tracing
2. Initialize agent and judge
3. Agent selects a tool for a user query
4. Judge evaluates the selection
5. Display results

Run 'uv run mlflow-tool-selection-judge --help' for usage examples.
"""

from genai.agents.tools_selection.tool_selection_judge_cls import AgentToolSelectionJudge
from genai.common.config import AgentConfig
import mlflow


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
        # Call the agent to select a tool based on the user request and available tools
        # This is the first call to LLM to select a tool based on the user task intent
        tool_selected = judge.select_tool(user_request, available_tools)
        print(f"  └─ ✓ Selected: {tool_selected}")

        # ========================================================================
        # TUTORIAL STEP 5: Judge Evaluates the Agent's Decision
        # ========================================================================
        print("\n[Step 5] Judge evaluates the selection...")
        trace_id = mlflow.get_last_active_trace_id()

        # Call the LLM judge to evaluate the trace
        # This is the second call to LLM to evaluate the tool selection
        # The judge analyzes the trace and returns feedback
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
