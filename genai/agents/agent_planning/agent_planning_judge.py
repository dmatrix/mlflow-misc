"""
LLM-as-a-Judge Tutorial: Multi-Step Planning Evaluation with MLflow - CLI Entry Point

This script provides a command-line interface for the multi-step planning evaluation tutorial.
It demonstrates how to use MLflow's LLM-as-a-Judge pattern to evaluate AI agent planning decisions.

For the class implementation, see agent_planning_judge_cls.py.

TUTORIAL WORKFLOW:
1. Setup MLflow tracing
2. Initialize agent and judge
3. Agent creates a multi-step plan
4. Judge evaluates the plan's quality
5. Execute plan with actual tools
6. Display results with score and detailed reasoning

Run 'uv run mlflow-agent-planning-judge --help' for usage examples.
"""

from genai.agents.agent_planning.agent_planning_judge_cls import AgentPlanningJudge
from genai.common.config import AgentConfig
import mlflow


def main():
    """
    TUTORIAL: Complete LLM-as-a-Judge Example for Planning

    This function demonstrates the complete workflow:
    1. Setup MLflow tracing
    2. Initialize agent and judge
    3. Agent creates a multi-step plan
    4. Judge evaluates the plan's quality
    5. Display results with score and detailed reasoning
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge Tutorial: Multi-Step Planning Evaluation with MLflow",
        epilog="""
Tutorial Examples:

  # Basic usage with Databricks
  export DATABRICKS_TOKEN='your-token'
  export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'
  uv run mlflow-agent-planning-judge

  # Use OpenAI instead
  export OPENAI_API_KEY='sk-...'
  uv run mlflow-agent-planning-judge --provider openai

  # Custom task and resources
  uv run mlflow-agent-planning-judge \\
    --task-goal "Send weekly report email to team" \\
    --available-resources "email_api,calendar_api,database_api"

  # Use different model for judging
  uv run mlflow-agent-planning-judge --judge-model databricks-claude-sonnet-4-5

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
        default="agent-planning-judge",
        help="MLflow experiment name (default: agent-planning-judge)"
    )
    parser.add_argument(
        "--judge-model",
        help="Judge model for evaluation (default: same as main model)"
    )
    parser.add_argument(
        "--task-goal",
        default="Book a flight from NYC to SF for next Tuesday and add to calendar",
        help="Task goal for the agent to plan (default: flight booking task)"
    )
    parser.add_argument(
        "--available-resources",
        default="flight_search_api,booking_api,calendar_api,hotel_search_api,email_api",
        help="Comma-separated list of available resources (default: flight booking resources)"
    )

    args = parser.parse_args()

    # Set model default based on provider if not specified
    if args.model is None:
        if args.provider == "databricks":
            args.model = "databricks-gpt-5"
        else:
            args.model = "gpt-4o-mini"

    # Parse available resources
    available_resources = [r.strip() for r in args.available_resources.split(",")]

    # ========================================================================
    # TUTORIAL STEP 1: Setup MLflow Tracing
    # ========================================================================
    print("\n" + "=" * 70)
    print("TUTORIAL: LLM-as-a-Judge for Multi-Step Planning with MLflow")
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

    judge = AgentPlanningJudge(config, judge_model=judge_model)

    # ========================================================================
    # TUTORIAL STEP 3: Define Test Scenario
    # ========================================================================
    print("\n[Step 3] Planning Scenario")
    print(f"  └─ Task Goal: {args.task_goal}")
    print(f"  └─ Available Resources: {available_resources}")

    # ========================================================================
    # TUTORIAL STEP 4: Agent Creates Plan (with tracing) using LLM
    # ========================================================================
    print("\n[Step 4] Agent creates a multi-step plan using LLM...")
    try:
        plan = judge.create_plan(args.task_goal, available_resources)
        print("  └─ ✓ Plan created:\n")
        # Indent the plan for better readability
        for line in plan.split('\n'):
            if line.strip():
                print(f"      {line}")

        # ========================================================================
        # TUTORIAL STEP 5: LLM Judge Evaluates the Plan
        # ========================================================================
        print("\n[Step 5] The LLM Judge evaluates the plan quality...")
        trace_id = mlflow.get_last_active_trace_id()
        result = judge.evaluate(trace_id)

        # ========================================================================
        # TUTORIAL STEP 6: Display Results
        # ========================================================================
        print("\n[Step 6] Evaluation Results")
        print("=" * 70)
        print(f"Quality: {result['quality'].upper()} (Score: {result['score']}/5)")
        print("\nDetailed Assessment:")
        print(f"{result['reasoning']}")
        print("=" * 70)

        # ========================================================================
        # STEP 7: Execute Plan with Tools
        # ========================================================================
        print("\n[Step 7] Executing plan with actual tools...")
        print("=" * 70)

        execution_result = judge.execute_plan_with_tools(plan, args.task_goal)

        print("\n  ✓ Execution Complete!")
        print(f"  └─ Total Steps: {execution_result['total_steps']}")
        print(f"  └─ Successful: {execution_result['successful_steps']}/{execution_result['total_steps']}")

        print("\n  Step-by-Step Results:")
        for step_result in execution_result['step_results']:
            step_num = step_result['step_number']
            tool = step_result.get('tool_used', 'No tool')
            success = '✓' if step_result.get('success') else '✗'

            print(f"  {success} Step {step_num}: {tool}")
            if step_result.get('result'):
                result_preview = str(step_result['result'])[:100]
                print(f"     Result: {result_preview}...")

        print("=" * 70)

        print("\n✓ Tutorial complete! View detailed traces in MLflow UI:")
        print("  mlflow ui\n")

    except Exception as e:
        print(f"\nError during planning evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
