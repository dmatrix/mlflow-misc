"""
Multi-Turn Conversation Evaluation Tutorial - CLI Entry Point

Demonstrates MLflow 3.7's session-level evaluation features:
- Session tracking with mlflow.update_current_trace()
- Session-level judges using {{ conversation }} template
- Multi-turn conversation evaluation

Run: uv run mlflow-multi-turn-support --help
"""

from genai.agents.multi_turn.customer_support_agent_cls import CustomerSupportAgent
from genai.agents.multi_turn.scenarios import get_all_scenarios
from genai.common.config import AgentConfig
from genai.common.mlflow_config import setup_mlflow_tracking
import argparse


def main():
    """
    TUTORIAL: MLflow 3.7 Multi-Turn Conversation Evaluation

    Workflow:
    1. Setup MLflow tracing
    2. Initialize agent with session-level judges
    3. Run customer support conversations
    4. Evaluate sessions with {{ conversation }} template judges
    """
    parser = argparse.ArgumentParser(
        description="MLflow 3.7 Multi-Turn Conversation Evaluation Tutorial",
        epilog="""
Tutorial Examples:

  # Run with Databricks
  export DATABRICKS_TOKEN='your-token'
  export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'
  uv run mlflow-multi-turn-support

  # Run with OpenAI
  export OPENAI_API_KEY='sk-...'
  uv run mlflow-multi-turn-support --provider openai

  # Use different judge model
  uv run mlflow-multi-turn-support --judge-model databricks-claude-sonnet-4-5

  # Run specific scenario only
  uv run mlflow-multi-turn-support --scenario printer

After running:
  mlflow ui
  # Navigate to experiment, explore traces with session metadata
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
        help="Model identifier (default: databricks-gpt-5-2 for databricks, gpt-4o-mini for openai)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature 0.0-2.0 (default: 0.7)"
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="multi-turn-support",
        help="MLflow experiment name (default: multi-turn-support)"
    )
    parser.add_argument(
        "--judge-model",
        help="Judge model for evaluation (default: databricks-gemini-2-5-flash)"
    )
    parser.add_argument(
        "--scenario",
        choices=["printer", "account", "all"],
        default="all",
        help="Which scenario to run (default: all)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    args = parser.parse_args()

    # Set model defaults
    if args.model is None:
        args.model = "databricks-gpt-5-2" if args.provider == "databricks" else "gpt-4o-mini"

    judge_model = args.judge_model or "databricks-gemini-2-5-flash"

    # ========================================================================
    # STEP 1: Setup MLflow Tracing
    # ========================================================================
    print("\n" + "="*70)
    print("TUTORIAL: MLflow 3.7 Multi-Turn Conversation Evaluation")
    print("="*70)

    setup_mlflow_tracking(
        experiment_name=args.mlflow_experiment,
        enable_autolog=True
    )

    print("\n[Step 1] MLflow tracing enabled")
    print(f"  └─ Experiment: {args.mlflow_experiment}")
    print("   └─ View traces: mlflow ui")

    # ========================================================================
    # STEP 2: Initialize Agent with Session-Level Judges
    # ========================================================================
    config = AgentConfig(
        model=args.model,
        provider=args.provider,
        temperature=args.temperature,
        mlflow_experiment=args.mlflow_experiment
    )

    print("\n[Step 2] Initializing Customer Support Agent and Judges")
    print(f"  └─ Provider: {config.provider}")
    print(f"  └─ Agent Model: {config.model}")
    print(f"  └─ Judge Model: {judge_model}")
    print(f"  └─ Temperature: {config.temperature}")

    agent = CustomerSupportAgent(config, judge_model=judge_model, debug=args.debug)

    # ========================================================================
    # STEP 3: Run Conversation Scenarios
    # ========================================================================
    print("\n[Step 3] Running Customer Support Conversations")

    scenarios = get_all_scenarios()
    if args.scenario != "all":
        scenario_map = {"printer": 0, "account": 1}
        scenarios = [scenarios[scenario_map[args.scenario]]]

    results = []

    for scenario in scenarios:
        print(f"\n{'─'*70}")
        print(f"Scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Session ID: {scenario['session_id']}")
        print(f"{'─'*70}")

        # Start MLflow run for this conversation
        import mlflow
        with mlflow.start_run(run_name=f"{scenario['name']}") as run:
            # Run conversation
            conv_result = agent.run_conversation(
                messages=scenario['messages'],
                session_id=scenario['session_id']
            )

            # ========================================================================
            # STEP 4: Evaluate Session with Session-Level Judges
            # ========================================================================
            print(f"\n[Step 4] Evaluating session '{scenario['session_id']}'...")

            try:
                eval_result = agent.evaluate_session(scenario['session_id'], run.info.run_id)

                # ========================================================================
                # STEP 5: Display Results
                # ========================================================================
                print(f"\n{'='*70}")
                print(f"Evaluation Results: {scenario['name']}")
                print(f"{'='*70}")

                # Coherence results
                coherence_symbol = "✓" if eval_result['coherence']['passed'] else "✗"
                coherence_label = "PASS" if eval_result['coherence']['passed'] else "FAIL"
                print(f"\n📊 Coherence: {coherence_symbol} {coherence_label}")
                print(f"   Value: {eval_result['coherence']['feedback_value']}")
                print(f"   Rationale: {eval_result['coherence']['rationale']}")

                # Context retention results
                retention_value = str(eval_result['context_retention']['feedback_value']).upper()
                retention_score = eval_result['context_retention']['score']
                print(f"\n🧠 Context Retention: {retention_value}")
                print(f"   Score: {retention_score}/4")
                print(f"   Rationale: {eval_result['context_retention']['rationale']}")

                print(f"\n{'='*70}\n")

                results.append({
                    "scenario": scenario,
                    "conversation": conv_result,
                    "evaluation": eval_result
                })

            except Exception as e:
                print(f"\n✗ Evaluation failed: {e}")
                print(f"Error details: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print("\nEnsure MLflow >= 3.7.0 for {{ conversation }} template support.\n")

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("Tutorial Complete!")
    print(f"{'='*70}")
    print(f"\n✓ Ran {len(results)} conversation scenario(s)")
    print("✓ Each conversation evaluated with 2 session-level judges")
    print("\nKey MLflow 3.7 Features Demonstrated:")
    print("  1. Session tracking: mlflow.update_current_trace(metadata={{'mlflow.trace.session': session_id}})")
    print("  2. Session-level judges: {{{{ conversation }}}} template variable")
    print("  3. Multi-turn evaluation: mlflow.search_traces() with session filter")
    print("\nView detailed traces:")
    print("  mlflow ui")
    print(f"  → Open '{args.mlflow_experiment}' experiment")
    print("  → Explore traces with session metadata\n")

    return 0


if __name__ == "__main__":
    main()
