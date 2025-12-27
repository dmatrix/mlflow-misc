"""
Multi-Turn Conversation Evaluation with DeepEval - CLI Entry Point

Demonstrates MLflow 3.8's DeepEval integration for session-level evaluation:
- Session tracking with mlflow.update_current_trace()
- DeepEval scorers via mlflow.genai.deepeval integration
- Multi-turn conversation evaluation with industry-standard metrics

Run: uv run mlflow-deepeval-support --help
"""

from genai.agents.multi_turn.customer_support_agent_simple import (
    CustomerSupportAgentSimple,
)
from genai.agents.multi_turn.scenarios import get_all_scenarios
from genai.common.config import AgentConfig
from genai.common.mlflow_config import setup_mlflow_tracking
import mlflow
from mlflow.genai.scorers.deepeval import ConversationCompleteness, KnowledgeRetention, TopicAdherence
import argparse
import os


def main():
    """
    TUTORIAL: MLflow 3.8 DeepEval Integration for Multi-Turn Conversations

    Workflow:
    1. Setup MLflow tracing
    2. Configure DeepEval environment
    3. Initialize agent for conversations
    4. Create DeepEval scorers via MLflow integration
    5. Run customer support conversations
    6. Evaluate sessions with DeepEval metrics
    """
    parser = argparse.ArgumentParser(
        description="MLflow 3.8 DeepEval Integration Tutorial",
        epilog="""
Tutorial Examples:

  # Run with Databricks
  export DATABRICKS_TOKEN='your-token'
  export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'
  uv run mlflow-deepeval-support

  # Run with OpenAI
  export OPENAI_API_KEY='sk-...'
  uv run mlflow-deepeval-support --provider openai

  # Use different judge model
  uv run mlflow-deepeval-support --judge-model databricks-claude-sonnet-4-5

  # Run specific scenario only
  uv run mlflow-deepeval-support --scenario printer

After running:
  mlflow ui
  # Navigate to experiment, explore traces and DeepEval evaluation results
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--provider",
        choices=["openai", "databricks"],
        default="databricks",
        help="LLM provider (default: databricks)",
    )
    parser.add_argument(
        "--model",
        help="Model identifier (default: databricks-gpt-5-2 for databricks, gpt-4o-mini for openai)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature 0.0-2.0 (default: 1.0)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="customer-support-deepeval",
        help="MLflow experiment name (default: customer-support-deepeval)",
    )
    parser.add_argument(
        "--judge-model",
        help="Judge model for DeepEval evaluation (default: databricks-gemini-2-5-flash)",
    )
    parser.add_argument(
        "--scenario",
        choices=["account", "printer", "all"],
        default="account",
        help="Which scenario to run (default: printer)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output"
    )

    args = parser.parse_args()

    # Set model defaults
    if args.model is None:
        args.model = (
            "databricks-gpt-5-2"
            if args.provider == "databricks"
            else "gpt-4o-mini"
        )

    judge_model = args.judge_model or "databricks-gemini-2-5-flash"

    # ========================================================================
    # STEP 1: Setup MLflow Tracing
    # ========================================================================
    print("\n" + "=" * 70)
    print("TUTORIAL: MLflow DeepEval Integration - Multi-Turn Conversations")
    print("=" * 70)

    setup_mlflow_tracking(
        experiment_name=args.mlflow_experiment, enable_autolog=True
    )

    print("\n[Step 1] MLflow tracing enabled")
    print(f"  └─ Experiment: {args.mlflow_experiment}")
    print("  └─ View traces: mlflow ui")

    # ========================================================================
    # STEP 2: Configure DeepEval Environment
    # ========================================================================
    print("\n[Step 2] Configuring DeepEval environment")
    print(f"  └─ Provider: {args.provider}")
    print(f"  └─ Judge Model: {judge_model}")

    # Configure environment for DeepEval to use Databricks/OpenAI endpoints
    if args.provider == "databricks":
        databricks_host = os.environ.get("DATABRICKS_HOST", "")
        if databricks_host:
            os.environ["OPENAI_API_KEY"] = os.environ.get(
                "DATABRICKS_TOKEN", ""
            )
            os.environ["OPENAI_API_BASE"] = (
                f"{databricks_host}/serving-endpoints"
            )
            print(f"  └─ DeepEval using Databricks endpoint: {databricks_host}")
        judge_model_uri = f"openai:/{judge_model}"
    else:
        judge_model_uri = judge_model

    # ========================================================================
    # STEP 3: Initialize Customer Support Agent
    # ========================================================================
    print("\n[Step 3] Initializing Customer Support Agent")
    print(f"  └─ Provider: {args.provider}")
    print(f"  └─ Agent Model: {args.model}")
    print(f"  └─ Temperature: {args.temperature}")

    config = AgentConfig(
        model=args.model,
        provider=args.provider,
        temperature=args.temperature,
        mlflow_experiment=args.mlflow_experiment,
    )

    agent = CustomerSupportAgentSimple(config)

    # ========================================================================
    # STEP 4: Create DeepEval Scorers via MLflow Integration
    # ========================================================================
    print("\n[Step 4] Creating DeepEval Scorers")

    try:
        # DeepEval Conversation Completeness Scorer
        completeness_scorer = ConversationCompleteness(
            model=judge_model_uri,
            include_reason=True
        )
        knowledge_retention_scorer = KnowledgeRetention(
            model=judge_model_uri,
            include_reason=True
        )
        topic_adherence_scorer = TopicAdherence(
            model=judge_model_uri,
            include_reason=True,
            relevant_topics=["customer support", "technical help", "printer_problems", "account access"]
        )

        print("  ✓ Conversation Completeness Scorer initialized")
        print("  ✓ Knowledge Retention Scorer initialized")
        print("  ✓ Topic Adherence Scorer initialized")
    except AttributeError as e:
        print(
            "\n  ✗ Error: DeepEval scorers not available in mlflow.genai.deepeval"
        )
        print("  Error details: {e}")
        print(
            "\n  Note: MLflow 3.8+ required for DeepEval integration. Available scorers:"
        )
        return 1

    # ========================================================================
    # STEP 5: Run Conversation Scenarios
    # ========================================================================
    print("\n[Step 5] Running Customer Support Conversations")

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
        with mlflow.start_run(run_name=f"{scenario['name']}") as run:
            # Run conversation
            agent.run_conversation(
                messages=scenario["messages"], session_id=scenario["session_id"]
            )

            # ====================================================================
            # STEP 6: Evaluate Session with DeepEval Scorers
            # ====================================================================
            print(
                f"\n[Step 6] Evaluating session '{scenario['session_id']}' with DeepEval..."
            )

            try:
                # Search for traces belonging to this session
                experiment = mlflow.get_experiment_by_name(args.mlflow_experiment)

                session_traces = mlflow.search_traces(
                    locations=[experiment.experiment_id],
                    filter_string=f"metadata.`mlflow.trace.session` = '{scenario['session_id']}'",
                )

                # Evaluate using mlflow.genai.evaluate() with DeepEval scorers
                results = mlflow.genai.evaluate(
                    data=session_traces, scorers=[completeness_scorer, knowledge_retention_scorer, topic_adherence_scorer]
                )
                print(f"Completeness metrics ✅: {results.metrics}")
                print(f"Completeness score   📊: {results.metrics.get('ConversationCompleteness/mean')}")
                print("--------------------------------")   
                print(f"Knowledge Retention metrics ✅: {results.metrics}")
                print(f"Knowledge Retention score   📊: {results.metrics.get('KnowledgeRetention/mean')}")
                print("--------------------------------")   
                print(f"Topic Adherence metrics ✅: {results.metrics}")
                print(f"Topic Adherence score   📊: {results.metrics.get('TopicAdherence/mean')}")
                print("--------------------------------")   
            except Exception as e:
                print(f"\n✗ DeepEval evaluation failed: {e}")
                print(f"Error type: {type(e).__name__}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                print(
                    "\nEnsure MLflow >= 3.8.0 and deepeval are installed for DeepEval integration.\n"
                )

    return 0


if __name__ == "__main__":
    exit(main())
