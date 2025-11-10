"""
Test the simplified MLflow GenAI evaluation implementation.

This script tests the inline evaluation pattern using the refactored InsectExpertAgent.
"""
import os
import sys
import mlflow

from genai.common.config import AgentConfig, EvaluationConfig
from genai.common.mlflow_config import setup_mlflow_tracking
from genai.agents.insect_expert import InsectExpertAgent

# Check for required environment variables
required_vars = ["DATABRICKS_TOKEN", "DATABRICKS_HOST"]
missing_vars = [var for var in required_vars if not os.environ.get(var)]

if missing_vars:
    print(f"\n❌ Missing required environment variables: {', '.join(missing_vars)}")
    print("\nPlease set:")
    for var in missing_vars:
        print(f"  export {var}=<your-value>")
    sys.exit(1)

print("\n" + "="*80)
print("Testing Simplified MLflow GenAI Evaluation")
print("="*80 + "\n")

# Setup MLflow
print("Setting up MLflow tracking...")
setup_mlflow_tracking("test-simplified-evaluation")
print("✓ MLflow configured\n")

# Create agent configuration
print("Creating agent configuration...")
config = AgentConfig(
    model="databricks-gemini-2-5-flash",
    temperature=1.0,
    provider="databricks",
    enable_evaluation=True,
    debug=True,
)

# Create evaluation configuration
eval_config = EvaluationConfig(
    enabled=True,
    judge_model="databricks-gemini-2-5-flash"
)

# Initialize agent
print("Initializing agent with evaluation enabled...")
agent = InsectExpertAgent(config=config, evaluation_config=eval_config)
print("✓ Agent initialized\n")

# Test question
question = "What makes bees able to fly?"

print(f"Question: {question}\n")
print("Generating answer with inline evaluation...\n")

# Run with MLflow tracking
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "model": agent.config.model,
        "temperature": agent.config.temperature,
        "provider": agent.config.provider,
    })

    # Get answer (with tracing)
    answer = agent.answer_question(question)

    # Evaluate after trace is complete
    eval_scores = agent.evaluate_last_response(question=question, answer=answer)

    # Log metrics
    mlflow.log_metrics({
        "answer_length": len(answer),
        "answer_words": len(answer.split()),
    })

    print("="*80)
    print("ANSWER:")
    print("="*80)
    print(answer)
    print()

    # Check evaluation scores
    if eval_scores:
        print("="*80)
        print("EVALUATION RESULTS:")
        print("="*80)

        if "rating" in eval_scores:
            rating = eval_scores["rating"]
            rating_emoji = {
                "excellent": "🟢",
                "good": "🟡",
                "fair": "🟠",
                "poor": "🔴"
            }
            emoji = rating_emoji.get(rating.lower(), "⚪")
            print(f"\n{emoji} Rating: {rating.upper()}\n")

        if "rationale" in eval_scores:
            print("Judge's Rationale:")
            print("-" * 80)
            print(eval_scores["rationale"])
            print()

        print("="*80)
        print("✅ Evaluation completed successfully!")
        print("="*80)
    else:
        print("⚠️  No evaluation scores found")

print(f"\nView traces in MLflow UI:")
print(f"  mlflow ui --backend-store-uri http://localhost:5000")
print(f"  Then open: http://localhost:5000\n")
