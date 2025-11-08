"""
Test the simplified MLflow GenAI evaluation implementation.

This script tests the inline evaluation pattern following MLflow documentation.
"""
import os
import sys

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

from genai.agents.insect_expert_openai import (
    InsectExpertOpenAIAgent,
    setup_mlflow_tracking,
)
import mlflow

# Setup MLflow
print("Setting up MLflow tracking...")
setup_mlflow_tracking("test-simplified-evaluation")
print("✓ MLflow configured\n")

# Initialize agent with evaluation enabled
print("Initializing agent with evaluation enabled...")
agent = InsectExpertOpenAIAgent(
    model="databricks-gemini-2-5-flash",
    temperature=1.0,
    enable_evaluation=True,
    judge_model="databricks-gemini-2-5-flash",
)
print("✓ Agent initialized\n")

# Test question
question = "What makes bees able to fly?"

print(f"Question: {question}\n")
print("Generating answer with inline evaluation...\n")

# Run with MLflow tracking
with mlflow.start_run():
    # Get answer (with tracing)
    answer = agent.answer_question(question)

    # Evaluate after trace is complete
    agent.evaluate_last_response()

    print("="*80)
    print("ANSWER:")
    print("="*80)
    print(answer)
    print()

    # Check evaluation scores
    if agent.last_eval_scores:
        print("="*80)
        print("EVALUATION RESULTS:")
        print("="*80)

        if "rating" in agent.last_eval_scores:
            rating = agent.last_eval_scores["rating"]
            rating_emoji = {
                "excellent": "🟢",
                "good": "🟡",
                "fair": "🟠",
                "poor": "🔴"
            }
            emoji = rating_emoji.get(rating.lower(), "⚪")
            print(f"\n{emoji} Rating: {rating.upper()}\n")

        if "rationale" in agent.last_eval_scores:
            print("Judge's Rationale:")
            print("-" * 80)
            print(agent.last_eval_scores["rationale"])
            print()

        print("="*80)
        print("✅ Evaluation completed successfully!")
        print("="*80)
    else:
        print("⚠️  No evaluation scores found")

print(f"\nView traces in MLflow UI:")
print(f"  mlflow ui --backend-store-uri http://localhost:5000")
print(f"  Then open: http://localhost:5000\n")
