"""
Simple Insect Expert Agent using MLflow 3.x GenAI framework.

This agent answers questions about insects using OpenAI and MLflow tracing.
"""

import argparse
import os

import mlflow
from mlflow.entities import SpanType
from openai import OpenAI


# ============================================================================
# Agent Implementation
# ============================================================================


class InsectExpertAgent:
    """
    A simple agent that answers questions about insects.

    This demonstrates MLflow 3.x tracing with a straightforward Q&A agent.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize the insect expert agent.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature (0.0-1.0, higher = more creative)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it with: export OPENAI_API_KEY='your-api-key'"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

        # System prompt for the insect expert
        self.system_prompt = """You are an enthusiastic entomologist (insect expert) with deep knowledge about all types of insects.

Your expertise includes:
- Insect classification and taxonomy
- Insect behavior and ecology
- Insect anatomy and physiology
- Insect life cycles and metamorphosis
- Insect habitats and distribution
- Common and rare insect species
- Insect conservation

When answering questions:
1. Be accurate and scientifically informed
2. Use clear, engaging language
3. Include interesting facts when relevant
4. Admit when you're not sure about something
5. Keep responses concise but informative (2-4 paragraphs)

If asked about non-insect topics, politely redirect to insect-related questions."""

    @mlflow.trace(span_type=SpanType.AGENT)
    def answer_question(self, question: str) -> str:
        """
        Answer a question about insects.

        Args:
            question: The question to answer

        Returns:
            The agent's response as a string
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        answer = response.choices[0].message.content
        return answer or "I couldn't generate a response."


# ============================================================================
# MLflow Integration
# ============================================================================


def setup_mlflow_tracking(experiment_name: str = "insect-expert-agent") -> str:
    """
    Set up MLflow tracking for the insect expert agent.

    Args:
        experiment_name: Name of the MLflow experiment

    Returns:
        The experiment name
    """
    # Enable autologging for OpenAI
    mlflow.openai.autolog()

    # Set or create experiment
    mlflow.set_experiment(experiment_name)

    return experiment_name


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """
    Main function to run the insect expert agent.
    """
    parser = argparse.ArgumentParser(
        description="Insect Expert Agent - Ask questions about insects!"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question about insects to ask the agent",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation (0.0-1.0, default: 0.7)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="insect-expert-agent",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )

    args = parser.parse_args()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("=" * 70)
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("=" * 70)
        print("\nPlease set it with:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("\nOr get an API key from: https://platform.openai.com/api-keys")
        print("=" * 70)
        return 1

    # Setup MLflow if not disabled
    if not args.no_mlflow:
        experiment_name = setup_mlflow_tracking(args.experiment_name)
        print("=" * 70)
        print("Insect Expert Agent - MLflow 3.x")
        print("=" * 70)
        print(f"Experiment: {experiment_name}")
        print(f"Model: {args.model}")
        print(f"Temperature: {args.temperature}")
        print("=" * 70)

    # Create agent
    agent = InsectExpertAgent(model=args.model, temperature=args.temperature)

    # Get question
    if args.question:
        question = args.question
    else:
        # Default example question
        question = "What makes bees able to fly, and how do their wings work?"

    # Print question
    print(f"\nQuestion: {question}\n")
    print("=" * 70)

    # Get answer from agent (with MLflow tracing if enabled)
    if not args.no_mlflow:
        # Run with MLflow tracking - traces will be automatically linked
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(
                {
                    "model": agent.model,
                    "temperature": agent.temperature,
                    "question_length": len(question),
                    "provider": "openai",
                }
            )

            # Call agent (OpenAI autologging will capture LLM calls)
            answer = agent.answer_question(question)

            # Log metrics
            mlflow.log_metrics(
                {
                    "answer_length": len(answer),
                    "answer_words": len(answer.split()),
                    "question_words": len(question.split()),
                }
            )
    else:
        # Run without MLflow tracking
        answer = agent.answer_question(question)

    # Print answer
    print(f"\n{answer}\n")
    print("=" * 70)

    # Print MLflow info if enabled
    if not args.no_mlflow:
        print("\n✓ Logged to MLflow with traces")
        print("✓ View traces in MLflow UI: mlflow ui")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
