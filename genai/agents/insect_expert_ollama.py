"""
Simple Insect Expert Agent using Ollama (local, free) with MLflow 3.x.

This agent answers questions about insects using Ollama and MLflow tracing.
No API key required - runs completely locally!
"""

import argparse
import subprocess

import mlflow
from mlflow.entities import SpanType
from ollama import Client


# ============================================================================
# Agent Implementation
# ============================================================================


class InsectExpertOllamaAgent:
    """
    A simple agent that answers questions about insects using Ollama.

    This demonstrates MLflow 3.x tracing with a local LLM (no API key needed).
    """

    def __init__(self, model: str = "llama3.2", temperature: float = 0.7):
        """
        Initialize the insect expert agent with Ollama.

        Args:
            model: Ollama model to use (default: llama3.2)
            temperature: LLM temperature (0.0-1.0, higher = more creative)
        """
        self.client = Client()
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
        Answer a question about insects using Ollama.

        Args:
            question: The question to answer

        Returns:
            The agent's response as a string
        """
        # Call Ollama API with manual tracing
        response = self._call_ollama(question)

        answer = response["message"]["content"]
        return answer or "I couldn't generate a response."

    @mlflow.trace(span_type=SpanType.LLM)
    def _call_ollama(self, question: str) -> dict:
        """
        Make a traced call to Ollama LLM.

        Args:
            question: The question to ask

        Returns:
            Ollama response dict
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
        )

        return response


# ============================================================================
# MLflow Integration
# ============================================================================


def setup_mlflow_tracking(experiment_name: str = "insect-expert-ollama") -> str:
    """
    Set up MLflow tracking for the insect expert agent.

    Args:
        experiment_name: Name of the MLflow experiment

    Returns:
        The experiment name
    """
    # Note: Ollama doesn't have autologging yet, but traces still work
    # Set or create experiment
    mlflow.set_experiment(experiment_name)

    return experiment_name


# ============================================================================
# Helper Functions
# ============================================================================


def check_ollama_installed() -> bool:
    """Check if Ollama is installed."""
    try:
        subprocess.run(
            ["ollama", "--version"], capture_output=True, check=True, timeout=5
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_model_available(model: str) -> bool:
    """Check if a specific Ollama model is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        return model in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def pull_model(model: str):
    """Pull an Ollama model."""
    print(f"\nDownloading {model} model...")
    print("This may take a few minutes on first run.")
    print("=" * 70)
    try:
        subprocess.run(["ollama", "pull", model], check=True)
        print(f"✓ Model {model} downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download model: {e}")
        raise


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """
    Main function to run the insect expert agent with Ollama.
    """
    parser = argparse.ArgumentParser(
        description="Insect Expert Agent (Ollama) - Ask questions about insects!"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question about insects to ask the agent",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2",
        help="Ollama model to use (default: llama3.2)",
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
        default="insect-expert-ollama",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )

    args = parser.parse_args()

    # Check if Ollama is installed
    if not check_ollama_installed():
        print("=" * 70)
        print("ERROR: Ollama is not installed.")
        print("=" * 70)
        print("\nPlease install Ollama:")
        print("  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh")
        print("  Or visit: https://ollama.com/download")
        print("\nAfter installing, the model will download automatically.")
        print("=" * 70)
        return 1

    # Check if model is available, pull if needed
    if not check_model_available(args.model):
        print(f"Model '{args.model}' not found locally.")
        try:
            pull_model(args.model)
        except Exception as e:
            print(f"\nFailed to download model: {e}")
            print("\nAvailable models:")
            print("  ollama list")
            print("\nTo manually download:")
            print(f"  ollama pull {args.model}")
            return 1

    # Setup MLflow if not disabled
    if not args.no_mlflow:
        experiment_name = setup_mlflow_tracking(args.experiment_name)
        print("=" * 70)
        print("Insect Expert Agent - Ollama (Local) + MLflow 3.x")
        print("=" * 70)
        print(f"Experiment: {experiment_name}")
        print(f"Model: {args.model}")
        print(f"Temperature: {args.temperature}")
        print(f"Provider: Ollama (local, free)")
        print("=" * 70)

    # Create agent
    agent = InsectExpertOllamaAgent(model=args.model, temperature=args.temperature)

    # Get question
    if args.question:
        question = args.question
    else:
        # Default example question
        question = "What makes bees able to fly, and how do their wings work?"

    # Print question
    print(f"\nQuestion: {question}\n")
    print("=" * 70)
    print("\nThinking...")

    # Get answer from agent (with MLflow tracing if enabled)
    if not args.no_mlflow:
        # Run with MLflow tracking - traces will be linked to this run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(
                {
                    "model": agent.model,
                    "temperature": agent.temperature,
                    "question_length": len(question),
                    "provider": "ollama",
                }
            )

            # Call agent (traces will be captured)
            try:
                answer = agent.answer_question(question)
            except Exception as e:
                print(f"\nError calling Ollama: {e}")
                print("\nMake sure Ollama is running:")
                print("  ollama serve")
                print("\nOr check if the model is available:")
                print(f"  ollama run {args.model}")
                return 1

            # Log metrics after we have the answer
            mlflow.log_metrics(
                {
                    "answer_length": len(answer),
                    "answer_words": len(answer.split()),
                    "question_words": len(question.split()),
                }
            )
    else:
        # Run without MLflow tracking
        try:
            answer = agent.answer_question(question)
        except Exception as e:
            print(f"\nError calling Ollama: {e}")
            print("\nMake sure Ollama is running:")
            print("  ollama serve")
            print("\nOr check if the model is available:")
            print(f"  ollama run {args.model}")
            return 1

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
