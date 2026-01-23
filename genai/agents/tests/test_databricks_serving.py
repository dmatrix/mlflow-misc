import argparse
import os
from databricks.sdk import WorkspaceClient

# Databricks hosted foundation models
# https://docs.databricks.com/en/machine-learning/foundation-models/supported-models.html
SUPPORTED_MODELS = [
    "databricks-gpt-5-2",
    "databricks-gemini-2-5-flash",
    "databricks-gemini-3-flash",
    "databricks-claude-sonnet-4-5",
]

DEFAULT_PROMPT = "What is MLflow? Answer in 2-3 sentences."


def get_workspace_client():
    """Initialize Databricks workspace client."""
    databricks_token = os.environ.get("DATABRICKS_TOKEN")
    databricks_host = os.environ.get("DATABRICKS_HOST")
    return WorkspaceClient(
        profile="DEFAULT",
        host=databricks_host,
        token=databricks_token,
    )


def test_model(openai_client, model: str, prompt: str) -> None:
    """Test a single model with the given prompt."""
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        print(f"Response:\n{response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Databricks hosted foundation models"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help=f"Specific model to test. If not provided, tests all supported models: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=DEFAULT_PROMPT,
        help="Custom prompt to send to the model(s)",
    )
    parser.add_argument(
        "--list-models",
        "-l",
        action="store_true",
        help="List all supported models and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Supported Databricks hosted models:")
        for model in SUPPORTED_MODELS:
            print(f"  - {model}")
        return

    workspace_client = get_workspace_client()
    openai_client = workspace_client.serving_endpoints.get_open_ai_client()

    models_to_test = [args.model] if args.model else SUPPORTED_MODELS

    print(f"Testing {len(models_to_test)} model(s) with prompt: '{args.prompt}'")

    for model in models_to_test:
        test_model(openai_client, model, args.prompt)

    print(f"\n{'='*60}")
    print("Testing complete!")


if __name__ == "__main__":
    main()
