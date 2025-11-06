"""
Insect Expert Agent using OpenAI/Databricks Foundation Model Serving endpoints with MLflow 3.x tracing.

A self-contained agent class for answering questions about insects using
OpenAI-compatible APIs (OpenAI, Databricks Foundation Model Serving endpoints, etc.) with MLflow tracing support.
"""

import os

import mlflow
from mlflow.entities import SpanType
from databricks.sdk import WorkspaceClient
from openai import OpenAI


class InsectExpertOpenAIAgent:
    """
    A simple agent that answers questions about insects using OpenAI API.

    This demonstrates MLflow 3.x tracing with OpenAI (or Databricks Foundation Model Serving endpoints).
    """

    def __init__(
        self,
        model: str = "databricks-gpt-5",
        temperature: float = 1.0,
        api_key: str | None = None,
        use_databricks: bool = True,
        databricks_host: str | None = None,
    ):
        """
        Initialize the insect expert agent with OpenAI or Databricks.

        Args:
            model: Model to use (default: databricks-gpt-5). Can be OpenAI model or Databricks model
            temperature: LLM temperature (default: 1.0)
            api_key: OpenAI API key or Databricks token. If None, uses OPENAI_API_KEY
                     or DATABRICKS_TOKEN environment variable
            use_databricks: If True, use Databricks SDK WorkspaceClient (default: True)
            databricks_host: Databricks workspace host URL. If None, uses DATABRICKS_HOST environment variable
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("DATABRICKS_TOKEN") if use_databricks else os.environ.get("OPENAI_API_KEY")
            if not api_key:
                env_var = "DATABRICKS_TOKEN" if use_databricks else "OPENAI_API_KEY"
                raise ValueError(
                    f"API key required. Set {env_var} environment variable, "
                    "or pass api_key parameter."
                )

        # Initialize OpenAI client
        if use_databricks:
            # Get Databricks host from environment if not provided
            if databricks_host is None:
                databricks_host = os.environ.get("DATABRICKS_HOST")
                if not databricks_host:
                    raise ValueError(
                        "Databricks host required. Set DATABRICKS_HOST environment variable, "
                        "or pass databricks_host parameter."
                    )

            # Use Databricks SDK to get OpenAI-compatible client
            workspace_client = WorkspaceClient(
                profile="DEFAULT",
                host=databricks_host,
                token=api_key
            )
            self.client = workspace_client.serving_endpoints.get_open_ai_client()
            self.use_databricks = True
            self.databricks_host = databricks_host
        else:
            # Use standard OpenAI client
            self.client = OpenAI(api_key=api_key)
            self.use_databricks = False
            self.databricks_host = None

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
        Answer a question about insects using OpenAI.

        Args:
            question: The question to answer

        Returns:
            The agent's response as a string
        """
        # Call OpenAI API with manual tracing
        response = self._call_openai(question)

        answer = response.choices[0].message.content
        return answer or "I couldn't generate a response."

    @mlflow.trace(span_type=SpanType.LLM)
    def _call_openai(self, question: str):
        """
        Make a traced call to OpenAI LLM.

        Args:
            question: The question to ask

        Returns:
            OpenAI ChatCompletion response
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=5000,
        )

        return response


def setup_mlflow_tracking(experiment_name: str = "insect-expert-openai") -> str:
    """
    Set up MLflow tracking for the insect expert agent.

    Args:
        experiment_name: Name of the MLflow experiment

    Returns:
        The experiment name
    """
    # Enable OpenAI autologging if available
    try:
        mlflow.openai.autolog()
    except Exception:
        # Autolog not available, traces will still work
        pass

    # Set or create experiment
    mlflow.set_experiment(experiment_name)

    return experiment_name
