"""
Insect Expert Agent using OpenAI/Databricks Foundation Model Serving endpoints with MLflow 3.x tracing.

A self-contained agent class for answering questions about insects using
OpenAI-compatible APIs (OpenAI, Databricks Foundation Model Serving endpoints, etc.) with MLflow tracing support.
"""

import os
from typing import Literal

import mlflow
from mlflow.entities import SpanType
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines
from mlflow.genai import evaluate
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
        enable_evaluation: bool = False,
        judge_model: str = "databricks-gemini-2-5-flash",
        debug: bool = False,
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
            enable_evaluation: If True, enable inline LLM-as-a-judge evaluation (default: False)
            judge_model: Databricks model to use for evaluation (default: databricks-gemini-2-5-flash)
            debug: If True, print debug statements to console (default: False)
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
        self.debug = debug

        # Evaluation settings
        self.enable_evaluation = enable_evaluation
        self.judge_model = judge_model
        self.last_eval_scores = {}

        # Initialize judge if evaluation is enabled
        if self.enable_evaluation:
            # Set up environment variables for LiteLLM (used by judge)
            if self.use_databricks:
                os.environ["OPENAI_API_KEY"] = os.environ.get("DATABRICKS_TOKEN", "")
                os.environ["OPENAI_API_BASE"] = f"{self.databricks_host}/serving-endpoints"

            self._init_judge()

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

    def _debug_print(self, message: str):
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(message)

    def is_insect_related(self, question: str) -> tuple[bool, str]:
        """
        Quick check if question is insect-related using a lightweight LLM call.

        NOTE: Currently disabled - returns True for all questions.
        The main agent's system prompt handles redirecting off-topic questions.

        Args:
            question: The user's question

        Returns:
            Tuple of (is_relevant: bool, reason: str)
        """
        # DISABLED: Relevance check was unreliable (returned empty responses)
        # Let all questions through - the main agent will handle redirection
        self._debug_print("[DEBUG] Relevance check disabled - allowing all questions")
        return True, "Relevance check disabled - using agent's system prompt for filtering"

    def _init_judge(self):
        """Initialize LLM judge using Databricks Foundation Model Serving endpoint for response evaluation."""
        # For Databricks, we need to use openai:/ prefix for MLflow/LiteLLM compatibility
        # This tells LiteLLM to use OpenAI-compatible format with custom base URL
        if self.use_databricks:
            # Use openai:/ prefix (note the colon) with OPENAI_API_BASE environment variable
            # MLflow expects format: openai:/model-name
            model_uri = f"openai:/{self.judge_model}"
        else:
            # For OpenAI, use the model name directly
            model_uri = self.judge_model

        self.judge = make_judge(
            name="insect_answer_quality",
            instructions=(
                "Analyze the insect expert response in {{ trace }}.\n\n"
                "Provide your analysis in this EXACT format with line breaks after each criterion:\n\n"
                "Insect-specific relevance: [Your analysis of whether the answer is about insects]\n\n"
                "Scientific accuracy and proper terminology: [Your analysis of factual correctness and terminology]\n\n"
                "Clarity and engagement: [Your analysis of how clear and engaging the response is]\n\n"
                "Appropriate length: [Your analysis of whether it's 2-4 paragraphs]\n\n"
                "Rate the overall answer quality as: 'excellent', 'good', 'fair', or 'poor'"
            ),
            feedback_value_type=Literal["excellent", "good", "fair", "poor"],
            model=model_uri,
        )

    def _evaluate_response(self) -> dict:
        """
        Evaluate response using inline LLM-as-a-judge pattern.

        This follows the MLflow recommended pattern:
        1. Get the trace from the last traced call
        2. Call the judge directly with the trace
        3. Extract rating and rationale from feedback

        Returns:
            Dictionary with 'rating' and 'rationale' keys
        """
        if not self.enable_evaluation:
            return {}

        try:
            # Get the trace from the last @mlflow.trace decorated call
            trace_id = mlflow.get_last_active_trace_id()
            if not trace_id:
                self._debug_print("[DEBUG] No active trace found")
                return {}

            trace = mlflow.get_trace(trace_id)

            # Call judge directly with the trace
            feedback = self.judge(trace=trace)

            # Extract rating and rationale
            scores = {
                "rating": feedback.value,  # Will be 'excellent', 'good', 'fair', or 'poor'
                "rationale": feedback.rationale  # Explanation from the judge
            }

            self._debug_print(f"[DEBUG] Evaluation complete - Rating: {scores['rating']}")
            return scores

        except Exception as e:
            # Log error but don't fail the main response
            import traceback
            error_msg = f"Evaluation error: {e}\n{traceback.format_exc()}"
            self._debug_print(error_msg)
            return {}

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

        answer = response.choices[0].message.content or "I couldn't generate a response."

        return answer

    def evaluate_last_response(self, question: str = None, answer: str = None, expected_facts: list[str] = None) -> dict:
        """
        Evaluate the last response after the trace is complete.

        This should be called AFTER answer_question() returns to ensure
        the trace has been finalized.

        Args:
            question: The question that was asked (optional, for batch evaluation)
            answer: The answer that was generated (optional, for batch evaluation)
            expected_facts: List of expected facts for correctness evaluation (optional)

        Returns:
            Dictionary with 'rating', 'rationale', and predefined scorer results
        """
        if not self.enable_evaluation:
            return {}

        # Get custom judge evaluation (trace-based)
        eval_scores = self._evaluate_response()

        # If question and answer are provided, also run predefined scorers
        if question and answer:
            try:
                batch_scores = self._evaluate_with_predefined_scorers(
                    question, answer, expected_facts
                )
                # Merge batch scores with judge scores
                eval_scores.update(batch_scores)
            except Exception as e:
                self._debug_print(f"[DEBUG] Predefined scorers error: {e}")

        # Log rating as a metric
        if eval_scores and "rating" in eval_scores:
            # Convert categorical rating to numeric for metrics
            rating_map = {"excellent": 1.0, "good": 0.75, "fair": 0.5, "poor": 0.25}
            numeric_rating = rating_map.get(eval_scores["rating"], 0.5)
            mlflow.log_metric("eval_rating_score", numeric_rating)

        # Log other numeric scores
        for key, value in eval_scores.items():
            if key != "rating" and isinstance(value, (int, float)):
                mlflow.log_metric(f"eval_{key}", float(value))

        self.last_eval_scores = eval_scores
        return eval_scores

    def _evaluate_with_predefined_scorers(
        self, question: str, answer: str, expected_facts: list[str] = None
    ) -> dict:
        """
        Evaluate using MLflow predefined scorers (Correctness, RelevanceToQuery, Guidelines).

        Args:
            question: The question that was asked
            answer: The answer that was generated
            expected_facts: List of expected facts for correctness evaluation

        Returns:
            Dictionary with scorer results
        """
        # Set up environment for LiteLLM to use Databricks
        # LiteLLM (used by MLflow scorers) needs these env vars
        if self.use_databricks:
            os.environ["DATABRICKS_API_KEY"] = os.environ.get("DATABRICKS_TOKEN", "")
            os.environ["DATABRICKS_API_BASE"] = self.databricks_host

        # Prepare evaluation data in MLflow format
        eval_data = [
            {
                "inputs": {"question": question},
                "outputs": answer,
                "expectations": {
                    "expected_facts": expected_facts or [
                        "The answer should be specifically about insects and scientifically accurate."
                    ]
                },
            }
        ]

        # Define scorers with Databricks model
        # IMPORTANT: Predefined scorers need explicit model configuration
        judge_model_uri = f"databricks:/{self.judge_model}"

        scorers = [
            Correctness(model=judge_model_uri),
            RelevanceToQuery(model=judge_model_uri),
            Guidelines(
                name="insect_guidelines",
                guidelines=(
                    "The answer must be specifically about insects (not other animals). "
                    "It should use proper scientific terminology and be clear and engaging. "
                    "Length should be 2-4 paragraphs."
                ),
                model=judge_model_uri,
            ),
        ]

        # Run evaluation
        # Note: Predefined scorers are logged to MLflow but not returned for display
        # They can be viewed in the MLflow UI under the trace details
        evaluate(data=eval_data, scorers=scorers)

        self._debug_print("[DEBUG] Predefined scorers evaluation completed")

        # Return empty dict - scores are logged to MLflow
        return {}

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
    Set up MLflow tracking for the insect expert agent using SQLite backend.

    Args:
        experiment_name: Name of the MLflow experiment

    Returns:
        The experiment name
    """
    # Set tracking URI to SQLite database in the project directory
    # tracking_uri = "sqlite:///mlflow.db"
    # Use localremote MLflow server
    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)

    # Enable OpenAI autologging if available
    try:
        mlflow.openai.autolog()
    except Exception:
        # Autolog not available, traces will still work
        pass

    # Set or create experiment
    mlflow.set_experiment(experiment_name)

    return experiment_name
