"""
Evaluation module for Insect Expert Agent.

Simple LLM-as-a-Judge evaluation using MLflow's make_judge.
"""

import os
import mlflow
from typing import Dict, Any, Optional
from typing_extensions import Literal
from mlflow.genai.judges import make_judge

from genai.common.config import EvaluationConfig
from genai.agents.insect_expert.prompts import get_evaluation_instructions


class InsectExpertEvaluator:
    """
    LLM-as-a-Judge evaluator for insect expert responses.

    Evaluates responses based on:
    - Insect-specific relevance
    - Scientific accuracy and terminology
    - Clarity and engagement
    - Appropriate length (2-4 paragraphs)

    Examples:
        >>> from genai.common.config import EvaluationConfig
        >>>
        >>> config = EvaluationConfig(
        ...     enabled=True,
        ...     judge_model="databricks-gemini-2-5-flash"
        ... )
        >>>
        >>> evaluator = InsectExpertEvaluator(
        ...     config=config,
        ...     use_databricks=True,
        ...     databricks_host="https://your-workspace.cloud.databricks.com"
        ... )
        >>>
        >>> scores = evaluator.evaluate(trace_id="abc123")
        >>> print(scores)
        {'rating': 'good', 'rationale': '...'}
    """

    def __init__(
        self,
        config: EvaluationConfig,
        use_databricks: bool = True,
        databricks_host: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize the evaluator.

        Args:
            config: Evaluation configuration
            use_databricks: Whether to use Databricks endpoints
            databricks_host: Databricks workspace host URL
            debug: Enable debug printing
        """
        self.config = config
        self.use_databricks = use_databricks
        self.databricks_host = databricks_host
        self.debug = debug

        # Set up environment for LiteLLM (used by judge)
        if self.use_databricks:
            os.environ["OPENAI_API_KEY"] = os.environ.get("DATABRICKS_TOKEN", "")
            os.environ["OPENAI_API_BASE"] = f"{self.databricks_host}/serving-endpoints"

        # Initialize judge
        self._init_judge()

    def _debug_print(self, message: str):
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(message)

    def _init_judge(self):
        """Initialize LLM judge using MLflow's make_judge."""
        # For Databricks, use openai:/ prefix for MLflow/LiteLLM compatibility
        if self.use_databricks:
            model_uri = f"openai:/{self.config.judge_model}"
        else:
            model_uri = self.config.judge_model

        self.judge = make_judge(
            name="insect_answer_quality",
            instructions=get_evaluation_instructions(),
            feedback_value_type=Literal["excellent", "good", "fair", "poor"],
            model=model_uri,
        )

    def evaluate(self, trace_id: str) -> Dict[str, Any]:
        """
        Evaluate a response using the judge.

        Args:
            trace_id: MLflow trace ID to evaluate

        Returns:
            Dictionary with 'rating' and 'rationale' keys
        """
        if not self.config.enabled:
            return {}

        try:
            # Get the trace
            trace = mlflow.get_trace(trace_id)

            # Call judge with the trace
            feedback = self.judge(trace=trace)

            # Extract rating and rationale
            scores = {
                "rating": feedback.value,  # 'excellent', 'good', 'fair', or 'poor'
                "rationale": feedback.rationale
            }

            self._debug_print(f"Evaluation complete - Rating: {scores['rating']}")
            return scores

        except Exception as e:
            # Log error but don't fail
            import traceback
            error_msg = f"Evaluation error: {e}\n{traceback.format_exc()}"
            self._debug_print(error_msg)
            return {}
