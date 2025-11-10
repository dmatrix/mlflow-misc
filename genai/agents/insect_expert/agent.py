"""
Insect Expert Agent implementation.

A simplified, self-contained agent for answering questions about insects.
Uses common utilities for provider management and configuration.
"""

import mlflow
import os
from typing import Optional, Dict, Any, List
from mlflow.entities import SpanType

from genai.common import get_client, AgentConfig, EvaluationConfig
from genai.agents.insect_expert.prompts import get_system_prompt
from genai.agents.insect_expert.evaluation import InsectExpertEvaluator

# Import MLflow scorers for batch evaluation
try:
    from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines
    from mlflow.genai import evaluate
    SCORERS_AVAILABLE = True
except ImportError:
    SCORERS_AVAILABLE = False


class InsectExpertAgent:
    """
    Agent specialized in answering questions about insects and entomology.

    Simple, self-contained agent with:
    - Provider management (OpenAI/Databricks)
    - MLflow tracing
    - LLM-as-a-Judge evaluation
    - Separated prompts for easy modification

    Examples:
        >>> from genai.common.config import AgentConfig, EvaluationConfig
        >>>
        >>> # Create agent configuration
        >>> config = AgentConfig(
        ...     model="databricks-gpt-5",
        ...     temperature=0.7,
        ...     provider="databricks",
        ...     enable_evaluation=True
        ... )
        >>>
        >>> # Initialize agent
        >>> agent = InsectExpertAgent(config)
        >>>
        >>> # Ask a question
        >>> response = agent.answer_question("What do bees eat?")
        >>> print(response)
        >>>
        >>> # Evaluate if evaluation is enabled
        >>> if config.enable_evaluation:
        ...     scores = agent.evaluate_last_response()
        ...     print(scores)
    """

    def __init__(
        self,
        config: AgentConfig,
        evaluation_config: Optional[EvaluationConfig] = None
    ):
        """
        Initialize the insect expert agent.

        Args:
            config: Agent configuration
            evaluation_config: Optional evaluation configuration.
                             If None and config.enable_evaluation is True,
                             creates default evaluation config.
        """
        # Store config and setup client
        self.config = config
        self.debug = config.debug

        # Initialize provider client
        provider_kwargs = config.get_provider_kwargs()
        self.client = get_client(config.provider, **provider_kwargs)

        # Set up evaluator if needed
        self.evaluator = None
        self.last_eval_scores = {}

        if config.enable_evaluation:
            if evaluation_config is None:
                # Create default evaluation config
                evaluation_config = EvaluationConfig(
                    enabled=True,
                    judge_model=getattr(config, 'judge_model', 'databricks-gemini-2-5-flash')
                )

            use_databricks = (config.provider == "databricks")
            databricks_host = config.databricks_host if use_databricks else None

            self.evaluator = InsectExpertEvaluator(
                config=evaluation_config,
                use_databricks=use_databricks,
                databricks_host=databricks_host,
                debug=config.debug
            )

            self._debug_print("Evaluation enabled with LLM-as-a-Judge")

    def _debug_print(self, message: str):
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(message)

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the insect expert.

        Returns:
            System prompt string
        """
        return get_system_prompt()

    @mlflow.trace(span_type=SpanType.AGENT)
    def answer_question(self, question: str) -> str:
        """
        Answer a question about insects.

        Args:
            question: User's question about insects

        Returns:
            Agent's response as a string
        """
        # Call LLM directly (traced via @mlflow.trace on _call_llm)
        response = self._call_llm(question)

        answer = response.choices[0].message.content or "I couldn't generate a response."

        return answer

    @mlflow.trace(span_type=SpanType.LLM)
    def _call_llm(self, question: str) -> Any:
        """
        Make a traced call to the LLM.

        This method directly makes the API call to ensure proper MLflow span capture.
        It overrides the base class method to add tracing.

        Args:
            question: The question to ask

        Returns:
            LLM completion response
        """
        # Create messages with system prompt
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": question},
        ]

        # Call LLM directly (no wrapper indirection)
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=5000,
        )

        return response

    def evaluate_last_response(
        self,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        expected_facts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the last response using LLM-as-a-Judge.

        This should be called AFTER answer_question() returns to ensure
        the trace has been finalized.

        Args:
            question: The question that was asked (optional, for batch evaluation)
            answer: The answer that was generated (optional, for batch evaluation)
            expected_facts: List of expected facts for correctness evaluation (optional)

        Returns:
            Dictionary with evaluation scores (rating, rationale, etc.)
        """
        if not self.evaluator:
            return {}

        # Get the trace from the last call
        trace_id = mlflow.get_last_active_trace_id()
        if not trace_id:
            self._debug_print("No active trace found for evaluation")
            return {}

        # Get custom judge evaluation (trace-based)
        eval_scores = self.evaluator.evaluate(trace_id)

        # NOTE: Predefined scorers (Correctness, RelevanceToQuery, Guidelines) are disabled
        # because mlflow.genai.evaluate() creates duplicate traces when called inside mlflow.start_run()
        # The custom judge above provides sufficient evaluation feedback

        # # If question and answer are provided, also run predefined scorers
        # if question and answer and SCORERS_AVAILABLE:
        #     try:
        #         batch_scores = self._evaluate_with_predefined_scorers(
        #             question, answer, expected_facts
        #         )
        #         # Merge batch scores with judge scores
        #         eval_scores.update(batch_scores)
        #     except Exception as e:
        #         self._debug_print(f"Predefined scorers error: {e}")

        # Log rating as a metric (convert categorical to numeric)
        if eval_scores and "rating" in eval_scores:
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
        self,
        question: str,
        answer: str,
        expected_facts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate using MLflow predefined scorers (Correctness, RelevanceToQuery, Guidelines).

        Args:
            question: The question that was asked
            answer: The answer that was generated
            expected_facts: List of expected facts for correctness evaluation

        Returns:
            Dictionary with scorer results
        """
        if not SCORERS_AVAILABLE:
            self._debug_print("MLflow scorers not available")
            return {}

        # Set up environment for LiteLLM to use Databricks
        use_databricks = (self.config.provider == "databricks")
        if use_databricks:
            os.environ["DATABRICKS_API_KEY"] = os.environ.get("DATABRICKS_TOKEN", "")
            os.environ["DATABRICKS_API_BASE"] = self.config.databricks_host or ""

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

        # Get judge model from evaluator config
        judge_model = self.evaluator.config.judge_model if self.evaluator else "gpt-4"

        # Define scorers with appropriate model
        if use_databricks:
            judge_model_uri = f"databricks:/{judge_model}"
        else:
            judge_model_uri = judge_model

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
        try:
            evaluate(data=eval_data, scorers=scorers)
            self._debug_print("Predefined scorers evaluation completed")
        except Exception as e:
            self._debug_print(f"Error running predefined scorers: {e}")

        # Return empty dict - scores are logged to MLflow
        return {}

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
        self._debug_print("Relevance check disabled - allowing all questions")
        return True, "Relevance check disabled - using agent's system prompt for filtering"

    def get_last_evaluation_scores(self) -> Dict[str, Any]:
        """
        Get scores from the last evaluation.

        Returns:
            Dictionary with last evaluation scores
        """
        return self.last_eval_scores
