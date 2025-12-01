"""
LLM-as-a-Judge Tutorial: Multi-Step Planning Evaluation with MLflow

This module contains the AgentPlanningJudge class for evaluating AI agent planning decisions.

TUTORIAL GOALS:
1. Use MLflow tracing to capture agent planning actions
2. Create a judge using mlflow.genai.judges.make_judge()
3. Evaluate multi-step plans using the judge
4. Integrate with MLflow experiments for reproducibility

SCENARIO:
An AI agent creates a multi-step plan to accomplish a task using available resources.
The judge evaluates whether the plan is logical, complete, efficient, and uses valid tools.

EVALUATION CRITERIA:
- Logical step ordering: Are steps in the correct sequence?
- Tool validity: Are only valid, available tools used?
- Task sufficiency: Will the plan accomplish the goal?
- Efficiency: Is the approach optimal?

Based on the tool_selection_judge.py pattern and adapted for planning evaluation.
"""

from genai.common import get_client
from genai.common.config import AgentConfig
from genai.agents.agent_planning.prompts import (
    get_judge_instructions,
    get_planning_prompt,
    get_quality_score
)
import mlflow
from mlflow.entities import SpanType
from mlflow.genai.judges import make_judge
from typing import Dict, Any, List
from typing_extensions import Literal
import os


class AgentPlanningJudge:
    """
    Tutorial: LLM-as-a-Judge for Multi-Step Planning Evaluation.

    This class demonstrates the complete LLM-as-a-Judge pattern:
    1. Agent creates a plan (create_plan) - traced with MLflow
    2. Judge evaluates the plan quality (evaluate) - uses make_judge()

    The judge is a specialized LLM that assesses whether the agent created
    a high-quality plan based on predefined criteria.
    """

    def __init__(self, config: AgentConfig, judge_model: str = None):
        """
        Initialize the agent and judge.

        Args:
            config: Configuration for the agent model
            judge_model: Optional separate model for judging (defaults to agent model)
        """
        # Initialize the agent's LLM client
        provider_kwargs = config.get_provider_kwargs()
        self.client = get_client(config.provider, **provider_kwargs)
        self.config = config
        self.judge_model = judge_model or config.model

        # Initialize the MLflow judge
        self._init_judge()

    def _init_judge(self):
        """
        TUTORIAL STEP 1: Create an MLflow Judge

        The judge is created using mlflow.genai.judges.make_judge() which:
        - Takes evaluation instructions (criteria)
        - Uses an LLM to perform the evaluation
        - Returns structured feedback (quality level + rationale)
        """
        # Set up environment for Databricks (needed by LiteLLM)
        if self.config.provider == "databricks":
            os.environ["OPENAI_API_KEY"] = os.environ.get("DATABRICKS_TOKEN", "")
            os.environ["OPENAI_API_BASE"] = f"{self.config.databricks_host}/serving-endpoints"
            model_uri = f"openai:/{self.judge_model}"
        else:
            model_uri = self.judge_model

        # Create the judge with:
        # - name: identifier for the judge
        # - instructions: evaluation criteria (from prompts.py)
        # - feedback_value_type: possible evaluation outcomes (quality levels)
        # - model: LLM to use for judging
        self.judge = make_judge(
            name="planning_quality",
            instructions=get_judge_instructions(),
            feedback_value_type=Literal["excellent", "good", "adequate", "poor", "very_poor"],
            model=model_uri
        )

    @mlflow.trace(span_type=SpanType.AGENT, name="create_plan")
    def create_plan(self, task_goal: str, available_resources: List[str]) -> str:
        """
        TUTORIAL STEP 2: Agent Action with MLflow Tracing

        The @mlflow.trace decorator automatically captures:
        - Input parameters (task_goal, available_resources)
        - Output (generated plan)
        - Execution time and metadata

        This trace is what the judge will evaluate.

        Args:
            task_goal: The goal/task to accomplish
            available_resources: List of available tools/APIs/resources

        Returns:
            Multi-step plan as a string
        """
        # Get the planning prompt
        prompt = get_planning_prompt(task_goal, available_resources)

        # Call the LLM to create a plan
        api_params = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5000
        }

        # Add temperature if supported
        # OpenAI models support custom temperatures
        # Some Databricks models only support default (1.0)
        if self.config.provider == "openai":
            api_params["temperature"] = self.config.temperature

        response = self._call_llm(**api_params)
        plan = response.choices[0].message.content.strip()

        # Note: No need to log_param here - the trace automatically captures inputs/outputs
        return plan

    def evaluate(self, trace_id: str) -> Dict[str, Any]:
        """
        TUTORIAL STEP 3: Evaluate with the Judge

        The judge evaluates the trace and returns structured feedback:
        - feedback.value: quality level (excellent/good/adequate/poor/very_poor)
        - feedback.rationale: Detailed explanation of the evaluation

        The quality level is then mapped to a numeric score (1-5).

        Args:
            trace_id: MLflow trace ID to evaluate

        Returns:
            Dictionary with evaluation results:
            - score: int (1-5)
            - quality: str (quality level)
            - reasoning: str (detailed explanation)

        Example:
            >>> trace_id = mlflow.get_last_active_trace_id()
            >>> result = judge.evaluate(trace_id)
            >>> print(result['score'])      # 5
            >>> print(result['quality'])    # "excellent"
            >>> print(result['reasoning'])  # Detailed explanation
        """
        # Fetch the trace from MLflow
        trace = mlflow.get_trace(trace_id)

        # Call the judge to evaluate the trace
        # The judge analyzes the trace and returns feedback
        feedback = self.judge(trace=trace)

        # Map quality level to numeric score (1-5)
        quality_level = feedback.value
        score = get_quality_score(quality_level)

        # Return structured result for easy consumption
        return {
            "score": score,
            "quality": quality_level,
            "reasoning": feedback.rationale
        }

    @mlflow.trace(span_type=SpanType.LLM, name="llm_call")
    def _call_llm(self, **api_params):
        """Call LLM with MLflow tracing."""
        return self.client.chat.completions.create(**api_params)

    def execute_plan_with_tools(self, plan: str, task_goal: str) -> Dict[str, Any]:
        """
        NEW: Execute a plan with actual tool calls.

        This method demonstrates the complete workflow:
        1. Agent has created a plan
        2. Executor runs the plan step-by-step calling real tools
        3. Each tool returns simulated results
        4. Results are passed between steps via context

        Args:
            plan: Multi-step plan text (from create_plan)
            task_goal: Original task goal

        Returns:
            Dict with execution results and summary

        Example:
            >>> plan = judge.create_plan(goal, resources)
            >>> result = judge.execute_plan_with_tools(plan, goal)
            >>> print(f"Success: {result['successful_steps']}/{result['total_steps']}")
        """
        from genai.agents.agent_planning.agent_planning_executor import AgentPlanningExecutor

        # Create executor with same config
        executor = AgentPlanningExecutor(self.config)

        # Execute the plan
        execution_result = executor.execute_plan(plan, task_goal)

        return execution_result
