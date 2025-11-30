"""
LLM-as-a-Judge Tutorial: Multi-Step Planning Evaluation with MLflow

This tutorial demonstrates how to use MLflow's LLM-as-a-Judge pattern to evaluate
AI agent planning decisions.

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


def main():
    """
    TUTORIAL: Complete LLM-as-a-Judge Example for Planning

    This function demonstrates the complete workflow:
    1. Setup MLflow tracing
    2. Initialize agent and judge
    3. Agent creates a multi-step plan
    4. Judge evaluates the plan's quality
    5. Display results with score and detailed reasoning
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge Tutorial: Multi-Step Planning Evaluation with MLflow",
        epilog="""
Tutorial Examples:

  # Basic usage with Databricks
  export DATABRICKS_TOKEN='your-token'
  export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'
  uv run mlflow-agent-planning-judge

  # Use OpenAI instead
  export OPENAI_API_KEY='sk-...'
  uv run mlflow-agent-planning-judge --provider openai

  # Custom task and resources
  uv run mlflow-agent-planning-judge \\
    --task-goal "Send weekly report email to team" \\
    --available-resources "email_api,calendar_api,database_api"

  # Use different model for judging
  uv run mlflow-agent-planning-judge --judge-model databricks-claude-sonnet-4-5

After running, view traces in MLflow UI:
  mlflow ui
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "databricks"],
        default="databricks",
        help="LLM provider (default: databricks)"
    )
    parser.add_argument(
        "--model",
        help="Model identifier (default: databricks-gpt-5 for databricks, gpt-4o-mini for openai)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature 0.0-2.0 (default: 1.0)"
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="agent-planning-judge",
        help="MLflow experiment name (default: agent-planning-judge)"
    )
    parser.add_argument(
        "--judge-model",
        help="Judge model for evaluation (default: same as main model)"
    )
    parser.add_argument(
        "--task-goal",
        default="Book a flight from NYC to SF for next Tuesday and add to calendar",
        help="Task goal for the agent to plan (default: flight booking task)"
    )
    parser.add_argument(
        "--available-resources",
        default="flight_search_api,booking_api,calendar_api,hotel_search_api,email_api",
        help="Comma-separated list of available resources (default: flight booking resources)"
    )

    args = parser.parse_args()

    # Set model default based on provider if not specified
    if args.model is None:
        if args.provider == "databricks":
            args.model = "databricks-gpt-5"
        else:
            args.model = "gpt-4o-mini"

    # Parse available resources
    available_resources = [r.strip() for r in args.available_resources.split(",")]

    # ========================================================================
    # TUTORIAL STEP 1: Setup MLflow Tracing
    # ========================================================================
    print("\n" + "=" * 70)
    print("TUTORIAL: LLM-as-a-Judge for Multi-Step Planning with MLflow")
    print("=" * 70)

    from genai.common.mlflow_config import setup_mlflow_tracking
    setup_mlflow_tracking(
        experiment_name=args.mlflow_experiment,
        enable_autolog=True
    )
    print("\n[Step 1] MLflow tracing enabled")
    print(f"  └─ Experiment: {args.mlflow_experiment}")
    print(f"  └─ View traces: mlflow ui")

    # ========================================================================
    # TUTORIAL STEP 2: Initialize Agent and Judge
    # ========================================================================
    # For Databricks: DATABRICKS_TOKEN and DATABRICKS_HOST must be set in environment
    # For OpenAI: OPENAI_API_KEY must be set in environment
    config = AgentConfig(
        model=args.model,
        provider=args.provider,
        temperature=args.temperature
    )

    judge_model = args.judge_model or "databricks-gemini-2-5-flash"
    print("\n[Step 2] Initializing Agent and Judge")
    print(f"  └─ Provider: {config.provider}")
    print(f"  └─ Agent Model: {config.model}")
    print(f"  └─ Judge Model: {judge_model}")
    print(f"  └─ Temperature: {config.temperature}")

    judge = AgentPlanningJudge(config, judge_model=judge_model)

    # ========================================================================
    # TUTORIAL STEP 3: Define Test Scenario
    # ========================================================================
    print("\n[Step 3] Planning Scenario")
    print(f"  └─ Task Goal: {args.task_goal}")
    print(f"  └─ Available Resources: {available_resources}")

    # ========================================================================
    # TUTORIAL STEP 4: Agent Creates Plan (with tracing)
    # ========================================================================
    print("\n[Step 4] Agent creates a multi-step plan...")
    try:
        plan = judge.create_plan(args.task_goal, available_resources)
        print(f"  └─ ✓ Plan created:\n")
        # Indent the plan for better readability
        for line in plan.split('\n'):
            if line.strip():
                print(f"      {line}")

        # ========================================================================
        # TUTORIAL STEP 5: Judge Evaluates the Plan
        # ========================================================================
        print("\n[Step 5] Judge evaluates the plan quality...")
        trace_id = mlflow.get_last_active_trace_id()
        result = judge.evaluate(trace_id)

        # ========================================================================
        # TUTORIAL STEP 6: Display Results
        # ========================================================================
        print("\n[Step 6] Evaluation Results")
        print("=" * 70)
        print(f"Quality: {result['quality'].upper()} (Score: {result['score']}/5)")
        print("\nDetailed Assessment:")
        print(f"{result['reasoning']}")
        print("=" * 70)

        # ========================================================================
        # STEP 7: Execute Plan with Tools
        # ========================================================================
        print("\n[Step 7] Executing plan with actual tools...")
        print("=" * 70)

        execution_result = judge.execute_plan_with_tools(plan, args.task_goal)

        print("\n  ✓ Execution Complete!")
        print(f"  └─ Total Steps: {execution_result['total_steps']}")
        print(f"  └─ Successful: {execution_result['successful_steps']}/{execution_result['total_steps']}")

        print("\n  Step-by-Step Results:")
        for step_result in execution_result['step_results']:
            step_num = step_result['step_number']
            tool = step_result.get('tool_used', 'No tool')
            success = '✓' if step_result.get('success') else '✗'

            print(f"  {success} Step {step_num}: {tool}")
            if step_result.get('result'):
                result_preview = str(step_result['result'])[:100]
                print(f"     Result: {result_preview}...")

        print("=" * 70)

        print("\n✓ Tutorial complete! View detailed traces in MLflow UI:")
        print("  mlflow ui\n")

    except Exception as e:
        print(f"\nError during planning evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
