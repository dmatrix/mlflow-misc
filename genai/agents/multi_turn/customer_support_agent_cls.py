"""
LLM-as-a-Judge Tutorial: Multi-Turn Conversation Evaluation with MLflow 3.7

This module contains the CustomerSupportAgent class for demonstrating session-level evaluation.

TUTORIAL GOALS:
1. Session tracking with mlflow.update_current_trace()
2. Session-level judges using {{ conversation }} template
3. Multi-turn conversation evaluation
4. Demonstrate MLflow 3.7 new features

SCENARIO:
Customer support agent handles multi-turn conversations. Judges evaluate:
- Coherence: Does conversation flow logically?
- Context Retention: Does agent remember earlier turns?
"""

from genai.common import get_client
from genai.common.config import AgentConfig
from genai.agents.multi_turn.prompts import (
    get_coherence_judge_instructions,
    get_context_retention_judge_instructions,
    get_support_agent_system_prompt,
    map_retention_to_score
)
import mlflow
from mlflow.entities import SpanType, assessment
from mlflow.genai.judges import make_judge
from typing import Dict, Any, List
from typing_extensions import Literal
import os


class CustomerSupportAgent:
    """
    Tutorial: Multi-Turn Conversation Evaluation with MLflow 3.7.

    This class demonstrates MLflow 3.7's session-level evaluation features:
    1. Session tracking: mlflow.update_current_trace() tags traces with session ID
    2. Session-level judges: {{ conversation }} template aggregates multi-turn conversations
    3. Multi-turn evaluation: Evaluate entire conversations, not just individual turns

    Key MLflow 3.7 Feature:
    Using {{ conversation }} in judge instructions automatically makes the judge
    session-level (judge.is_session_level_scorer == True).
    """

    def __init__(self, config: AgentConfig, judge_model: str = None, debug: bool = False):
        """
        Initialize the customer support agent and session-level judges.

        Args:
            config: Configuration for the agent model
            judge_model: Optional separate model for judging (defaults to agent model)
            debug: Enable debug output (default: False)
        """
        # Initialize the agent's LLM client
        provider_kwargs = config.get_provider_kwargs()
        self.client = get_client(config.provider, **provider_kwargs)
        self.config = config
        self.judge_model = judge_model or config.model
        self.debug = debug

        # Track conversation history per session
        self.session_histories = {}  # session_id -> List[Dict]

        # Initialize the MLflow judges
        self._init_judges()

    def _init_judges(self):
        """
        TUTORIAL KEY FEATURE: Initialize Session-Level Judges

        The {{ conversation }} template variable in judge instructions automatically
        makes judges session-level aware. MLflow will:
        1. Search for all traces with matching session ID
        2. Aggregate them into a conversation view
        3. Pass the complete conversation to the judge

        This is the main MLflow 3.7 feature being demonstrated!
        """
        # Set up environment for Databricks (needed by LiteLLM)
        if self.config.provider == "databricks":
            os.environ["OPENAI_API_KEY"] = os.environ.get("DATABRICKS_TOKEN", "")
            os.environ["OPENAI_API_BASE"] = f"{self.config.databricks_host}/serving-endpoints"
            model_uri = f"openai:/{self.judge_model}"
        else:
            model_uri = self.judge_model

        # COHERENCE JUDGE (Boolean feedback)
        # Uses {{ conversation }} template → automatically session-level
        self.coherence_judge = make_judge(
            name="conversation_coherence",
            model=model_uri,
            instructions=get_coherence_judge_instructions(),  # Contains {{ conversation }}
            feedback_value_type=bool  # True = coherent, False = incoherent
        )

        # CONTEXT RETENTION JUDGE (Categorical feedback)
        # Also uses {{ conversation }} template → automatically session-level
        self.context_judge = make_judge(
            name="context_retention",
            model=model_uri,
            instructions=get_context_retention_judge_instructions(),  # Contains {{ conversation }}
            feedback_value_type=Literal["excellent", "good", "fair", "poor"]
        )

        # Verify judges are session-level (important for tutorial demonstration)
        print(f"  └─ Coherence judge is session-level: {self.coherence_judge.is_session_level_scorer}")
        print(f"  └─ Context judge is session-level: {self.context_judge.is_session_level_scorer}")

    @mlflow.trace(span_type=SpanType.CHAT_MODEL, name="handle_support_message")
    def handle_message(self, message: str, session_id: str) -> str:
        """
        TUTORIAL KEY FEATURE: Handle a single message with session tracking.

        The critical step is calling mlflow.update_current_trace() with
        session metadata. This is how MLflow knows which traces belong
        to the same conversation session.

        Args:
            message: User's message
            session_id: Unique session identifier

        Returns:
            Agent's response
        """
        # CRITICAL: Update trace with session metadata
        # This enables session-level evaluation
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": session_id}
        )

        # Get or initialize session history
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []

        # Add user message to history
        history = self.session_histories[session_id]
        history.append({"role": "user", "content": message})

        # Build messages for LLM
        # Include system prompt + full conversation history
        messages = [
            {"role": "system", "content": get_support_agent_system_prompt()}
        ] + history

        # Call LLM
        api_params = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": 500
        }

        if self.config.provider == "openai":
            api_params["temperature"] = self.config.temperature

        response = self._call_llm(**api_params)
        assistant_message = response.choices[0].message.content.strip()

        # Add assistant response to history
        history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def run_conversation(
        self,
        messages: List[str],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Run a complete multi-turn conversation.

        Args:
            messages: List of user messages (turns)
            session_id: Session identifier

        Returns:
            Dictionary with conversation transcript and metadata

        Example:
            >>> messages = [
            ...     "My printer won't turn on",
            ...     "I checked the power cable, still nothing",
            ...     "Ok, I'll try that. Thanks!"
            ... ]
            >>> result = agent.run_conversation(messages, "session-001")
            >>> print(result['turns'])  # 3 turns
        """
        transcript = []

        print(f"\n{'='*70}")
        print(f"Running {len(messages)}-turn conversation (Session: {session_id})")
        print(f"{'='*70}\n")

        for turn_num, user_message in enumerate(messages, 1):
            print(f"Turn {turn_num}/{len(messages)}")
            print(f"  User: {user_message}")

            # Handle message (creates trace with session metadata)
            try:
                response = self.handle_message(user_message, session_id)
                print(f"  Agent: {response}\n")

                transcript.append({
                    "turn": turn_num,
                    "user": user_message,
                    "assistant": response
                })
            except Exception as e:
                print(f"  ✗ Error in turn {turn_num}: {e}\n")
                transcript.append({
                    "turn": turn_num,
                    "user": user_message,
                    "assistant": f"[Error: {str(e)}]",
                    "error": True
                })

        return {
            "session_id": session_id,
            "turns": len(messages),
            "transcript": transcript,
            "history": self.session_histories.get(session_id, [])
        }

    def evaluate_session(self, session_id: str, run_id: str) -> Dict[str, Any]:
        """
        TUTORIAL KEY FEATURE: Evaluate entire conversation session.

        This demonstrates how to:
        1. Search for all traces belonging to a session
        2. Use mlflow.genai.evaluate() with session-level judges
        3. Get feedback on conversation quality

        Args:
            session_id: Session ID to evaluate
            run_id: MLflow run ID for the evaluation

        Returns:
            Dictionary with evaluation results from both judges

        Example:
            >>> result = agent.evaluate_session("session-001", run.info.run_id)
            >>> print(result['coherence']['feedback_value'])  # True/False
            >>> print(result['context_retention']['feedback_value'])  # excellent/good/fair/poor
        """
        # Get current experiment
        experiment = mlflow.get_experiment_by_name(self.config.mlflow_experiment)
        if experiment is None:
            raise ValueError(f"Experiment '{self.config.mlflow_experiment}' not found")

        # Search for all traces in this session
        # Uses MLflow's built-in session filtering
        session_traces = mlflow.search_traces(
            locations=[experiment.experiment_id],
            filter_string=f"run_id = '{run_id}'"
        )

        if len(session_traces) == 0:
            raise ValueError(f"No traces found for run {run_id}")

        print(f"\nEvaluating session with {len(session_traces)} traces...")

        # Evaluate using mlflow.genai.evaluate() with session-level judges
        # This is the MLflow 3.7 way to evaluate conversations
        eval_results = mlflow.genai.evaluate(
            data=session_traces,
            scorers=[self.coherence_judge, self.context_judge]
        )

        # Extract results from the evaluation DataFrame
        result_df = eval_results.result_df

        # Find the actual column names (they might vary in MLflow 3.7)
        coherence_cols = [col for col in result_df.columns if 'coherence' in col.lower()]
        context_cols = [col for col in result_df.columns if 'context' in col.lower()]

        # Get coherence results (should be one row for session-level)
        coherence_value_col = [col for col in coherence_cols if '/value' in col]
        coherence_rationale_col = [col for col in coherence_cols if '/justification' in col or '/rationale' in col]

        coherence_values = result_df[coherence_value_col[0]].dropna() if coherence_value_col else []
        coherence_rationales = result_df[coherence_rationale_col[0]].dropna() if coherence_rationale_col else []

        # Get context retention results
        context_value_col = [col for col in context_cols if '/value' in col]
        context_rationale_col = [col for col in context_cols if '/justification' in col or '/rationale' in col]

        context_values = result_df[context_value_col[0]].dropna() if context_value_col else []
        context_rationales = result_df[context_rationale_col[0]].dropna() if context_rationale_col else []

        # Debug output (only if debug mode enabled)
        if self.debug:
            print(f"  Available columns: {list(result_df.columns)}")
            print("\n  DataFrame preview:")
            print(result_df.head())
            print(f"\n  DataFrame shape: {result_df.shape}")
            print(f"\n  Coherence columns: {coherence_cols}")
            print(f"  Context columns: {context_cols}")
            print(f"\n  Coherence value columns: {coherence_value_col}")
            print(f"  Coherence rationale columns: {coherence_rationale_col}")
            print(f"  Coherence values: {coherence_values.tolist() if hasattr(coherence_values, 'tolist') else coherence_values}")
            print(f"  Coherence rationales: {coherence_rationales.tolist() if hasattr(coherence_rationales, 'tolist') else coherence_rationales}")
            print(f"\n  Context value columns: {context_value_col}")
            print(f"  Context rationale columns: {context_rationale_col}")
            print(f"  Context values: {context_values.tolist() if hasattr(context_values, 'tolist') else context_values}")
            print(f"  Context rationales: {context_rationales.tolist() if hasattr(context_rationales, 'tolist') else context_rationales}")

        # Format results
        results = {
            "session_id": session_id,
            "num_traces": len(session_traces),
            "num_assessments": len(coherence_values),
            "coherence": {
                "feedback_value": coherence_values.iloc[0] if len(coherence_values) > 0 else None,
                "rationale": coherence_rationales.iloc[0] if len(coherence_rationales) > 0 else "",
                "passed": coherence_values.iloc[0] if len(coherence_values) > 0 else False
            },
            "context_retention": {
                "feedback_value": context_values.iloc[0] if len(context_values) > 0 else "fair",
                "rationale": context_rationales.iloc[0] if len(context_rationales) > 0 else "",
                "score": map_retention_to_score(context_values.iloc[0]) if len(context_values) > 0 else 2
            }
        }

        return results

    @mlflow.trace(span_type=SpanType.LLM, name="llm_call")
    def _call_llm(self, **api_params):
        """Call LLM with automatic MLflow tracing."""
        return self.client.chat.completions.create(**api_params)
