"""
LLM-as-a-Judge Tutorial: Simple Multi-Turn Conversation Agent (No Evaluation Logic)

This module contains a simplified CustomerSupportAgent class for handling conversations only.
Evaluation logic is handled separately in notebooks to demonstrate MLflow evaluation APIs.

This is used by customer_support_agent_v2.ipynb to showcase MLflow evaluation methods.
"""

from genai.common import get_client
from genai.common.config import AgentConfig
from genai.agents.multi_turn.prompts import get_support_agent_system_prompt
import mlflow
from mlflow.entities import SpanType
from typing import Dict, Any, List


class CustomerSupportAgentSimple:
    """
    Simplified Multi-Turn Conversation Agent (Conversation-Only).

    This class demonstrates MLflow 3.7's conversation tracking features:
    1. Session tracking: mlflow.update_current_trace() tags traces with session ID
    2. Multi-turn conversation handling with context retention
    3. MLflow tracing for complete conversation flows

    NOTE: This class handles conversation logic only. For evaluation examples,
    see the customer_support_agent_v2.ipynb notebook which demonstrates
    make_judge(), mlflow.search_traces(), and mlflow.genai.evaluate().
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the customer support agent for multi-turn conversations.

        Args:
            config: Configuration for the agent model (provider, model name, temperature, etc.)
        """
        # Initialize the agent's LLM client
        provider_kwargs = config.get_provider_kwargs()
        self.client = get_client(config.provider, **provider_kwargs)
        self.config = config

        # Track conversation history per session
        self.session_histories = {}  # session_id -> List[Dict]

    @mlflow.trace(span_type=SpanType.CHAT_MODEL, name="handle_support_message")
    def handle_message(self, message: str, session_id: str) -> str:
        """
        Handle a single message with session tracking.

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

    @mlflow.trace(span_type=SpanType.LLM, name="llm_call")
    def _call_llm(self, **api_params):
        """Call LLM with automatic MLflow tracing."""
        return self.client.chat.completions.create(**api_params)
