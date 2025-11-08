"""
Streamlit Chat App for Insect Expert Agent using OpenAI/Databricks Foundation Model Serving endpoints and MLflow.

This provides an interactive web interface for the insect expert agent.
Run with: streamlit run genai/agents/insect_expert_streamlit.py
Or: uv run mlflow-insect-expert-streamlit

Command line options:
  --debug    Enable debug output (prints evaluation details to console)
"""

import os
import sys
import streamlit as st
import mlflow

from genai.agents.insect_expert_openai import (
    InsectExpertOpenAIAgent,
    setup_mlflow_tracking,
)

# Check for --debug flag in command line arguments
DEBUG_MODE = "--debug" in sys.argv


# ============================================================================
# Page Configuration
# ============================================================================


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Insect Expert Assistant",
        page_icon="🦋",
        layout="centered",
        initial_sidebar_state="expanded",
    )


# ============================================================================
# Sidebar Controls
# ============================================================================


def render_sidebar():
    """Render sidebar with model settings and MLflow controls."""
    with st.sidebar:
        st.title("⚙️ Settings")

        # API Configuration
        st.subheader("🔌 API Configuration")

        # Provider selection
        provider = st.radio(
            "Provider",
            ["Databricks", "OpenAI"],
            index=0,
            help="Choose the API provider",
        )

        # API Key input
        api_key_env = "DATABRICKS_TOKEN" if provider == "Databricks" else "OPENAI_API_KEY"
        api_key_from_env = os.environ.get(api_key_env, "")

        if api_key_from_env:
            st.success(f"{api_key_env} found in environment ✓")
            api_key = api_key_from_env
        else:
            api_key = st.text_input(
                f"{api_key_env}",
                type="password",
                help=f"Enter your {provider} API key",
            )
            if not api_key:
                st.error(f"Please provide {api_key_env}")
                st.info(
                    f"""
                    **Set environment variable:**
                    ```bash
                    export {api_key_env}='your-key'
                    ```
                    """
                )
                st.stop()

        # Databricks Host input (only for Databricks provider)
        databricks_host = None
        if provider == "Databricks":
            databricks_host_from_env = os.environ.get("DATABRICKS_HOST", "")

            if databricks_host_from_env:
                st.success("DATABRICKS_HOST found in environment ✓")
                databricks_host = databricks_host_from_env
            else:
                databricks_host = st.text_input(
                    "DATABRICKS_HOST",
                    type="password",
                    help="Enter your Databricks workspace host URL",
                    placeholder="https://your-workspace.cloud.databricks.com"
                )
                if not databricks_host:
                    st.error("Please provide DATABRICKS_HOST")
                    st.info(
                        """
                        **Set environment variable:**
                        ```bash
                        export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'
                        ```
                        """
                    )
                    st.stop()

        st.divider()

        # Model selection
        st.subheader("🤖 Model Settings")

        if provider == "Databricks":
            # Model options with display names
            model_options = {
                "GPT-5": "databricks-gpt-5",
                "Gemini 2.5 Flash": "databricks-gemini-2-5-flash",
                "Claude Sonnet 4.5": "databricks-claude-sonnet-4-5",
            }
            use_databricks = True
        else:
            # OpenAI models
            model_options = {
                "GPT-4": "gpt-4",
                "GPT-4 Turbo": "gpt-4-turbo",
                "GPT-3.5 Turbo": "gpt-3.5-turbo",
            }
            use_databricks = False

        # Display model selection with friendly names
        model_display = st.selectbox(
            "Select Model",
            list(model_options.keys()),
            index=0,
            help=f"Choose the {provider} model to use",
        )

        # Get the actual model ID
        model = model_options[model_display]

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Higher = more creative, Lower = more focused",
        )

        st.divider()

        # MLflow settings
        st.subheader("📊 MLflow Tracking")
        enable_mlflow = st.checkbox("Enable MLflow", value=True)

        experiment_name = st.text_input(
            "Experiment Name",
            value="insect-expert-streamlit",
            disabled=not enable_mlflow,
        )

        if enable_mlflow:
            st.info(
                """
                **View traces:**
                ```bash
                mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
                ```
                Then open: http://localhost:5000
                """
            )

        st.divider()

        # Evaluation Settings
        st.subheader("🔍 Real-Time Evaluation")

        enable_evaluation = st.checkbox(
            "Enable LLM-as-a-Judge",
            value=True,
            help="Evaluate each response quality using Databricks Foundation Model Serving endpoints (adds ~2-3s latency)",
        )

        if enable_evaluation:
            # Judge model selection (Databricks models only)
            judge_models = {
                "Gemini 2.5 Flash (Recommended)": "databricks-gemini-2-5-flash",
                "Claude Sonnet 4.5": "databricks-claude-sonnet-4-5",
                "GPT-5": "databricks-gpt-5",
            }

            judge_display = st.selectbox(
                "Judge Model",
                list(judge_models.keys()),
                index=0,
                help="Databricks model used to evaluate responses",
            )

            judge_model = judge_models[judge_display]

            st.info(
                """
                **Evaluation Metrics:**
                - 🎯 Insect-specific relevance
                - 🔬 Scientific accuracy
                - ✨ Clarity and engagement
                - 📏 Appropriate length (2-4 paragraphs)

                **Note:** Uses same Databricks credentials
                """
            )
        else:
            judge_model = "databricks-gemini-2-5-flash"

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Store settings in session state
        return {
            "model": model,
            "temperature": temperature,
            "enable_mlflow": enable_mlflow,
            "experiment_name": experiment_name,
            "api_key": api_key,
            "use_databricks": use_databricks,
            "databricks_host": databricks_host,
            "provider": provider.lower(),
            "enable_evaluation": enable_evaluation,
            "judge_model": judge_model,
        }


# ============================================================================
# Session State Management
# ============================================================================


def initialize_session_state(settings):
    """Initialize or update session state based on settings."""
    # Initialize chat messages if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize or update agent if settings changed
    agent_key = f"{settings['model']}_{settings['temperature']}_{settings['provider']}_{settings['enable_evaluation']}_{settings.get('judge_model', '')}"

    if "agent_key" not in st.session_state or st.session_state.agent_key != agent_key:
        try:
            agent_kwargs = {
                "model": settings["model"],
                "temperature": settings["temperature"],
                "api_key": settings["api_key"],
                "use_databricks": settings["use_databricks"],
                "enable_evaluation": settings["enable_evaluation"],
                "judge_model": settings["judge_model"],
                "debug": DEBUG_MODE,  # Pass debug flag
            }

            # Add databricks_host if using Databricks
            if settings["use_databricks"] and settings.get("databricks_host"):
                agent_kwargs["databricks_host"] = settings["databricks_host"]

            st.session_state.agent = InsectExpertOpenAIAgent(**agent_kwargs)
            st.session_state.agent_key = agent_key
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.stop()

    # Setup MLflow experiment if enabled
    if settings["enable_mlflow"]:
        if "mlflow_experiment" not in st.session_state or st.session_state.mlflow_experiment != settings["experiment_name"]:
            setup_mlflow_tracking(settings["experiment_name"])
            st.session_state.mlflow_experiment = settings["experiment_name"]


# ============================================================================
# Chat Interface
# ============================================================================


def format_judge_analysis(rationale: str) -> str:
    """
    Format judge's analysis by making criterion labels bold.

    Args:
        rationale: Raw rationale text from judge

    Returns:
        Formatted rationale with bold labels
    """
    # List of criterion labels to make bold
    criteria = [
        "Insect-specific relevance:",
        "Scientific accuracy and proper terminology:",
        "Clarity and engagement:",
        "Appropriate length:",
    ]

    # Replace each criterion label with bold version
    formatted = rationale
    for criterion in criteria:
        formatted = formatted.replace(criterion, f"**{criterion}**")

    return formatted


def display_chat_history():
    """Display all chat messages from history with evaluation scores."""
    if DEBUG_MODE:
        print(f"[DEBUG] display_chat_history called. Total messages: {len(st.session_state.messages)}")

    for idx, message in enumerate(st.session_state.messages):
        if DEBUG_MODE and message["role"] == "assistant":
            print(f"[DEBUG] Message {idx}: role={message['role']}, has_eval_scores={'eval_scores' in message}")
            if "eval_scores" in message:
                print(f"[DEBUG] Message {idx} eval_scores keys: {list(message['eval_scores'].keys())}")

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display evaluation scores if they exist for this message
            if message["role"] == "assistant" and "eval_scores" in message:
                scores = message["eval_scores"]

                if scores and "rating" in scores:
                    with st.expander("📊 Evaluation Results", expanded=False):
                        # Display rating
                        rating = scores["rating"].lower()
                        rating_display = {
                            "excellent": "🟢 Excellent",
                            "good": "🟡 Good",
                            "fair": "🟠 Fair",
                            "poor": "🔴 Poor",
                        }
                        display_text = rating_display.get(rating, "⚪ Unknown")
                        st.markdown(f"### {display_text}")

                        # Show rationale
                        if "rationale" in scores:
                            st.markdown("**Judge's Analysis:**")
                            formatted_rationale = format_judge_analysis(scores["rationale"])
                            st.info(formatted_rationale)
                elif DEBUG_MODE:
                    print(f"[DEBUG] Message {idx}: eval_scores exists but no rating or empty")


def get_agent_response(question: str, settings: dict) -> tuple[str, bool]:
    """
    Get response from agent with optional MLflow tracking.

    Performs relevance check before calling the main agent to save costs.

    Args:
        question: User's question
        settings: Current app settings

    Returns:
        Tuple of (response: str, is_relevant: bool)
    """
    agent = st.session_state.agent

    # Pre-filter: Check if question is insect-related (saves costs!)
    is_relevant, reason = agent.is_insect_related(question)

    if not is_relevant:
        # Return early without calling the expensive main agent
        rejection_message = f"""I appreciate your question, but I can only answer questions about **insects** 🦋🐛🐝

**Why this question doesn't qualify:** {reason}

**Try asking about:**
- Insect behavior and biology
- Insect identification
- Insect habitats and ecology
- Insect life cycles
- Common or rare insect species

Feel free to ask me anything about insects!"""

        if settings["enable_mlflow"]:
            # Log the rejection for tracking
            with mlflow.start_run():
                mlflow.log_params({
                    "model": agent.model,
                    "question_length": len(question),
                    "provider": settings["provider"],
                    "interface": "streamlit",
                    "rejected": True,
                    "rejection_reason": reason,
                })

        return rejection_message, False

    # Question is relevant, proceed with normal flow
    if settings["enable_mlflow"]:
        # Run with MLflow tracking
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(
                {
                    "model": agent.model,
                    "temperature": agent.temperature,
                    "question_length": len(question),
                    "provider": settings["provider"],
                    "interface": "streamlit",
                    "rejected": False,
                }
            )

            # Get answer (with tracing)
            answer = agent.answer_question(question)

            # Evaluate after trace is complete (if enabled)
            if settings["enable_evaluation"]:
                agent.evaluate_last_response(question=question, answer=answer)

            # Log metrics
            mlflow.log_metrics(
                {
                    "answer_length": len(answer),
                    "answer_words": len(answer.split()),
                    "question_words": len(question.split()),
                }
            )

            return answer, True
    else:
        # Run without MLflow tracking
        answer = agent.answer_question(question)

        # Evaluate after trace is complete (if enabled)
        if settings["enable_evaluation"]:
            agent.evaluate_last_response(question=question, answer=answer)

        return answer, True


# ============================================================================
# Main App
# ============================================================================


def main():
    """Main Streamlit app entry point."""
    configure_page()

    # Title and description
    st.title("🦋 Insect Expert Assistant")

    # Show debug mode indicator if enabled
    if DEBUG_MODE:
        st.warning("🐛 **Debug Mode Enabled** - Evaluation details will be printed to console")

    st.markdown(
        """
        Ask me anything about insects! I'm an enthusiastic entomologist ready to answer
        your questions about insect classification, behavior, anatomy, and more.

        *Powered by OpenAI/Databricks Foundation Model Serving endpoints + MLflow 3.x tracing*
        """
    )

    # Requirements notice
    st.info(
        """
        **Requirements:**
        - **Databricks**: Requires `DATABRICKS_TOKEN` and `DATABRICKS_HOST` environment variables, plus access to Databricks Foundation Model Serving endpoints
        - **OpenAI**: Requires `OPENAI_API_KEY` environment variable
        """
    )

    # Render sidebar and get settings
    settings = render_sidebar()

    # Initialize session state
    initialize_session_state(settings)

    # Display chat history
    display_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask me about insects..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, is_relevant = get_agent_response(prompt, settings)
                    st.markdown(response)

                    # Display evaluation scores only if question was relevant and evaluation is enabled
                    if is_relevant and settings["enable_evaluation"]:
                        agent = st.session_state.agent

                        if hasattr(agent, "last_eval_scores") and agent.last_eval_scores:
                            scores = agent.last_eval_scores

                            if scores and "rating" in scores:
                                with st.expander("📊 Evaluation Results", expanded=True):
                                    judge_name = (
                                        settings["judge_model"]
                                        .replace("databricks-", "")
                                        .replace("-", " ")
                                        .title()
                                    )
                                    st.caption(f"⚖️ Judge: **{judge_name}** (Databricks)")

                                    # Display overall rating with color-coded emoji
                                    rating = scores["rating"].lower()
                                    rating_display = {
                                        "excellent": ("🟢 Excellent", "green"),
                                        "good": ("🟡 Good", "blue"),
                                        "fair": ("🟠 Fair", "orange"),
                                        "poor": ("🔴 Poor", "red"),
                                    }

                                    display_text, color = rating_display.get(
                                        rating, ("⚪ Unknown", "gray")
                                    )

                                    # Show rating prominently
                                    st.markdown(
                                        f"### {display_text}",
                                        help="Overall quality rating from custom LLM judge"
                                    )

                                    # Show rationale if available
                                    if "rationale" in scores and scores["rationale"]:
                                        st.markdown("**Judge's Analysis:**")
                                        formatted_rationale = format_judge_analysis(scores["rationale"])
                                        st.info(formatted_rationale)

                    # Add assistant message to chat history with evaluation scores
                    message_data = {"role": "assistant", "content": response}

                    # Debug: Check conditions for storing eval scores
                    if DEBUG_MODE:
                        agent = st.session_state.agent
                        print(f"[DEBUG] is_relevant: {is_relevant}")
                        print(f"[DEBUG] enable_evaluation: {settings['enable_evaluation']}")
                        print(f"[DEBUG] hasattr(agent, 'last_eval_scores'): {hasattr(agent, 'last_eval_scores')}")
                        if hasattr(agent, "last_eval_scores"):
                            print(f"[DEBUG] agent.last_eval_scores: {agent.last_eval_scores}")
                            print(f"[DEBUG] bool(agent.last_eval_scores): {bool(agent.last_eval_scores)}")

                    # Store evaluation scores with the message
                    if is_relevant and settings["enable_evaluation"]:
                        agent = st.session_state.agent
                        if hasattr(agent, "last_eval_scores") and agent.last_eval_scores:
                            message_data["eval_scores"] = agent.last_eval_scores.copy()

                            # Debug: Show what's being stored
                            if DEBUG_MODE:
                                print(f"[DEBUG] ✅ Storing eval_scores in message: {message_data['eval_scores']}")
                                print(f"[DEBUG] eval_scores keys: {list(message_data['eval_scores'].keys())}")
                        elif DEBUG_MODE:
                            print("[DEBUG] ❌ NOT storing eval_scores - last_eval_scores is empty or missing")

                    st.session_state.messages.append(message_data)

                    # Debug: Verify it was stored
                    if DEBUG_MODE:
                        last_msg = st.session_state.messages[-1]
                        print(f"[DEBUG] Message appended. Has eval_scores: {'eval_scores' in last_msg}")
                        if "eval_scores" in last_msg:
                            print(f"[DEBUG] Stored eval_scores: {last_msg['eval_scores']}")

                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)

                    # Show helpful troubleshooting
                    if settings["provider"] == "databricks":
                        st.markdown(
                            """
                            **Troubleshooting:**
                            - Make sure you have a valid `DATABRICKS_TOKEN`
                            - Verify you have set `DATABRICKS_HOST` correctly
                            - Verify you have access to Databricks Foundation Model Serving endpoints
                            - Check that the model name matches your endpoint configuration
                            """
                        )
                    else:
                        st.markdown(
                            """
                            **Troubleshooting:**
                            - Check that your `OPENAI_API_KEY` is valid
                            - Verify the model name is correct
                            - Ensure you have API access to the selected model
                            """
                        )

    # Show example questions if chat is empty
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Example Questions")
        st.markdown(
            """
            - What makes bees able to fly, and how do their wings work?
            - How do fireflies produce light?
            - What's the difference between butterflies and moths?
            - Why do ants follow each other in lines?
            - How many eyes do spiders have?
            """
        )


if __name__ == "__main__":
    main()
