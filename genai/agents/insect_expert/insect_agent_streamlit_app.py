"""
Streamlit Chat App for Insect Expert Agent using OpenAI/Databricks Foundation Model Serving endpoints and MLflow.

This provides an interactive web interface for the insect expert agent using the refactored modular architecture.

Run with: streamlit run genai/agents/insect_expert/streamlit_app.py
Or: uv run streamlit run genai/agents/insect_expert/streamlit_app.py
"""

import os
import sys
import streamlit as st
import mlflow

from genai.common.config import AgentConfig, EvaluationConfig
from genai.common.mlflow_config import setup_mlflow_tracking
from genai.agents.insect_expert import InsectExpertAgent

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
            # Create agent configuration
            config = AgentConfig(
                model=settings["model"],
                temperature=settings["temperature"],
                provider=settings["provider"],
                api_key=settings.get("api_key"),
                databricks_host=settings.get("databricks_host"),
                databricks_token=settings.get("api_key") if settings["use_databricks"] else None,
                enable_evaluation=settings["enable_evaluation"],
                debug=DEBUG_MODE,
            )

            # Create evaluation config if needed
            eval_config = None
            if settings["enable_evaluation"]:
                eval_config = EvaluationConfig(
                    enabled=True,
                    judge_model=settings["judge_model"]
                )

            # Initialize agent
            st.session_state.agent = InsectExpertAgent(config=config, evaluation_config=eval_config)
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


# ============================================================================
# Main App
# ============================================================================


def main():
    """Main Streamlit application."""
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

    # Initialize session state and agent
    initialize_session_state(settings)

    # Display chat history
    display_chat_history()

    # Chat input
    if question := st.chat_input("Ask me anything about insects..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})

        # Display user message
        with st.chat_message("user"):
            st.markdown(question)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent = st.session_state.agent

                # Run with MLflow tracking if enabled
                if settings["enable_mlflow"]:
                    with mlflow.start_run():
                        # Log parameters
                        mlflow.log_params({
                            "model": agent.config.model,
                            "temperature": agent.config.temperature,
                            "question_length": len(question),
                            "provider": settings["provider"],
                            "interface": "streamlit",
                        })

                        # Get response (with tracing)
                        response = agent.answer_question(question)

                        # Evaluate if enabled (pass question and answer for predefined scorers)
                        eval_scores = {}
                        if settings["enable_evaluation"]:
                            try:
                                eval_scores = agent.evaluate_last_response(question=question, answer=response)
                                if DEBUG_MODE:
                                    print(f"[DEBUG] Evaluation scores: {eval_scores}")
                            except Exception as e:
                                if DEBUG_MODE:
                                    print(f"[DEBUG] Evaluation error: {e}")

                        # Log metrics
                        mlflow.log_metrics({
                            "answer_length": len(response),
                            "answer_words": len(response.split()),
                            "question_words": len(question.split()),
                        })
                else:
                    # Run without MLflow tracking
                    response = agent.answer_question(question)

                    # Evaluate if enabled
                    eval_scores = {}
                    if settings["enable_evaluation"]:
                        try:
                            eval_scores = agent.evaluate_last_response(question=question, answer=response)
                            if DEBUG_MODE:
                                print(f"[DEBUG] Evaluation scores: {eval_scores}")
                        except Exception as e:
                            if DEBUG_MODE:
                                print(f"[DEBUG] Evaluation error: {e}")

                # Display response
                st.markdown(response)

                # Display evaluation inline if available
                if eval_scores and "rating" in eval_scores:
                    with st.expander("📊 Evaluation Results", expanded=False):
                        rating = eval_scores["rating"].lower()
                        rating_display = {
                            "excellent": "🟢 Excellent",
                            "good": "🟡 Good",
                            "fair": "🟠 Fair",
                            "poor": "🔴 Poor",
                        }
                        display_text = rating_display.get(rating, "⚪ Unknown")
                        st.markdown(f"### {display_text}")

                        if "rationale" in eval_scores:
                            st.markdown("**Judge's Analysis:**")
                            formatted_rationale = format_judge_analysis(eval_scores["rationale"])
                            st.info(formatted_rationale)

                # Add assistant message to history with eval scores
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "eval_scores": eval_scores if eval_scores else {}
                })

    # Always show example questions
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
