"""
Streamlit Chat App for Insect Expert Agent using OpenAI/Databricks Foundation Model Serving endpoints and MLflow.

This provides an interactive web interface for the insect expert agent.
Run with: streamlit run genai/agents/insect_expert_streamlit.py
Or: uv run mlflow-insect-expert-streamlit
"""

import os
import streamlit as st
import mlflow

from genai.agents.insect_expert_openai import (
    InsectExpertOpenAIAgent,
    setup_mlflow_tracking,
)


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
                mlflow ui
                ```
                Then open: http://localhost:5000
                """
            )

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
    agent_key = f"{settings['model']}_{settings['temperature']}_{settings['provider']}"

    if "agent_key" not in st.session_state or st.session_state.agent_key != agent_key:
        try:
            agent_kwargs = {
                "model": settings["model"],
                "temperature": settings["temperature"],
                "api_key": settings["api_key"],
                "use_databricks": settings["use_databricks"],
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


def display_chat_history():
    """Display all chat messages from history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def get_agent_response(question: str, settings: dict) -> str:
    """
    Get response from agent with optional MLflow tracking.

    Args:
        question: User's question
        settings: Current app settings

    Returns:
        Agent's response
    """
    agent = st.session_state.agent

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
                }
            )

            # Get answer (with tracing)
            answer = agent.answer_question(question)

            # Log metrics
            mlflow.log_metrics(
                {
                    "answer_length": len(answer),
                    "answer_words": len(answer.split()),
                    "question_words": len(question.split()),
                }
            )

            return answer
    else:
        # Run without MLflow tracking
        return agent.answer_question(question)


# ============================================================================
# Main App
# ============================================================================


def main():
    """Main Streamlit app entry point."""
    configure_page()

    # Title and description
    st.title("🦋 Insect Expert Assistant")
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
                    response = get_agent_response(prompt, settings)
                    st.markdown(response)

                    # Add assistant message to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

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
