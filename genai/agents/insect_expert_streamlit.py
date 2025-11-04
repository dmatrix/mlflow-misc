"""
Streamlit Chat App for Insect Expert Agent using Ollama and MLflow.

This provides an interactive web interface for the insect expert agent.
Run with: streamlit run genai/agents/insect_expert_streamlit.py
Or: uv run mlflow-insect-expert-streamlit
"""

import streamlit as st
import mlflow
from ollama import Client

from genai.agents.insect_expert_ollama import (
    InsectExpertOllamaAgent,
    check_ollama_installed,
    check_model_available,
    pull_model,
    setup_mlflow_tracking,
)


# ============================================================================
# Page Configuration
# ============================================================================


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Insect Expert Chat",
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

        # Ollama connection status
        st.subheader("🔌 Ollama Status")
        if check_ollama_installed():
            st.success("Ollama is installed ✓")
        else:
            st.error("Ollama not installed ✗")
            st.markdown(
                """
                **Installation required:**
                ```bash
                # macOS/Linux
                curl -fsSL https://ollama.com/install.sh | sh

                # Or visit: https://ollama.com/download
                ```
                """
            )
            st.stop()

        st.divider()

        # Model selection
        st.subheader("🤖 Model Settings")
        available_models = [
            "llama3.2",
            "llama3.1",
            "llama3.2:1b",
            "llama3.1:8b",
            "mistral",
            "phi3",
        ]

        model = st.selectbox(
            "Select Model",
            available_models,
            index=0,
            help="Choose the Ollama model to use",
        )

        # Check if model is available
        if not check_model_available(model):
            st.warning(f"Model '{model}' not found locally")
            if st.button(f"Download {model}"):
                with st.spinner(f"Downloading {model}... This may take a few minutes."):
                    try:
                        pull_model(model)
                        st.success(f"Model {model} downloaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to download model: {e}")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
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
    agent_key = f"{settings['model']}_{settings['temperature']}"

    if "agent_key" not in st.session_state or st.session_state.agent_key != agent_key:
        st.session_state.agent = InsectExpertOllamaAgent(
            model=settings["model"], temperature=settings["temperature"]
        )
        st.session_state.agent_key = agent_key

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
                    "provider": "ollama",
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
    st.title("🦋 Insect Expert Chat")
    st.markdown(
        """
        Ask me anything about insects! I'm an enthusiastic entomologist ready to answer
        your questions about insect classification, behavior, anatomy, and more.

        *Powered by Ollama (local LLM) + MLflow 3.x tracing*
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
                    st.markdown(
                        """
                        **Troubleshooting:**
                        - Make sure Ollama is running: `ollama serve`
                        - Check if model is available: `ollama list`
                        - Try running the model: `ollama run {model}`
                        """.format(
                            model=settings["model"]
                        )
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
