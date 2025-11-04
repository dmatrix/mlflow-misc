# MLflow 3.x Insect Expert Agent

A simple Q&A agent that answers questions about insects using Ollama (local LLM) with MLflow tracing. No API key required - runs completely free!

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Install dependencies
uv sync

# 2. Start Ollama (in one terminal)
ollama serve
```

### Option 1: Command Line (Recommended for learning)

```bash
# Run with default question
uv run mlflow-insect-expert-ollama

# Ask your own question
uv run mlflow-insect-expert-ollama --question "How do bees navigate?"

# Try different models
uv run mlflow-insect-expert-ollama --model llama3.1

# Adjust temperature (0.0 = focused, 1.0 = creative)
uv run mlflow-insect-expert-ollama --temperature 0.9
```

**First run downloads the model (~2GB). Subsequent runs are instant!**

### Option 2: Streamlit Web UI (Recommended for experimentation)

```bash
# Launch the web app
streamlit run genai/agents/insect_expert_streamlit.py

# Opens automatically at http://localhost:8501
```

**Features:**
- 💬 Full conversation history
- 🎛️ Interactive controls for model and temperature
- 📊 Toggle MLflow tracking on/off
- 🎨 Beautiful chat interface

### View Traces in MLflow

```bash
mlflow ui
# Open http://localhost:5000 → Traces tab
```

---

## 📖 Example Questions

```bash
# Basics
uv run mlflow-insect-expert-ollama --question "What's the difference between insects and spiders?"
uv run mlflow-insect-expert-ollama --question "How many legs do insects have?"

# Behavior
uv run mlflow-insect-expert-ollama --question "How do ants communicate?"
uv run mlflow-insect-expert-ollama --question "Why are moths attracted to light?"

# Biology
uv run mlflow-insect-expert-ollama --question "Explain butterfly metamorphosis"
uv run mlflow-insect-expert-ollama --question "How do fireflies produce light?"
```

---

## 🔧 Setup Details

### Installing Ollama

If Ollama is not installed:

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from: https://ollama.com/download
```

### Available Models

The agent works with any Ollama model. Popular choices:
- `llama3.2` (default, ~2GB)
- `llama3.1` (larger, better quality)
- `mistral`
- `phi3`

Models download automatically on first use.

---

## 📂 Files

```
genai/agents/
├── insect_expert_ollama.py      # CLI version
└── insect_expert_streamlit.py   # Web UI version
```

---

## 🔍 Troubleshooting

**"ollama server not responding"**
```bash
ollama serve
```

**"model not found"**
```bash
ollama pull llama3.2
```

**Check available models**
```bash
ollama list
```

**No traces in MLflow UI**
- Make sure you ran the agent AFTER starting `mlflow ui`
- Don't use `--no-mlflow` flag

---

## 💡 Tips

1. **Start with CLI** - Understand the basics first
2. **Then try Streamlit** - Better for experimentation and demos
3. **Check MLflow UI** - See how tracing captures agent behavior
4. **Try different models** - Compare quality vs speed
5. **Adjust temperature** - Lower for facts, higher for creativity

---

## 🎓 What You'll Learn

- How to build an agent with MLflow 3.x
- Using `@mlflow.trace` decorators for tracing
- Integrating local LLMs with Ollama
- MLflow experiment tracking and traces
- Building interactive UIs with Streamlit

---

Made with ❤️ using MLflow 3.x + Ollama
