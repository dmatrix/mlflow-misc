# 🦋 Insect Expert Chat - Streamlit App

An interactive AI agent that answers questions about insects using **Databricks Foundation Model Serving endpoints** or **OpenAI** models with **MLflow 3.x tracing** and **LLM-as-a-Judge evaluation**.

## 🚀 Quick Start

### 1. Set Environment Variables

**For Databricks Foundation Model Serving endpoints (Default):**
```bash
export DATABRICKS_TOKEN='your-databricks-token'
export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'
```

**Note:** Requires access to Databricks Foundation Model Serving endpoints.

**For OpenAI (Alternative):**
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Launch the App

```bash
# Normal mode
streamlit run insect_expert_streamlit.py

# Debug mode (shows evaluation details in console)
streamlit run insect_expert_streamlit.py -- --debug
```

The app will open at http://localhost:8501

### 4. View MLflow Traces (Optional)

```bash
# In a separate terminal
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000 → Traces tab
```

---

## 🤖 Available Models

### Databricks Foundation Model Serving endpoints
- **GPT-5** - Latest OpenAI model via Databricks
- **Gemini 2.5 Flash** - Google's fast model via Databricks
- **Claude Sonnet 4.5** - Anthropic's Claude via Databricks

### OpenAI Models
- **GPT-4**
- **GPT-4 Turbo**
- **GPT-3.5 Turbo**

---

## 💬 Example Questions

Try asking:
- What's the difference between butterflies and moths?
- How do fireflies produce light?
- Why are bee populations declining?
- How do ants communicate?

---

## 🔧 Features

### Core Capabilities
- **Multiple Models**: Switch between GPT-5, Gemini, and Claude via Databricks Foundation Model Serving endpoints
- **Provider Selection**: Toggle between Databricks Foundation Model Serving endpoints and OpenAI
- **Temperature Control**: Adjust response creativity (0.0-1.0)
- **Chat History**: Persistent conversation tracking with evaluation scores
- **Secure Config**: Environment variable-based credentials

### MLflow Integration
- **Full Tracing**: Complete observability with `@mlflow.trace` decorators
- **LLM-as-a-Judge**: Automatic response quality evaluation with custom judges
- **Predefined Scorers**: Correctness, Relevance, and Guidelines evaluation (logged to MLflow)
- **Categorical Ratings**: Excellent, Good, Fair, Poor with detailed rationale

### Evaluation Features
Enable "LLM-as-a-Judge Evaluation" in the sidebar to get:
- **Real-time Quality Ratings**: Each response evaluated for quality (🟢 Excellent → 🔴 Poor)
- **Detailed Analysis**: Judge provides structured feedback on:
  - Insect-specific relevance
  - Scientific accuracy and terminology
  - Clarity and engagement
  - Appropriate length (2-4 paragraphs)
- **Judge Model Selection**: Choose between Gemini, Claude, or GPT-5 as your judge
- **MLflow Metrics**: Ratings logged as metrics for tracking over time

---

## 🔍 Troubleshooting

**"Failed to initialize agent"**

For Databricks:
```bash
# Check environment variables
echo $DATABRICKS_TOKEN
echo $DATABRICKS_HOST

# Verify format
# Token: dapi... or similar
# Host: https://your-workspace.cloud.databricks.com
```

For OpenAI:
```bash
# Check API key
echo $OPENAI_API_KEY
```

**Common Issues:**
- Missing environment variables
- Invalid token/API key
- No access to Databricks Foundation Model Serving endpoints
- Incorrect workspace URL

---

## 🔐 Security

**Best Practices:**
- ✅ Use environment variables for credentials
- ✅ Never commit tokens to version control
- ✅ Rotate tokens regularly
- ❌ Don't hardcode credentials in code
- ❌ Don't share tokens in screenshots

---

## 📦 Dependencies

```toml
mlflow>=3.3.2           # Experiment tracking
streamlit>=1.39.0       # Web interface
openai>=1.0.0           # OpenAI client
databricks-sdk>=0.20.0  # Databricks integration
```

---

## 📖 Documentation

**Databricks Resources:**
- [Personal Access Token Guide](https://docs.databricks.com/en/dev-tools/auth/pat.html)
- [Foundation Model APIs Documentation](https://docs.databricks.com/en/machine-learning/foundation-models/index.html)

**Architecture:**
- `insect_expert_openai.py` - Agent class with MLflow tracing
- `insect_expert_streamlit.py` - Streamlit UI with chat interface

---

Made with ❤️ using **MLflow 3.x** + **Databricks Foundation Model Serving endpoints** + **Streamlit**
