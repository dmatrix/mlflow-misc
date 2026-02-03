# MLflow GenAI Tutorial Series

## Tutorial 1: Getting Started with GenAI and MLflow

This tutorial series teaches you how to use MLflow's open source platform for building, tracking, and debugging GenAI applications.

### 📚 Tutorial Structure

#### Notebook 1.1: Setup and Introduction (15-20 min)
- Understanding MLflow for GenAI
- Installation and configuration
- First tracked run
- MLflow UI basics

#### Notebook 1.2: Experiment Tracking for LLMs (25-30 min)
- Tracking LLM parameters and metrics
- Comparing model configurations
- Cost tracking and optimization
- Organizing experiments with tags
- Parent-child runs for workflows

#### Notebook 1.3: Introduction to Tracing (30-35 min)
*(Coming next)*
- Auto-tracing with MLflow
- Understanding the trace model
- Manual instrumentation
- Viewing traces in UI

#### Notebook 1.4: Manual Tracing and Advanced Observability (30-35 min)
*(Coming next)*
- Custom span decorators
- Tracing complex workflows
- Debugging with traces
- Multi-step agentic patterns

#### Notebook 1.5: Prompt Management (15-20 min)
*(Coming next)*
- Creating prompt templates
- Versioning prompts
- Linking prompts to experiments

#### Notebook 1.6: Framework Integrations (10 min)
*(Coming next)*
- OpenAI integration
- LangChain integration
- LlamaIndex integration

#### Notebook 1.7: Complete RAG Application (20-25 min)
*(Coming next)*
- Building a full RAG pipeline
- End-to-end tracing
- Performance analysis

### 🚀 Getting Started

1. **Install Dependencies**
```bash
pip install mlflow>=2.10.0 openai python-dotenv jupyter
```

2. **Configure API Keys**
Create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
MLFLOW_TRACKING_URI="http://localhost:5000
```

3. **Start Jupyter**
```bash
jupyter notebook
```

4. **Start MLflow UI** (in a separate terminal)
```bash
mlflow ui --port 5000
```

5. **Open Browser**
Navigate to http://localhost:5000

### 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Basic understanding of Python and LLMs
- Jupyter Notebook or JupyterLab

### 🎯 Learning Objectives

By the end of Tutorial 1, you will:

- ✅ Understand MLflow's core GenAI components
- ✅ Track LLM experiments systematically
- ✅ Implement comprehensive tracing for observability
- ✅ Debug GenAI applications using trace visualizations
- ✅ Manage prompts with version control and Prompt Registery
- ✅ Build production-ready RAG applications

### 📂 Directory Structure

```
mlflow-genai-tutorial-1/
├── 01_setup_and_introduction.ipynb
├── 02_experiment_tracking.ipynb
├── 03_introduction_to_tracing.ipynb     
├── 04_manual_tracing_advanced.ipynb     
├── 05_prompt_management.ipynb           
├── 06_framework_integrations.ipynb      (coming soon)
├── 07_complete_rag_application.ipynb    (coming soon)
├── .env                                  (create this)
└── README.md
```

### 🔗 Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [MLflow GenAI Guide](https://mlflow.org/docs/latest/genai/)
- [MLflow GitHub](https://github.com/mlflow/mlflow)
- [MLflow Community](https://mlflow.org/community/)

### 🎓 Next Tutorial Series

- **Tutorial 2**: Prompt Engineering and Version Control
- **Tutorial 3**: Tracing and Debugging LLM and Agentic Workflows
- **Tutorial 4**: Evaluating Agents with MLflow
- **Tutorial 5**: Optimizing Prompts for Performance and Better Results

### 💡 Tips

- Keep the MLflow UI open while working through notebooks
- Experiment with different parameter values
- Compare runs in the UI to understand trade-offs
- Tag runs for easy organization
- Track costs from the beginning

### ❓ Troubleshooting

**Issue**: MLflow UI won't start
```bash
# Try a different port
mlflow ui --port 5001
```

**Issue**: API key not recognized
```python
# Restart Jupyter kernel after adding to .env
# Or set manually:
import os
os.environ["OPENAI_API_KEY"] = "your-key"
```

**Issue**: Module not found
```bash
pip install mlflow openai python-dotenv --upgrade
```

### 📝 License

This tutorial series is provided as educational content for learning MLflow's GenAI capabilities.

### 🤝 Contributing

Feedback and suggestions welcome! Please open an issue or submit a pull request.

---

**Author**: Jules (Databricks Developer Relations)
**Date**: January 2025
**MLflow Version**: 2.10.0+
