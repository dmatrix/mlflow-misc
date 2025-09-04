
# 🚀 MLflow-Misc: Comprehensive MLflow Examples & Utilities

```
 ███╗   ███╗██╗     ███████╗██╗      ██████╗ ██╗    ██╗
 ████╗ ████║██║     ██╔════╝██║     ██╔═══██╗██║    ██║
 ██╔████╔██║██║     █████╗  ██║     ██║   ██║██║ █╗ ██║
 ██║╚██╔╝██║██║     ██╔══╝  ██║     ██║   ██║██║███╗██║
 ██║ ╚═╝ ██║███████╗██║     ███████╗╚██████╔╝╚███╔███╔╝
 ╚═╝     ╚═╝╚══════╝╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝ 
                                                        
    Modern MLflow Practices & Educational Examples
```

A comprehensive repository demonstrating **modern MLflow practices** for machine learning lifecycle management, featuring **autolog-first approaches**, **modular utilities**, **multiple backend store options**, **generative AI** capabilites, and much more coming soon...

## 🎯 Repository Purpose

```
┌─────────────────────────────────────────────────────────────────┐
│  EDUCATIONAL RESOURCE + HANDS-ON TUTORIALS                      │
│                                                                 │
│  Perfect for ML Engineers, Data Scientists, and Students who    │
│  want to learn modern MLflow practices through working examples │
└─────────────────────────────────────────────────────────────────┘
```

This repository serves as a **comprehensive learning resource** and **educational guide** for MLflow implementations, covering:

- **Modern MLflow Tracking** with autolog (90% less code than manual logging)
- **Multiple Backend Stores** (File, SQLite, Remote servers)
- **Modular Utility System** for reusable ML operations
- **Educational Examples** with clear explanations and best practices
- **Comprehensive Documentation** with troubleshooting guides
- **UV Project Management** for reproducible development environments

## 📁 Repository Structure

```
mlflow-misc/
├── tracking/                      # MLflow tracking examples & documentation
│   ├── simple_tracking_basic.py  # Main autolog example with CLI support
│   └── README.md                  # Comprehensive tracking documentation
├── spark/                         # Spark MLflow integration examples
│   ├── spark_synthetic_nyc_taxi_data.py     # Synthetic NYC taxi data with RandomForest
│   ├── spark_real_nyc_taxi_data.py          # Real NYC taxi data with RandomForest
│   ├── spark_langchain_multiple_mode.py     # Multi-LLM sentiment analysis (OpenAI/Ollama/Mock)
│   ├── spark_langchain_ollama.py            # Streamlined Ollama + Spark + MLflow integration
│   ├── spark_llamaindex_rag.py              # LlamaIndex RAG with distributed document processing
│   ├── spark_ml_utils.py                    # Common ML functions (training, inference, features)
│   └── README.md                            # Spark MLflow integration guide with Ollama setup
├── utils/                         # Reusable MLflow utility modules
│   ├── mlflow_setup.py           # MLflow configuration & experiment setup
│   ├── data_generation.py        # Synthetic dataset generation
│   ├── sample_documents.py       # Document generator for RAG testing
│   ├── visualization.py          # Plotting & visualization utilities
│   ├── model_evaluation.py       # Model evaluation & metrics calculation
│   ├── loader.py                 # Dynamic module loading with importlib.util
│   └── __init__.py               # Utility imports & convenience functions
├── models/                        # Model-related utilities (future expansion)
├── genai/                         # GenAI/LLM utilities (future expansion)
├── pyproject.toml                 # UV project configuration with entry points
└── README.md                      # This file - project overview
```

## ⭐ Key Features

### **Autolog-First Approach**
- **90% less code** compared to manual logging
- **Automatic parameter, metric, and model logging**
- **Built-in model signatures and input examples**
- **Framework-agnostic** (sklearn, xgboost, pytorch, etc.)

### **Multiple Backend Store Support**
- **File Store** (default) - Perfect for learning and development
- **SQLite Database** - Better performance for local experiments  
- **Remote Tracking Server** - Team collaboration and sharing
- **Command-line interface** for easy backend switching

### **Distributed Computing Support**
- **Apache Spark Integration** with MLflow hybrid logging (sklearn autolog + manual Spark ML)
- **RandomForest regression** on both synthetic and real NYC taxi datasets
- **Refactored Architecture** with common ML utilities eliminating code duplication
- **Identical logic structure** for easy comparison between synthetic vs real data
- **Large-scale synthetic data generation** (500K+ rows) and real-world data processing
- **Scalable ML workflows** with Spark MLlib and proper memory management
- **LangChain + Spark UDFs** distributed NLP processing with modern sentiment analysis
- **LlamaIndex RAG** distributed document processing with vector indexing and retrieval-augmented generation
- **🦙 Ollama Integration** - Local LLM processing with MLflow trace logging (tested with Llama3.2 and Mistral models)

### **Modular Utility System**
- **Dynamic module loading** using `importlib.util`
- **Reusable components** across different ML projects
- **Separation of concerns** - setup, data, visualization, evaluation
- **Easy to extend** for new frameworks and use cases

### **Developer-Friendly Structure**
- **UV project management** for reproducible development environments
- **Entry points** for easy script execution and experimentation
- **Comprehensive documentation** with step-by-step examples
- **Cross-platform compatibility** for all development setups

## 🎓 Learning Path

```
┌─────────────────────────────────────────────────────────────────┐
│                    GET STARTED IN 3 STEPS                       │
└─────────────────────────────────────────────────────────────────┘
```

### **1. Start Here: Basic MLflow Tracking**
```bash
# Quick start with default file store
uv run mlflow-tracking-example

# Or with SQLite backend
python tracking/simple_tracking_basic.py --backend sqlite --db-path ./experiments.db
```

### **2. Explore Documentation**
- **[Tracking Examples](./tracking/README.md)** - Complete MLflow tracking guide
- **[Utility Modules](./utils/)** - Reusable MLflow components

### **3. Customize for Your Use Case**
- Modify `tracking/simple_tracking_basic.py` for your data/models
- Use utility modules in your own projects
- Extend with additional frameworks and backends

## ⚡ Quick Setup

```
╔══════════════════════════════════════════════════════════════════╗
║                      60-SECOND SETUP                             ║
║                   (Optimized for MacBook Pro)                    ║
╚══════════════════════════════════════════════════════════════════╝
```

### 📋 **Prerequisites**
```bash
# Install UV (Python package manager) - Required!
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart your terminal or source your shell profile
source ~/.zshrc  # or ~/.bashrc
```

### **Setup Steps**
```bash
# Clone and setup
git clone <repository-url>
cd mlflow-misc

# Install dependencies with UV
uv sync

# Run basic example
uv run mlflow-tracking-example

# Run Spark examples (requires PySpark)
uv sync --extra spark
uv run mlflow-spark-synthetic    # Synthetic NYC taxi data
uv run mlflow-spark-nyc-taxi     # Real NYC taxi data

# Run Ollama + Spark + MLflow integration (Simplified - RECOMMENDED)
uv sync --extra spark --extra langchain

# Simplified Ollama-only integration (no API keys needed, with MLflow trace logging)
# First install and setup Ollama: brew install ollama && ollama serve && ollama pull llama3.2
uv run mlflow-spark-langchain-ollama --ollama-model llama3.2 --num-samples 1000

# Legacy: Multi-LLM script (more complex, conditional autologging for Ollama)
uv run mlflow-spark-langchain-multiple --llm-type ollama --ollama-model llama3.2 --num-samples 100

# View results in MLflow UI
mlflow ui
```

### **Alternative Installation Methods**
```bash
# Via Homebrew (macOS)
brew install uv

# Via pip (if you prefer)
pip install uv

# Via pipx (recommended for global tools)
pipx install uv
```

## 🎯 Use Cases

```
┌─────────────────────────────────────────────────────────────────┐
│                    WHO IS THIS FOR?                             │
└─────────────────────────────────────────────────────────────────┘
```

### **For ML Engineers**
- **Learn modern MLflow practices** with minimal code
- **Understand different backend stores** and when to use them
- **Get hands-on examples** for ML experiment tracking

### **For Teams**
- **Standardize MLflow usage** across projects
- **Share reusable utilities** for common ML operations
- **Implement consistent experiment tracking** practices

### **For Learning**
- **Hands-on MLflow examples** with real code
- **Best practices demonstration** with modern approaches
- **Comprehensive documentation** with troubleshooting

## 📚 Documentation

| Component | Description | Link |
|-----------|-------------|------|
| **Tracking Examples** | Complete MLflow tracking guide with autolog | [tracking/README.md](./tracking/README.md) |
| **Spark Integration** | Distributed ML + LangChain NLP with Ollama models | [spark/README.md](./spark/README.md) |
| **LlamaIndex RAG** | RAG system with Spark + LlamaIndex + MLflow integration | [README_LLAMA_RAG.md](./README_LLAMA_RAG.md) |
| **🦙 Ollama Integration** | Local LLM setup and MLflow trace logging | [spark/README.md](./spark/README.md) |
| **Utility Modules** | Reusable MLflow components and helpers | [utils/](./utils/) |
| **Project Configuration** | UV setup, dependencies, and entry points | [pyproject.toml](./pyproject.toml) |

## Contributing

This educational repository welcomes contributions to help others learn MLflow:
- Additional ML framework examples and tutorials
- New utility modules with clear documentation
- Documentation improvements and clarifications
- Backend store implementations with usage guides

## License

MIT License - See project configuration for details.

---

**Ready to get started?** Jump to [Tracking Examples](./tracking/README.md) for hands-on MLflow tracking with autolog!
