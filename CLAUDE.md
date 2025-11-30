# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Development Setup
```bash
# Install dependencies 
uv sync

# Install with optional dependencies
uv sync --extra spark --extra langchain --extra llamaindex

# Install dev dependencies
uv sync --extra dev
```

### Running Examples
```bash
# Basic MLflow tracking example
uv run mlflow-tracking-example

# Spark examples
uv run mlflow-spark-synthetic        # Synthetic NYC taxi data
uv run mlflow-spark-nyc-taxi         # Real NYC taxi data
uv run mlflow-spark-langchain-ollama # Ollama + Spark + MLflow
uv run mlflow-spark-llamaindex-rag   # LlamaIndex RAG with Spark

# GenAI Agent examples
uv run mlflow-tool-selection-judge   # LLM-as-a-judge for tool selection evaluation
uv run mlflow-agent-planning-judge   # Multi-step agent planning evaluation

# View MLflow UI
mlflow ui
```

### Code Quality & Testing
```bash
# Run tests
pytest

# Code formatting
black .
isort .

# Type checking  
mypy .

# Linting
flake8 .

# Test coverage
pytest --cov=mlflow_misc --cov-report=html
```

### Build System
```bash
# Build wheel
uv build

# Install in editable mode
uv pip install -e .
```

## Architecture Overview

### Core Structure
- **`utils/`** - Reusable MLflow utilities with dynamic loading pattern via `loader.py`
- **`tracking/`** - Basic MLflow tracking examples with CLI backend switching
- **`spark/`** - Distributed computing examples integrating Spark + MLflow
- **`models/`** - Model-related utilities (future expansion)
- **`genai/`** - GenAI/LLM utilities (future expansion)

### Key Design Patterns

#### Dynamic Utility Loading
The codebase uses `utils/loader.py` with `importlib.util` for dynamic module loading:
```python
from utils.loader import load_mlflow_setup, load_data_generation
mlflow_setup = load_mlflow_setup()
```

#### MLflow Setup Centralization
All MLflow configuration goes through `utils/mlflow_setup.py`:
- `setup_mlflow_tracking()` - Main setup function with autolog
- `setup_experiment_only()` - Experiment-only setup
- `enable_autolog_for_framework()` - Framework-specific autolog

#### Autolog-First Approach
The codebase prioritizes MLflow autolog over manual logging:
- Uses `mlflow.sklearn.autolog()` with comprehensive config
- Enables automatic parameter, metric, and model logging
- Reduces code by ~90% compared to manual logging

#### Backend Store Flexibility
Examples support multiple MLflow backends via CLI:
- File store (default): `--backend file`
- SQLite: `--backend sqlite --db-path ./mlflow.db` 
- Custom URI: `--tracking-uri <uri>`

#### Spark Integration Pattern
Spark examples follow consistent structure:
- Use `spark/spark_ml_utils.py` for common ML operations
- Combine Spark MLlib with sklearn autolog for hybrid logging
- Handle large-scale data processing with proper memory management

### Important Dependencies
- **MLflow 3.3.2** - Core experiment tracking
- **PySpark 4.0.0** - Distributed computing
- **UV** - Package management and project execution
- **LangChain** - LLM integrations (optional)
- **LlamaIndex** - RAG capabilities (optional)

### Entry Points Configuration
All examples are executable via `pyproject.toml` scripts:
- Scripts map to module functions (e.g., `tracking.simple_tracking_basic:main`)
- Enables consistent `uv run` execution pattern
- Supports CLI argument passing to underlying scripts

### Testing & Quality
- **pytest** with coverage reporting
- **black** + **isort** for code formatting (line length 88)
- **mypy** for type checking with strict settings
- **flake8** for linting
- Coverage target: `mlflow_misc` package with test exclusions