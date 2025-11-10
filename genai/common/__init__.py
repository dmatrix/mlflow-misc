"""
Common utilities for GenAI agents.

Simple, focused utilities:
- Provider client factory
- MLflow setup
- Configuration dataclasses
"""

from genai.common.providers import get_client
from genai.common.mlflow_config import setup_mlflow_tracking
from genai.common.config import AgentConfig, EvaluationConfig

__all__ = [
    "get_client",
    "setup_mlflow_tracking",
    "AgentConfig",
    "EvaluationConfig",
]
