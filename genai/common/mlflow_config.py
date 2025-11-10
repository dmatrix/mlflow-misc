"""
MLflow configuration and tracking utilities for GenAI agents.

This module provides standardized MLflow setup for all agents.

Note: For tracing, use @mlflow.trace() decorators directly in your agent code
for simplicity and to avoid wrapper indirection issues.
"""

import mlflow
from typing import Optional


def setup_mlflow_tracking(
    experiment_name: str,
    tracking_uri: Optional[str] = "http://localhost:5000",
    enable_autolog: bool = True,
    enable_system_metrics: bool = False
) -> str:
    """
    Standardized MLflow setup for all agents.

    Args:
        experiment_name: Name of MLflow experiment
        tracking_uri: MLflow tracking URI (defaults to http://localhost:5000)
        enable_autolog: Enable OpenAI autologging (default: True)
        enable_system_metrics: Log CPU/memory metrics (default: False)

    Returns:
        Experiment name

    Examples:
        >>> # Basic setup (uses localhost:5000 by default)
        >>> setup_mlflow_tracking("my-agent")
        >>>
        >>> # With custom tracking server
        >>> setup_mlflow_tracking(
        ...     "my-agent",
        ...     tracking_uri="http://remote-server:5000",
        ...     enable_system_metrics=True
        ... )
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if enable_autolog:
        try:
            mlflow.openai.autolog()
        except Exception:
            # Autolog not available, traces will still work
            pass

    if enable_system_metrics:
        mlflow.enable_system_metrics_logging()

    mlflow.set_experiment(experiment_name)

    return experiment_name
