"""
MLflow configuration and tracking utilities for GenAI agents.

This module provides standardized MLflow setup and run management for all agents.

Note: For tracing, use @mlflow.trace() decorators directly in your agent code
for simplicity and to avoid wrapper indirection issues.
"""

import mlflow
from typing import Optional, Dict, Any


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


class MLflowRunManager:
    """
    Context manager for MLflow runs.

    Provides a clean interface for managing MLflow run lifecycle
    with automatic cleanup.

    Examples:
        >>> manager = MLflowRunManager(run_name="test-run")
        >>> with manager:
        ...     manager.log_params({"model": "gpt-4"})
        ...     manager.log_metrics({"accuracy": 0.95})
    """

    def __init__(self, run_name: Optional[str] = None):
        """
        Initialize run manager.

        Args:
            run_name: Optional name for the run
        """
        self.run_name = run_name
        self.run = None

    def __enter__(self):
        """Start MLflow run."""
        self.run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run."""
        mlflow.end_run()

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to current run.

        Args:
            params: Dictionary of parameters to log
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log metrics to current run.

        Args:
            metrics: Dictionary of metrics to log
        """
        mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact to current run.

        Args:
            local_path: Path to local file
            artifact_path: Optional artifact path within run
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_text(self, text: str, artifact_file: str) -> None:
        """
        Log text content as an artifact.

        Args:
            text: Text content to log
            artifact_file: Artifact filename
        """
        mlflow.log_text(text, artifact_file)

    def set_tag(self, key: str, value: Any) -> None:
        """
        Set a tag on the current run.

        Args:
            key: Tag key
            value: Tag value
        """
        mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """
        Set multiple tags on the current run.

        Args:
            tags: Dictionary of tags to set
        """
        mlflow.set_tags(tags)
