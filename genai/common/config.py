"""
Configuration dataclasses for GenAI agents.

This module provides shared configuration structures used across agents.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any


@dataclass
class AgentConfig:
    """
    Standard agent configuration.

    This dataclass provides a common configuration structure for all agents,
    reducing boilerplate and ensuring consistency.

    Attributes:
        model: Model identifier
        temperature: Sampling temperature (0.0-2.0)
        provider: LLM provider ("openai" or "databricks")
        api_key: API key for the provider (optional, can use env var)
        databricks_host: Databricks workspace URL (required for Databricks)
        databricks_token: Databricks token (optional, can use env var)
        enable_evaluation: Whether to enable LLM-as-judge evaluation
        enable_mlflow: Whether to enable MLflow tracking
        mlflow_experiment: MLflow experiment name
        debug: Enable debug logging

    Examples:
        >>> # OpenAI configuration
        >>> config = AgentConfig(
        ...     model="gpt-4",
        ...     temperature=0.2,
        ...     provider="openai"
        ... )
        >>>
        >>> # Databricks configuration
        >>> config = AgentConfig(
        ...     model="databricks-gpt-4o",
        ...     temperature=0.7,
        ...     provider="databricks",
        ...     databricks_host="https://your-workspace.cloud.databricks.com"
        ... )
    """
    model: str
    temperature: float = 0.2
    provider: Literal["openai", "databricks"] = "openai"
    api_key: Optional[str] = None
    databricks_host: Optional[str] = None
    databricks_token: Optional[str] = None
    enable_evaluation: bool = False
    enable_mlflow: bool = True
    mlflow_experiment: str = "default-agent"
    mlflow_tracking_uri: Optional[str] = None
    debug: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        import os

        # Validate temperature
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")

        # Auto-populate from environment variables if not provided
        if self.provider == "databricks":
            if self.databricks_host is None:
                self.databricks_host = os.environ.get("DATABRICKS_HOST")
            if self.databricks_token is None:
                self.databricks_token = os.environ.get("DATABRICKS_TOKEN")

        if self.provider == "openai":
            if self.api_key is None:
                self.api_key = os.environ.get("OPENAI_API_KEY")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of config
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "provider": self.provider,
            "enable_evaluation": self.enable_evaluation,
            "enable_mlflow": self.enable_mlflow,
            "mlflow_experiment": self.mlflow_experiment,
            "debug": self.debug
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration

        Returns:
            AgentConfig instance
        """
        return cls(**config_dict)

    def get_provider_kwargs(self) -> Dict[str, Any]:
        """
        Get provider-specific kwargs for client creation.

        Returns:
            Dictionary of kwargs to pass to provider factory
        """
        if self.provider == "openai":
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            return kwargs
        elif self.provider == "databricks":
            kwargs = {}
            if self.databricks_token:
                kwargs["token"] = self.databricks_token
            if self.databricks_host:
                kwargs["host"] = self.databricks_host
            return kwargs
        else:
            return {}


@dataclass
class EvaluationConfig:
    """
    Evaluation-specific configuration.

    Attributes:
        enabled: Whether evaluation is enabled
        judge_model: Model to use for LLM-as-judge evaluation
        criteria: Evaluation criteria (e.g., ["accuracy", "relevance"])
        custom_prompt: Custom evaluation prompt (optional)
    """
    enabled: bool = False
    judge_model: Optional[str] = None
    criteria: list = field(default_factory=lambda: ["accuracy", "relevance", "completeness"])
    custom_prompt: Optional[str] = None

    def __post_init__(self):
        """Validate evaluation configuration."""
        if self.enabled and not self.judge_model:
            raise ValueError("judge_model is required when evaluation is enabled")


@dataclass
class MLflowConfig:
    """
    MLflow-specific configuration.

    Attributes:
        enabled: Whether MLflow tracking is enabled
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking server URI
        enable_autolog: Enable OpenAI autologging
        enable_system_metrics: Enable system metrics logging
        log_artifacts: Whether to log artifacts (prompts, responses)
    """
    enabled: bool = True
    experiment_name: str = "default-agent"
    tracking_uri: Optional[str] = None
    enable_autolog: bool = True
    enable_system_metrics: bool = False
    log_artifacts: bool = True

    def setup(self) -> None:
        """
        Apply this MLflow configuration.

        Sets up MLflow tracking with the configured settings.
        """
        if not self.enabled:
            return

        from genai.common.mlflow_config import setup_mlflow_tracking

        setup_mlflow_tracking(
            experiment_name=self.experiment_name,
            tracking_uri=self.tracking_uri,
            enable_autolog=self.enable_autolog,
            enable_system_metrics=self.enable_system_metrics
        )
