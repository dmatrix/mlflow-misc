"""
Provider client factory for LLM providers.

This module provides a factory pattern for creating OpenAI-compatible clients
for different LLM providers (OpenAI, Databricks Foundation Models, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI
from databricks.sdk import WorkspaceClient
import os


class ClientFactory(ABC):
    """Abstract factory for LLM clients."""

    @abstractmethod
    def create_client(self, **kwargs) -> OpenAI:
        """
        Create and return an OpenAI-compatible client.

        Args:
            **kwargs: Provider-specific arguments

        Returns:
            OpenAI-compatible client instance
        """
        pass


class OpenAIClientFactory(ClientFactory):
    """Factory for OpenAI clients."""

    def create_client(
        self,
        api_key: Optional[str] = None,
        **kwargs
    ) -> OpenAI:
        """
        Create OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            **kwargs: Additional arguments passed to OpenAI constructor

        Returns:
            OpenAI client instance

        Raises:
            ValueError: If API key is not provided and not in environment
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Set it with: export OPENAI_API_KEY='sk-...'"
            )

        return OpenAI(api_key=api_key, **kwargs)


class DatabricksClientFactory(ClientFactory):
    """Factory for Databricks Foundation Model clients."""

    def create_client(
        self,
        token: Optional[str] = None,
        host: Optional[str] = None,
        profile: Optional[str] = None,
        **kwargs
    ) -> OpenAI:
        """
        Create Databricks client (OpenAI-compatible).

        Args:
            token: Databricks token (defaults to DATABRICKS_TOKEN env var)
            host: Databricks host URL (defaults to DATABRICKS_HOST env var)
            profile: Databricks CLI profile name (default: "DEFAULT")
            **kwargs: Additional arguments (currently unused)

        Returns:
            OpenAI-compatible client from Databricks

        Raises:
            ValueError: If token or host are not provided and not in environment
        """
        token = token or os.getenv("DATABRICKS_TOKEN")
        host = host or os.getenv("DATABRICKS_HOST")
        profile = profile or os.getenv("DATABRICKS_PROFILE") or "DEFAULT"

        if not token:
            raise ValueError(
                "DATABRICKS_TOKEN not found. "
                "Generate a token from: Workspace → Settings → Developer → Access Tokens\n"
                "Set it with: export DATABRICKS_TOKEN='your-token'"
            )
        if not host:
            raise ValueError(
                "DATABRICKS_HOST not found. "
                "Format: https://your-workspace.cloud.databricks.com\n"
                "Set it with: export DATABRICKS_HOST='your-host'"
            )

        workspace_client = WorkspaceClient(
            profile=profile,
            host=host,
            token=token
        )

        return workspace_client.serving_endpoints.get_open_ai_client()


def get_client(
    provider: str,
    **kwargs
) -> OpenAI:
    """
    Factory function to get LLM client.

    Args:
        provider: Provider name ("openai" or "databricks")
        **kwargs: Provider-specific arguments

    Returns:
        OpenAI-compatible client

    Raises:
        ValueError: If provider is unknown

    Examples:
        >>> # Create OpenAI client
        >>> client = get_client("openai", api_key="sk-...")
        >>>
        >>> # Create Databricks client
        >>> client = get_client("databricks", token="...", host="...")
        >>>
        >>> # Use environment variables
        >>> client = get_client("openai")  # Uses OPENAI_API_KEY from env
    """
    factories = {
        "openai": OpenAIClientFactory(),
        "databricks": DatabricksClientFactory()
    }

    factory = factories.get(provider.lower())
    if not factory:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported providers: {list(factories.keys())}"
        )

    return factory.create_client(**kwargs)
