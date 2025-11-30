"""
Base Tool Infrastructure for Agent Planning Tutorial.

Provides abstract base class for all callable tools with:
- Automatic MLflow tracing
- Standardized input/output format
- Error handling
- Schema generation for LLM function calling
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import mlflow
from mlflow.entities import SpanType
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class ToolResult:
    """
    Standardized tool execution result.

    All tools return this format for consistency and easy tracing.
    """
    success: bool
    data: Any
    message: str
    execution_time_ms: float
    tool_name: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "execution_time_ms": self.execution_time_ms,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp.isoformat()
        }


class BaseTool(ABC):
    """
    Abstract base class for all callable tools.

    Features:
    - Automatic MLflow tracing for observability
    - Standardized result format
    - Built-in error handling
    - Schema generation for LLM function calling
    - Simulated latency for realistic behavior

    Usage:
        class MyTool(BaseTool):
            def __init__(self):
                super().__init__(
                    name="my_tool",
                    description="What this tool does"
                )

            def _execute(self, param1: str, param2: int) -> Any:
                # Tool-specific logic here
                return result

            def _get_parameters(self) -> Dict[str, Any]:
                # JSON Schema for parameters
                return {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "..."},
                        "param2": {"type": "integer", "description": "..."}
                    },
                    "required": ["param1", "param2"]
                }
    """

    def __init__(self, name: str, description: str):
        """
        Initialize base tool.

        Args:
            name: Tool identifier (snake_case, e.g., "flight_search_api")
            description: Human-readable description for LLM
        """
        self.name = name
        self.description = description

    @mlflow.trace(span_type=SpanType.TOOL, name="tool_call")
    def execute(self, **params) -> ToolResult:
        """
        Execute tool with MLflow tracing.

        This wraps _execute() with:
        - Timing measurement
        - Error handling
        - Result standardization
        - MLflow tracing

        Args:
            **params: Tool-specific parameters

        Returns:
            ToolResult with success status, data, and metadata
        """
        start = time.time()

        try:
            # Call tool-specific implementation
            result = self._execute(**params)
            elapsed = (time.time() - start) * 1000

            return ToolResult(
                success=True,
                data=result,
                message=f"{self.name} executed successfully",
                execution_time_ms=elapsed,
                tool_name=self.name,
                timestamp=datetime.now()
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return ToolResult(
                success=False,
                data=None,
                message=f"Error in {self.name}: {str(e)}",
                execution_time_ms=elapsed,
                tool_name=self.name,
                timestamp=datetime.now()
            )

    @abstractmethod
    def _execute(self, **params) -> Any:
        """
        Implement tool-specific logic.

        This method should:
        - Perform the actual tool operation
        - Return the result data
        - Raise exceptions for errors

        Args:
            **params: Tool-specific parameters

        Returns:
            Tool-specific result data
        """
        pass

    def get_schema(self) -> Dict[str, Any]:
        """
        Return tool schema for LLM function calling.

        Returns JSON Schema format compatible with OpenAI function calling.

        Returns:
            Schema dict with name, description, and parameters
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters()
        }

    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """
        Define tool parameters in JSON Schema format.

        Must return a JSON Schema object describing the parameters.

        Example:
            {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "Description of param1"
                    },
                    "param2": {
                        "type": "integer",
                        "description": "Description of param2"
                    }
                },
                "required": ["param1"]
            }

        Returns:
            JSON Schema dict for parameters
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
