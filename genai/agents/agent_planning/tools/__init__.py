"""
Tool Registry for Agent Planning Tutorial.

Provides centralized access to all available tools that agents can call.
Each tool is registered and can be retrieved by name or as a complete set.
"""

from .base import BaseTool, ToolResult
from .flight_tools import FlightSearchAPI, BookingAPI
from .calendar_tools import CalendarAPI
from .email_tools import EmailAPI
from .hotel_tools import HotelSearchAPI


# Central registry of all available tools
# Tools are instantiated once and reused for all calls
TOOL_REGISTRY = {
    "flight_search_api": FlightSearchAPI(),
    "booking_api": BookingAPI(),
    "calendar_api": CalendarAPI(),
    "hotel_search_api": HotelSearchAPI(),
    "email_api": EmailAPI(),
}


def get_tool(name: str) -> BaseTool:
    """
    Get a tool by name.

    Args:
        name: Tool name (e.g., "flight_search_api")

    Returns:
        Tool instance

    Raises:
        ValueError: If tool name is not registered
    """
    if name not in TOOL_REGISTRY:
        available = ", ".join(TOOL_REGISTRY.keys())
        raise ValueError(
            f"Unknown tool: '{name}'. Available tools: {available}"
        )
    return TOOL_REGISTRY[name]


def get_all_tools() -> dict:
    """
    Get all registered tools.

    Returns:
        Dictionary mapping tool names to tool instances
    """
    return TOOL_REGISTRY.copy()


def get_tool_schemas() -> dict:
    """
    Get schemas for all tools (for LLM function calling).

    Returns schemas in the format expected by OpenAI function calling.

    Returns:
        Dictionary mapping tool names to their schemas
    """
    return {name: tool.get_schema() for name, tool in TOOL_REGISTRY.items()}


def list_tool_names() -> list:
    """
    Get list of all registered tool names.

    Returns:
        List of tool name strings
    """
    return list(TOOL_REGISTRY.keys())


__all__ = [
    # Base classes
    "BaseTool",
    "ToolResult",
    # Tool classes
    "FlightSearchAPI",
    "BookingAPI",
    "CalendarAPI",
    "EmailAPI",
    "HotelSearchAPI",
    # Registry functions
    "TOOL_REGISTRY",
    "get_tool",
    "get_all_tools",
    "get_tool_schemas",
    "list_tool_names",
]
