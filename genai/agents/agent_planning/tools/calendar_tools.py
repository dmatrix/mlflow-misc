"""
Calendar management tools.

Provides simulated calendar operations.
"""

from typing import Dict, Any, List
from datetime import datetime
from .base import BaseTool
from .utils import simulate_latency, random_id, random_datetime
import random


class CalendarAPI(BaseTool):
    """Manage calendar events."""

    def __init__(self):
        super().__init__(
            name="calendar_api",
            description="Add, update, or retrieve calendar events. Supports adding new events, getting events for a date range, and updating existing events."
        )

    def _execute(self, action: str, **params) -> Dict[str, Any]:
        """
        Perform calendar operations (simulated).

        Actions:
        - add_event: Add new calendar event
        - get_events: Get events for a date range
        - update_event: Update existing event

        Args:
            action: Operation to perform
            **params: Action-specific parameters

        Returns:
            Action result
        """
        simulate_latency(100, 250)

        if action == "add_event":
            return {
                "event_id": random_id("EVT"),
                "title": params.get("title", "Untitled Event"),
                "start_time": params.get("start_time"),
                "end_time": params.get("end_time"),
                "location": params.get("location"),
                "description": params.get("description"),
                "status": "created",
                "calendar": "primary",
                "created_at": datetime.now().isoformat()
            }
        elif action == "get_events":
            # Return simulated events
            num_events = random.randint(2, 6)
            return {
                "events": [
                    {
                        "event_id": random_id("EVT"),
                        "title": f"{random.choice(['Meeting', 'Call', 'Appointment', 'Review'])} {i+1}",
                        "start_time": random_datetime(days_offset=random.randint(-7, 30)),
                        "attendees": random.randint(1, 5)
                    }
                    for i in range(num_events)
                ],
                "count": num_events
            }
        elif action == "update_event":
            return {
                "event_id": params.get("event_id"),
                "status": "updated",
                "updated_at": datetime.now().isoformat()
            }
        else:
            return {"status": "unknown_action", "action": action}

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add_event", "get_events", "update_event"],
                    "description": "Calendar operation to perform"
                },
                "title": {
                    "type": "string",
                    "description": "Event title (for add_event)"
                },
                "start_time": {
                    "type": "string",
                    "description": "Event start time in ISO format (for add_event)"
                },
                "end_time": {
                    "type": "string",
                    "description": "Event end time in ISO format (for add_event)"
                },
                "location": {
                    "type": "string",
                    "description": "Event location (optional)"
                },
                "description": {
                    "type": "string",
                    "description": "Event description (optional)"
                },
                "event_id": {
                    "type": "string",
                    "description": "Event ID for update_event"
                },
                "date_range": {
                    "type": "string",
                    "description": "Date range for get_events (e.g., '2024-12-01 to 2024-12-31')"
                }
            },
            "required": ["action"]
        }
