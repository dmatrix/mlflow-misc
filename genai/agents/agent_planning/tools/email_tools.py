"""Email sending tools."""

from typing import Dict, Any, List
from datetime import datetime
from .base import BaseTool
from .utils import simulate_latency, random_id


class EmailAPI(BaseTool):
    """Send emails to recipients."""

    def __init__(self):
        super().__init__(
            name="email_api",
            description="Send emails to one or more recipients. Supports subject, body, and attachments."
        )

    def _execute(self, to: str, subject: str, body: str, attachments: List[str] = None) -> Dict[str, Any]:
        """Send email (simulated)."""
        simulate_latency(150, 350)

        return {
            "message_id": random_id("MSG", 6),
            "to": to,
            "subject": subject,
            "status": "sent",
            "sent_at": datetime.now().isoformat(),
            "attachments_count": len(attachments) if attachments else 0,
            "size_bytes": len(body) + sum(len(a) for a in (attachments or []))
        }

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body content"},
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attachment file paths (optional)"
                }
            },
            "required": ["to", "subject", "body"]
        }
