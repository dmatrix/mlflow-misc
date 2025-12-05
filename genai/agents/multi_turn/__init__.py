"""
Multi-Turn Conversation Evaluation module.

Demonstrates MLflow 3.7 session-level evaluation features:
- Session tracking with mlflow.update_current_trace()
- Session-level judges using {{ conversation }} template
- Multi-turn customer support conversations
"""

from genai.agents.multi_turn.customer_support_agent_cls import CustomerSupportAgent
from genai.agents.multi_turn.scenarios import (
    get_scenario_printer_troubleshooting,
    get_scenario_account_access,
    get_all_scenarios
)

__all__ = [
    "CustomerSupportAgent",
    "get_scenario_printer_troubleshooting",
    "get_scenario_account_access",
    "get_all_scenarios"
]
