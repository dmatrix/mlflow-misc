"""
Customer support conversation scenarios for evaluation tutorial.

Each scenario demonstrates different conversation characteristics
for testing session-level evaluation.
"""

from typing import List, Dict, Any


def get_scenario_printer_troubleshooting() -> Dict[str, Any]:
    """
    Scenario 1: Printer troubleshooting (Good context retention).

    Demonstrates:
    - Agent remembers printer model
    - Progressive troubleshooting steps
    - Good coherence and context retention

    Returns:
        Scenario dictionary with messages and metadata
    """
    return {
        "name": "Printer Troubleshooting",
        "session_id": "session-printer-001",
        "description": "User troubleshoots printer issue with good agent support",
        "expected_coherence": True,
        "expected_retention": "excellent",
        "messages": [
            "My HP LaserJet 3000 won't turn on at all. The power light doesn't come on.",
            "Yes, I've checked the power cable and it's plugged in securely to both the printer and wall outlet.",
            "I tried a different outlet in another room and still nothing. What should I try next?",
            "Okay, I'll contact HP support for a warranty replacement. Thanks for your help!"
        ]
    }


def get_scenario_account_access() -> Dict[str, Any]:
    """
    Scenario 2: Account access issue (Fair context retention).

    Demonstrates:
    - Some context retention issues
    - Agent occasionally re-asks for info
    - Acceptable but not excellent coherence

    Returns:
        Scenario dictionary with messages and metadata
    """
    return {
        "name": "Account Access Issue",
        "session_id": "session-account-002",
        "description": "User can't access account, agent has minor context issues",
        "expected_coherence": True,
        "expected_retention": "good",
        "messages": [
            "I can't log into my account. It says my password is incorrect.",
            "I used the reset link but didn't receive the email yet. It's been 10 minutes.",
            "I checked spam folder too, nothing there. My email is john.doe@email.com.",
            "Great, I got the email now and reset my password. I'm able to log in. Thanks!"
        ]
    }


def get_all_scenarios() -> List[Dict[str, Any]]:
    """
    Get all conversation scenarios.

    Returns:
        List of scenario dictionaries
    """
    return [
        get_scenario_printer_troubleshooting(),
        get_scenario_account_access()
    ]
