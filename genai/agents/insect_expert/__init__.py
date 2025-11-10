"""
Insect Expert Agent.

A specialized agent for answering questions about insects, entomology,
and related topics.
"""

from genai.agents.insect_expert.agent import InsectExpertAgent
from genai.agents.insect_expert.evaluation import InsectExpertEvaluator

__all__ = [
    "InsectExpertAgent",
    "InsectExpertEvaluator",
]
