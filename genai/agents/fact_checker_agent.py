"""
Fact-Checker Agent using MLflow 3.x GenAI framework.

This agent verifies claims using tools and provides verdicts with reasoning.
Fully integrated with MLflow for tracking, tracing, and evaluation.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import mlflow
from mlflow.entities import SpanType
from openai import OpenAI


# ============================================================================
# STEP 1: Tool Definitions
# ============================================================================


@mlflow.trace(span_type=SpanType.TOOL)
def web_search(query: str) -> str:
    """
    Search the web for information to verify a claim.

    Args:
        query: The search query to verify the claim

    Returns:
        Search results as a string
    """
    # Mock implementation for demonstration
    # In production, integrate with SerpAPI, Brave Search, or similar
    mock_results = {
        "mlflow": "MLflow is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment.",
        "python": "Python was created by Guido van Rossum and first released in 1991.",
        "earth": "Earth is the third planet from the Sun and is approximately spherical in shape, with a slight equatorial bulge.",
        "water": "Water boils at 100°C (212°F) at sea level under standard atmospheric pressure (1 atm).",
    }

    query_lower = query.lower()
    for key, value in mock_results.items():
        if key in query_lower:
            return f"Search results for '{query}':\n\n{value}\n\nSource: Scientific consensus and verified documentation."

    return f"Search results for '{query}':\n\nNo specific information found. Further verification needed."


@mlflow.trace(span_type=SpanType.TOOL)
def check_knowledge_base(claim: str) -> Dict[str, Any]:
    """
    Check a knowledge base for verified facts.

    Args:
        claim: The claim to verify

    Returns:
        Dictionary with verification status and source
    """
    # Mock knowledge base - in production, use vector DB or fact database
    knowledge_base = {
        "mlflow is an open-source platform": {
            "verified": True,
            "confidence": 0.95,
            "source": "MLflow official documentation",
            "message": "Verified: MLflow is indeed an open-source platform for ML lifecycle management.",
        },
        "earth is flat": {
            "verified": False,
            "confidence": 1.0,
            "source": "Scientific consensus",
            "message": "False: Earth is spherical, confirmed by extensive scientific evidence.",
        },
        "python was created by guido van rossum": {
            "verified": True,
            "confidence": 1.0,
            "source": "Python Software Foundation",
            "message": "Verified: Python was created by Guido van Rossum in 1991.",
        },
    }

    claim_lower = claim.lower()
    for key, value in knowledge_base.items():
        if key in claim_lower:
            return value

    return {
        "verified": False,
        "confidence": 0.0,
        "source": None,
        "message": "No matching facts found in knowledge base. Requires additional verification.",
    }


@mlflow.trace(span_type=SpanType.TOOL)
def extract_entities(text: str) -> List[str]:
    """
    Extract key entities from a claim for targeted fact-checking.

    Args:
        text: The claim text

    Returns:
        List of extracted entities
    """
    # Simple keyword extraction - in production, use NER or LLM
    keywords = []
    text_lower = text.lower()

    # Common entities for demo
    entity_map = {
        "mlflow": "MLflow",
        "python": "Python",
        "guido van rossum": "Guido van Rossum",
        "earth": "Earth",
        "water": "Water",
        "temperature": "Temperature",
        "boiling point": "Boiling Point",
    }

    for key, entity in entity_map.items():
        if key in text_lower:
            keywords.append(entity)

    return keywords if keywords else ["General claim"]


# Tool definitions for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information to verify a claim. Use this to find current facts and evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to verify the claim",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_knowledge_base",
            "description": "Check a knowledge base for verified facts. Use this to quickly verify well-known facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The claim to verify against the knowledge base",
                    }
                },
                "required": ["claim"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": "Extract key entities from a claim for targeted fact-checking. Use this to identify what to verify.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The claim text to extract entities from",
                    }
                },
                "required": ["text"],
            },
        },
    },
]

# Tool function mapping
AVAILABLE_TOOLS = {
    "web_search": web_search,
    "check_knowledge_base": check_knowledge_base,
    "extract_entities": extract_entities,
}


# ============================================================================
# STEP 2: Agent Implementation
# ============================================================================


class FactCheckerAgent:
    """
    A fact-checking agent that uses tools to verify claims.

    This agent follows MLflow 3.x best practices with full tracing integration.
    """

    def __init__(
        self, model: str = "gpt-4o-mini", max_iterations: int = 5, temperature: float = 0.1
    ):
        """
        Initialize the fact-checker agent.

        Args:
            model: OpenAI model to use
            max_iterations: Maximum agent iterations
            temperature: LLM temperature (lower = more deterministic)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it with: export OPENAI_API_KEY='your-api-key'"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature

        # System prompt for fact-checking
        self.system_prompt = """You are a meticulous fact-checker AI assistant. Your goal is to verify claims using available tools.

Follow these steps:
1. Analyze the claim and identify key facts to verify
2. Use extract_entities to identify what needs verification
3. Use check_knowledge_base first for quick fact verification
4. Use web_search for additional evidence if needed
5. Evaluate the evidence objectively
6. Provide a verdict using EXACTLY this format:

VERDICT: [TRUE | FALSE | PARTIALLY TRUE | UNVERIFIABLE]

Then explain your reasoning with:
- What evidence you found
- Which sources you consulted
- Why you reached this conclusion

Be thorough but efficient. Cite your sources clearly."""

    @mlflow.trace(span_type=SpanType.AGENT)
    def verify_claim(self, claim: str) -> Dict[str, Any]:
        """
        Main method to verify a claim using the agent.

        Args:
            claim: The claim to fact-check

        Returns:
            Dictionary with verdict, reasoning, and metadata
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Please fact-check this claim: {claim}"},
        ]

        for iteration in range(self.max_iterations):
            # Call LLM with tools
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=TOOLS, temperature=self.temperature
            )

            assistant_message = response.choices[0].message

            # Check if we're done (no tool calls)
            if not assistant_message.tool_calls:
                # Agent has reached a conclusion
                return {
                    "verdict": self._extract_verdict(assistant_message.content or ""),
                    "reasoning": assistant_message.content,
                    "iterations": iteration + 1,
                    "claim": claim,
                }

            # Add assistant message to conversation
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in assistant_message.tool_calls
                    ],
                }
            )

            # Execute tool calls
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # Execute the tool
                tool_result = self._execute_tool(function_name, function_args)

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(tool_result),
                    }
                )

        # Max iterations reached
        return {
            "verdict": "UNVERIFIABLE",
            "reasoning": "Could not complete verification within iteration limit",
            "iterations": self.max_iterations,
            "claim": claim,
        }

    def _execute_tool(self, function_name: str, function_args: Dict) -> Any:
        """Execute a tool function safely."""
        if function_name not in AVAILABLE_TOOLS:
            return {"error": f"Unknown tool: {function_name}"}

        try:
            tool_func = AVAILABLE_TOOLS[function_name]
            result = tool_func(**function_args)
            return result
        except Exception as e:
            return {"error": str(e)}

    def _extract_verdict(self, content: str) -> str:
        """Extract the verdict from the agent's response."""
        content_upper = content.upper()

        # Look for verdict in format "VERDICT: TRUE"
        if "VERDICT: TRUE" in content_upper and "PARTIALLY" not in content_upper:
            return "TRUE"
        elif "VERDICT: FALSE" in content_upper:
            return "FALSE"
        elif "VERDICT: PARTIALLY TRUE" in content_upper or "PARTIALLY TRUE" in content_upper:
            return "PARTIALLY TRUE"
        elif "VERDICT: UNVERIFIABLE" in content_upper or "UNVERIFIABLE" in content_upper:
            return "UNVERIFIABLE"

        # Fallback patterns
        if "is true" in content.lower() and "not true" not in content.lower():
            return "TRUE"
        elif "is false" in content.lower():
            return "FALSE"

        return "UNVERIFIABLE"


# ============================================================================
# STEP 3: MLflow Integration and Logging
# ============================================================================


def setup_mlflow_tracking(experiment_name: str = "fact-checker-agent") -> str:
    """
    Set up MLflow tracking for the fact-checker agent.

    Args:
        experiment_name: Name of the MLflow experiment

    Returns:
        The experiment name
    """
    # Enable autologging
    mlflow.openai.autolog()

    # Set or create experiment
    mlflow.set_experiment(experiment_name)

    return experiment_name


def log_fact_checker_agent(agent: FactCheckerAgent, run_name: str = "fact-checker-v1"):
    """
    Log the fact-checker agent to MLflow using models-from-code pattern.

    Args:
        agent: The FactCheckerAgent instance
        run_name: Name for the MLflow run

    Returns:
        ModelInfo object with details about the logged model
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params(
            {
                "model": agent.model,
                "max_iterations": agent.max_iterations,
                "temperature": agent.temperature,
                "num_tools": len(AVAILABLE_TOOLS),
                "tool_names": list(AVAILABLE_TOOLS.keys()),
            }
        )

        # Set as active model for trace linking
        model_name = f"{run.info.experiment_id}/{run.info.run_id}"
        mlflow.set_active_model(name=model_name)

        print(f"✓ Logged agent parameters to MLflow")
        print(f"✓ Active model set: {model_name}")
        print(f"✓ Run ID: {run.info.run_id}")

        return run.info


# ============================================================================
# STEP 4: Evaluation Framework
# ============================================================================


def create_evaluation_dataset() -> List[Dict]:
    """
    Create an evaluation dataset for the fact-checker agent.

    Returns:
        List of evaluation examples with inputs and expected outputs
    """
    return [
        {
            "claim": "The Earth is flat.",
            "expected_verdict": "FALSE",
        },
        {
            "claim": "Python was created by Guido van Rossum.",
            "expected_verdict": "TRUE",
        },
        {
            "claim": "Water boils at 100°C at sea level.",
            "expected_verdict": "TRUE",
        },
        {
            "claim": "MLflow is an open-source platform for machine learning lifecycle management.",
            "expected_verdict": "TRUE",
        },
    ]


def evaluate_fact_checker(agent: FactCheckerAgent):
    """
    Run evaluation on the fact-checker agent.

    Args:
        agent: The FactCheckerAgent to evaluate

    Returns:
        Evaluation results
    """
    eval_data = create_evaluation_dataset()

    print("\n" + "=" * 60)
    print("Running Manual Evaluation")
    print("=" * 60)

    results = []
    for i, example in enumerate(eval_data, 1):
        claim = example["claim"]
        expected = example["expected_verdict"]

        print(f"\n[{i}/{len(eval_data)}] Claim: {claim}")
        result = agent.verify_claim(claim)

        verdict = result["verdict"]
        correct = verdict == expected

        results.append(
            {"claim": claim, "expected": expected, "actual": verdict, "correct": correct, "result": result}
        )

        status = "✓" if correct else "✗"
        print(f"    Expected: {expected}")
        print(f"    Actual: {verdict} {status}")
        print(f"    Iterations: {result['iterations']}")

    # Calculate accuracy
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    print(f"\n{'=' * 60}")
    print(f"Evaluation Accuracy: {accuracy:.1%} ({sum(1 for r in results if r['correct'])}/{len(results)})")
    print(f"{'=' * 60}")

    return results


# ============================================================================
# STEP 5: Main Entry Point
# ============================================================================


def main():
    """
    Main function to run the fact-checker agent with MLflow tracking.
    """
    parser = argparse.ArgumentParser(description="Fact-Checker Agent using MLflow 3.x")
    parser.add_argument("--claim", type=str, help="Claim to fact-check")
    parser.add_argument(
        "--evaluate", action="store_true", help="Run evaluation on test dataset"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="OpenAI model to use"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=5, help="Maximum agent iterations"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="fact-checker-agent",
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("\nPlease set it with:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("\nOr get an API key from: https://platform.openai.com/api-keys")
        return 1

    # Setup MLflow
    experiment_name = setup_mlflow_tracking(args.experiment_name)
    print(f"\n{'=' * 60}")
    print(f"MLflow Fact-Checker Agent")
    print(f"{'=' * 60}")
    print(f"Experiment: {experiment_name}")

    # Create agent
    agent = FactCheckerAgent(
        model=args.model, max_iterations=args.max_iterations, temperature=0.1
    )

    # Log agent to MLflow
    run_info = log_fact_checker_agent(agent)

    # Run evaluation mode
    if args.evaluate:
        evaluate_fact_checker(agent)
        print(f"\n✓ View results in MLflow UI: mlflow ui")
        return 0

    # Interactive fact-checking mode
    if args.claim:
        claim = args.claim
    else:
        # Default example
        claim = "MLflow 3.0 introduces the LoggedModel entity for GenAI applications."

    print(f"\n{'=' * 60}")
    print(f"Claim: {claim}")
    print(f"{'=' * 60}")

    result = agent.verify_claim(claim)

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {result['verdict']}")
    print(f"{'=' * 60}")
    print(f"\nReasoning:")
    print(result["reasoning"])
    print(f"\nIterations: {result['iterations']}")
    print(f"\n{'=' * 60}")
    print(f"✓ View traces in MLflow UI: mlflow ui")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    exit(main())
