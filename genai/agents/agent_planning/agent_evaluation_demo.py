"""
MLflow Agent Evaluation Demo

This script demonstrates how to:
1. Build a simple AI agent with tools
2. Create an evaluation dataset
3. Define custom scorers
4. Run MLflow evaluation and view results in the UI
"""

import mlflow
from mlflow.genai import scorer

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Step 1: Build a simple agent with tools
# For this demo, we'll simulate an agent without requiring OpenAI API
# In production, you'd use a framework like LangChain, LlamaIndex, or custom agent

class SimpleCalculatorAgent:
    """A simple calculator agent that can perform basic math operations"""

    def __init__(self):
        self.tools = {
            "add": lambda a, b: a + b,
            "multiply": lambda a, b: a * b,
            "subtract": lambda a, b: a - b,
            "divide": lambda a, b: a / b if b != 0 else "Error: Division by zero"
        }

    def run(self, task: str) -> dict:
        """
        Simulate agent execution with tool calls
        Returns a dict with 'output' and 'tool_calls' for evaluation
        """
        # Simple rule-based parsing for demo purposes
        task_lower = task.lower()

        if "15% of 240" in task_lower or "15 percent of 240" in task_lower:
            # Simulating: multiply(240, 0.15) = 36
            tool_calls = [{"name": "multiply", "args": {"a": 240, "b": 0.15}}]
            output = "36"
        elif "sum of 25 and 37" in task_lower or "25 + 37" in task_lower:
            # Simulating: add(25, 37) = 62
            tool_calls = [{"name": "add", "args": {"a": 25, "b": 37}}]
            output = "62"
        elif "120 divided by 4" in task_lower or "120 / 4" in task_lower:
            # Simulating: divide(120, 4) = 30
            tool_calls = [{"name": "divide", "args": {"a": 120, "b": 4}}]
            output = "30"
        elif "product of 8 and 9" in task_lower or "8 * 9" in task_lower:
            # Simulating: multiply(8, 9) = 72
            tool_calls = [{"name": "multiply", "args": {"a": 8, "b": 9}}]
            output = "72"
        else:
            tool_calls = []
            output = "I don't understand that question"

        return {
            "output": output,
            "tool_calls": tool_calls
        }


# Step 2: Create evaluation dataset
eval_dataset = [
    {
        "inputs": {"inputs": "What is 15% of 240?"},
        "expectations": {"answer": "36", "expected_tools": ["multiply"]},
        "tags": {"topic": "percentage", "difficulty": "easy"},
    },
    {
        "inputs": {"inputs": "Calculate the sum of 25 and 37"},
        "expectations": {"answer": "62", "expected_tools": ["add"]},
        "tags": {"topic": "addition", "difficulty": "easy"},
    },
    {
        "inputs": {"inputs": "What is 120 divided by 4?"},
        "expectations": {"answer": "30", "expected_tools": ["divide"]},
        "tags": {"topic": "division", "difficulty": "easy"},
    },
    {
        "inputs": {"inputs": "Find the product of 8 and 9"},
        "expectations": {"answer": "72", "expected_tools": ["multiply"]},
        "tags": {"topic": "multiplication", "difficulty": "easy"},
    },
]


# Step 3: Define custom scorers
@scorer
def exact_match(outputs, expectations) -> bool:
    """Check if the agent's output exactly matches the expected answer"""
    try:
        return str(outputs.get("output", "")).strip() == str(expectations["answer"]).strip()
    except Exception as e:
        return False


@scorer
def correct_tool_used(outputs, expectations) -> bool:
    """Check if the agent used the correct tool(s)"""
    try:
        tool_calls = outputs.get("tool_calls", [])
        if not tool_calls:
            return False

        expected_tools = expectations.get("expected_tools", [])
        used_tools = [call["name"] for call in tool_calls]

        # Check if all expected tools were used
        return all(tool in used_tools for tool in expected_tools)
    except Exception as e:
        return False


@scorer
def efficiency_score(outputs, expectations) -> float:
    """Score based on number of tool calls (fewer is better for simple tasks)"""
    try:
        tool_calls = outputs.get("tool_calls", [])
        num_calls = len(tool_calls)

        # For simple math, 1 tool call is optimal
        if num_calls == 0:
            return 0.0
        elif num_calls == 1:
            return 1.0
        else:
            # Penalize for using too many tools
            return max(0.0, 1.0 - (num_calls - 1) * 0.2)
    except Exception as e:
        return 0.0


# Step 4: Create predict function for evaluation
agent = SimpleCalculatorAgent()

def predict_fn(inputs: str) -> dict:
    """Wrapper function for MLflow evaluation"""
    # MLflow will pass the inputs dict directly unpacked
    result = agent.run(inputs)
    return result


# Step 5: Run evaluation
print("=" * 80)
print("Running MLflow Agent Evaluation Demo")
print("=" * 80)
print()

# Create an experiment for this evaluation
experiment_name = "agent-evaluation-demo"
mlflow.set_experiment(experiment_name)

print(f"Experiment: {experiment_name}")
print(f"Number of test cases: {len(eval_dataset)}")
print(f"Scorers: exact_match, correct_tool_used, efficiency_score")
print()
print("Running evaluation...")
print()

# Run the evaluation
with mlflow.start_run(run_name="calculator-agent-eval") as run:
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=[exact_match, correct_tool_used, efficiency_score],
    )

    run_id = run.info.run_id

print("=" * 80)
print("Evaluation Complete!")
print("=" * 80)
print()
print(f"Run ID: {run_id}")
print()
print("View results in MLflow UI:")
print(f"  http://localhost:5000/#/experiments/{mlflow.get_experiment_by_name(experiment_name).experiment_id}/runs/{run_id}")
print()
print("Summary of Results:")
print("-" * 80)

# Display aggregate metrics
if hasattr(results, 'metrics') and results.metrics:
    for metric_name, metric_value in results.metrics.items():
        print(f"  {metric_name}: {metric_value:.2%}" if isinstance(metric_value, float) else f"  {metric_name}: {metric_value}")
else:
    print("  Check the MLflow UI for detailed results and traces")

print()
print("=" * 80)
print("Next Steps:")
print("=" * 80)
print("1. Open the MLflow UI link above to see detailed evaluation results")
print("2. Inspect individual test cases and their scores")
print("3. Examine traces to understand agent behavior")
print("4. Iterate on your agent based on the evaluation insights")
print()
