# LLM-as-a-Judge Tutorial with MLflow

A hands-on tutorial demonstrating how to use MLflow's LLM-as-a-Judge pattern to evaluate AI agent decisions.

## What You'll Learn

This tutorial teaches you how to:

1. **Trace agent actions** with MLflow's `@mlflow.trace()` decorator
2. **Create a judge** using `mlflow.genai.judges.make_judge()`
3. **Evaluate agent decisions** using the judge
4. **Integrate with MLflow experiments** for reproducibility

## Tutorial Scenario

**Problem**: An AI agent selects a tool to answer user queries. How do we evaluate if it chose the right tool?

**Solution**: Use an LLM-as-a-Judge to automatically evaluate the agent's tool selection based on predefined criteria.

**Evaluation Criteria**:
- Does the selected tool match the user's intent?
- Can this tool address the task requirements?
- Are there more suitable tools available?

### Visual Workflow

See [workflow diagrams](docs/workflow-diagram.md) for visual representations of:
- Notebook cell execution flow
- Python script execution flow
- LLM-as-a-Judge pattern sequence diagram
- Side-by-side format comparison

## Files

- `tool_selection_judge.py` - Complete tutorial implementation with step-by-step comments
- `tool_selection_judge.ipynb` - Interactive Jupyter notebook version of the tutorial
- `prompts.py` - All prompts (agent + judge instructions) - easy to customize
- `__init__.py` - Package exports

## Quick Start

### Option A: Interactive Jupyter Notebook (Recommended for Learning)

The notebook provides an interactive learning experience with detailed explanations and multiple examples.

**1. Install dependencies:**
```bash
pip install python-dotenv  # Optional but recommended for credential management
```

**2. Set up credentials:**

Create a `.env` file in the `genai/agents/tools_selection/` directory:

**For Databricks:**
```
DATABRICKS_TOKEN=your-token-here
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
```

**For OpenAI:**
```
OPENAI_API_KEY=sk-your-key-here
```

**3. Launch the notebook:**
```bash
jupyter notebook tool_selection_judge.ipynb
```

**4. Follow the tutorial:**
- Run each cell sequentially
- Try different queries in the interactive examples
- Experiment with the batch testing scenarios

**Notebook Features:**
- ✅ Auto-loads credentials from `.env` file
- ✅ Step-by-step explanations with educational comments
- ✅ Interactive examples to try different queries
- ✅ Batch testing with multiple scenarios
- ✅ Customization examples showing how to modify prompts

### Option B: Command-Line Script

**Option 1: Using Databricks**
```bash
export DATABRICKS_TOKEN='your-token'
export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'
uv run mlflow-tool-selection-judge

# View traces and results in MLflow UI
mlflow ui
```

**Option 2: Using OpenAI**
```bash
export OPENAI_API_KEY='sk-...'
uv run mlflow-tool-selection-judge --provider openai

# View traces and results in MLflow UI
mlflow ui
```

### Advanced Options

```bash
# Try your own query
uv run mlflow-tool-selection-judge --query "Send email to John about the meeting"

# Use a different model for the judge
uv run mlflow-tool-selection-judge --judge-model databricks-claude-sonnet-4-5

# Customize the experiment name
uv run mlflow-tool-selection-judge --mlflow-experiment my-tutorial

# Adjust temperature
uv run mlflow-tool-selection-judge --temperature 0.5

# Run with a different model
uv run mlflow-tool-selection-judge --model gpt-4o --provider openai

# Combine multiple options
uv run mlflow-tool-selection-judge \
  --provider openai \
  --query "What's the current stock price?" \
  --judge-model gpt-4o
```

## Tutorial Walkthrough

### Step-by-Step Code Example

```python
import mlflow
from genai.common.config import AgentConfig
from genai.agents.tools_selection import AgentToolSelectionJudge

# Step 1: Configure the agent
config = AgentConfig(
    model="databricks-gpt-5",
    provider="databricks",
    temperature=0.0
)

# Step 2: Initialize the judge (creates MLflow judge internally)
judge = AgentToolSelectionJudge(
    config=config,
    judge_model="databricks-gemini-2-5-flash"  # Optional: use different model for judging
)

# Step 3: Define your scenario
user_request = "What's the weather like in San Francisco?"
available_tools = ["get_weather_api", "search_web", "get_calendar", "send_email"]

# Step 4: Agent performs action (automatically traced by MLflow)
tool_selected = judge.select_tool(user_request, available_tools)
print(f"Agent selected: {tool_selected}")

# Step 5: Judge evaluates the agent's decision
trace_id = mlflow.get_last_active_trace_id()
result = judge.evaluate(trace_id)

# Step 6: Review the evaluation
print(f"Correct: {result['is_correct']}")
print(f"Reasoning: {result['reasoning']}")
```

### What Happens Under the Hood

1. **Tracing**: The `@mlflow.trace()` decorator captures:
   - Input parameters (user request, available tools)
   - Agent's decision (selected tool)
   - Execution metadata

2. **Judge Creation**: `make_judge()` creates a specialized evaluator:
   - Uses evaluation instructions from `prompts.py`
   - Defines feedback format (correct/incorrect)
   - Configures the judge model

3. **Evaluation**: The judge analyzes the trace:
   - Reviews the agent's input and output
   - Applies evaluation criteria
   - Returns structured feedback (value + rationale)

## Configuration Options

### Agent Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `provider` | LLM provider ("openai" or "databricks") | "databricks" |
| `model` | Model for agent | "databricks-gpt-5" / "gpt-4o-mini" |
| `temperature` | Sampling temperature (0.0-2.0) | 1.0 |
| `judge_model` | Model for judge (optional) | Same as agent model |

### Environment Variables

**Databricks:**
```bash
export DATABRICKS_TOKEN='your-token'
export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'
```

**OpenAI:**
```bash
export OPENAI_API_KEY='sk-...'
```

## Evaluation Output

The judge returns structured feedback:

```python
{
    "is_correct": bool,   # True if tool selection was appropriate
    "reasoning": str      # Detailed explanation from the judge
}
```

## Key Concepts

### MLflow Tracing
The `@mlflow.trace()` decorator automatically captures:
- Function inputs and outputs
- Execution time and metadata
- Nested function calls (parent-child relationships)

### MLflow Judge
Created with `make_judge()`:
- Takes predefined evaluation criteria (instructions)
- Uses an LLM to assess traces
- Returns structured feedback (value + rationale)

### Separation of Concerns
- **Agent**: Performs the task (tool selection)
- **Judge**: Evaluates the agent's performance
- **Prompts**: Define evaluation criteria (easy to modify)

## Customization

All prompts are centralized in [prompts.py](prompts.py) for easy customization.

### Modify Agent Behavior

Change how the agent selects tools:

```python
def get_tool_selection_prompt(user_request: str, available_tools: list) -> str:
    return f"""Your custom tool selection instructions here...

User Request: {user_request}
Available Tools: {available_tools}

Select the best tool and explain why."""
```

### Modify Evaluation Criteria

Change how the judge evaluates:

```python
def get_judge_instructions() -> str:
    return """Your custom evaluation instructions here...

{{ trace }}

Your evaluation criteria here..."""
```

**Important**: Judge instructions must include at least one template variable:
- `{{ trace }}` - The full MLflow trace (recommended)
- `{{ inputs }}` - Just the input parameters
- `{{ outputs }}` - Just the output values
- `{{ expectations }}` - Expected behavior (if defined)

### Add More Test Scenarios

Edit the `main()` function in [tool_selection_judge.py](tool_selection_judge.py):

```python
# Add your own test cases
scenarios = [
    ("What's the weather?", ["get_weather_api", "search_web"]),
    ("Send email to John", ["send_email", "get_calendar"]),
    # ... more scenarios
]
```

## Tutorial Formats Comparison

### Quick Visual Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│  📓 Jupyter Notebook          🐍 Python Script                      │
│  (tool_selection_judge.ipynb) (tool_selection_judge.py)             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Load .env file ✓          1. Export env vars                    │
│  2. Configure provider         2. Run: uv run mlflow-...            │
│  3. Initialize judge           3. Single execution →                │
│  4. Run example                4. View result                       │
│  5. Try more examples ✓                                             │
│  6. Batch test (4 scenarios) ✓                                      │
│  7. View prompts ✓                                                  │
│                                                                     │
│  ✅ Interactive learning       ✅ Automation ready                  │
│  ✅ Multiple examples          ✅ CI/CD integration                 │
│  ✅ Easy experimentation       ✅ Scriptable                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Detailed Comparison

| Feature | Jupyter Notebook | Python Script |
|---------|------------------|---------------|
| **Best For** | Learning & experimentation | Production & automation |
| **Interactivity** | ✅ Run cells individually | ❌ Single execution |
| **Credentials** | `.env` file (auto-loaded) | Environment variables |
| **Examples** | Multiple interactive examples | Single query via CLI |
| **Explanations** | Rich markdown with inline docs | Code comments |
| **Batch Testing** | ✅ Built-in test scenarios | ❌ Manual scripting |
| **Customization** | ✅ Easy to modify and re-run | Requires code edits |

**Recommendation**: Start with the Jupyter notebook to understand the concepts, then use the Python script for automation.

## Next Steps

After completing this tutorial:
1. **Explore MLflow UI**: Run `mlflow ui` to see detailed traces
2. **Modify Prompts**: Edit `prompts.py` to change evaluation criteria
3. **Try Different Models**: Experiment with different agent and judge models
4. **Add More Tools**: Expand the `available_tools` list
5. **Batch Evaluation**: Use the notebook's test scenarios feature
6. **Apply to Your Use Case**: Adapt this pattern for your own agent evaluations

## References

- [MLflow GenAI Judges Documentation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
- [Original Tutorial](https://medium.com/@juanc.olamendy/using-llm-as-a-judge-to-evaluate-agent-outputs-a-comprehensive-tutorial-00b6f1f356cc)
- [Jupyter Notebook](tool_selection_judge.ipynb) - Interactive tutorial in this repository
