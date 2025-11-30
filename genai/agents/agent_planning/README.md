# Agent Planning Tutorial

This tutorial demonstrates multi-agent planning with tool execution and MLflow-based evaluation.

## Overview

This module showcases a complete workflow for AI agent planning:

1. **Plan Generation**: LLM creates a multi-step plan for a given task
2. **Plan Execution**: Agent executes the plan by calling actual tools
3. **Plan Evaluation**: MLflow judge evaluates both the plan quality and execution results

## Architecture: How LLM Calls Tools

The LLM calls the tools via function calling. The executor orchestrates this process:

```
┌─────────────────────────────────────────────────────────┐
│ 1. Agent creates plan (plain text, numbered steps)     │
│    - LLM generates: "1. Search flights 2. Book flight" │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Executor parses plan into steps                     │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 3. FOR EACH STEP:                                       │
│    a) LLM receives: step description, tool schemas,    │
│       context from previous steps                       │
│    b) LLM uses function calling to select tool and     │
│       determine parameters                              │
│    c) Executor calls the selected tool                 │
└─────────────────────────────────────────────────────────┘
```

### Key Implementation Details

The execution flow is implemented in [agent_planning_executor.py](agent_planning_executor.py):

```python
# Step execution with LLM function calling
def _execute_step(self, step_number: int, step_description: str, task_goal: str):
    # Get tool schemas for LLM function calling
    tool_schemas = get_tool_schemas()

    # Create prompt for LLM
    prompt = get_execution_prompt(
        step_description=step_description,
        task_goal=task_goal,
        available_tools=list(tool_schemas.keys()),
        context=self.execution_context
    )

    # Call LLM with function calling capability
    response = self.client.chat.completions.create(
        model=self.config.model,
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "function", "function": schema} for schema in tool_schemas.values()],
        tool_choice="auto"  # LLM decides which tool to use
    )

    # Extract LLM's tool selection
    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_params = json.loads(tool_call.function.arguments)

    # Execute the tool selected by the LLM
    tool = get_tool(tool_name)
    result = tool.execute(**tool_params)
```

**Important**: The agent doesn't hardcode which tools to call. The LLM dynamically selects:
- **Which tool** to use for each step (e.g., `flight_search_api` vs `hotel_search_api`)
- **What parameters** to pass (e.g., `{"origin": "NYC", "destination": "SFO"}`)
- **Whether** a tool is needed at all (some steps may be informational)

## Available Tools

The tutorial provides 5 simulated APIs that agents can call:

| Tool | Description | Example Parameters |
|------|-------------|-------------------|
| `flight_search_api` | Search for flights | `origin`, `destination`, `date` |
| `booking_api` | Book flights | `flight_id`, `passenger_name` |
| `hotel_search_api` | Search for hotels | `location`, `check_in`, `check_out` |
| `calendar_api` | Manage calendar events | `action`, `title`, `start_time`, `end_time` |
| `email_api` | Send emails | `recipient`, `subject`, `body` |

All tools:
- Inherit from `BaseTool` base class
- Return standardized `ToolResult` objects
- Are automatically traced with MLflow (`@mlflow.trace(span_type=SpanType.TOOL)`)
- Simulate realistic latency and data

See [tools/](tools/) directory for implementation details.

## Components

### 1. AgentPlanningJudge

Main class for plan generation and evaluation.

**Features**:
- Generates multi-step plans using LLM
- Creates MLflow judges for evaluation
- Evaluates plan quality on 5-point scale
- Integrates with MLflow experiments

**Usage**:
```python
from genai.common.config import AgentConfig
from genai.agents.agent_planning import AgentPlanningJudge

config = AgentConfig(
    provider="databricks",
    model="databricks-meta-llama-3-1-70b-instruct"
)

judge = AgentPlanningJudge(config)
results = judge.run_complete_workflow(
    task_goal="Book a flight from NYC to San Francisco for next week"
)
```

### 2. AgentPlanningExecutor

Executes plans by calling actual tools.

**Features**:
- Parses multi-step plans
- Uses LLM function calling for tool selection
- Manages execution context between steps
- Traces all tool calls with MLflow

**Usage**:
```python
from genai.agents.agent_planning import AgentPlanningExecutor

executor = AgentPlanningExecutor(config)
execution_results = executor.execute_plan(
    plan="1. Search for flights from NYC to SFO\n2. Book the cheapest flight",
    task_goal="Book a flight from NYC to San Francisco"
)

print(f"Executed {execution_results['successful_steps']}/{execution_results['total_steps']} steps")
```

### 3. Tool System

Modular tool architecture with:
- **Base class**: `BaseTool` - Abstract class all tools inherit from
- **Tool registry**: Centralized tool management
- **Schema generation**: Automatic JSON schema creation for LLM function calling
- **Result standardization**: `ToolResult` dataclass for consistent outputs

## Complete Workflow Example

```python
from genai.common.config import AgentConfig
from genai.agents.agent_planning import AgentPlanningJudge

# 1. Configure agent
config = AgentConfig(
    provider="databricks",
    model="databricks-meta-llama-3-1-70b-instruct"
)

# 2. Create judge
judge = AgentPlanningJudge(config)

# 3. Define task
task = "Plan a business trip: book flight from NYC to SF, reserve hotel, and schedule meetings"

# 4. Run complete workflow (Plan → Execute → Evaluate)
results = judge.run_complete_workflow(task_goal=task)

# 5. View results
print(f"Plan Quality: {results['evaluation']['quality']} ({results['evaluation']['score']}/5)")
print(f"Execution: {results['execution']['successful_steps']}/{results['execution']['total_steps']} steps successful")
```

## MLflow Integration

All agent actions are traced with MLflow:

```python
@mlflow.trace(span_type=SpanType.AGENT, name="execute_plan")
def execute_plan(self, plan: str, task_goal: str):
    # Execution logic with automatic tracing
    ...

@mlflow.trace(span_type=SpanType.TOOL, name="tool_call")
def execute(self, **params):
    # Tool execution with automatic tracing
    ...
```

### Evaluation with MLflow Judges

The tutorial uses `mlflow.genai.judges.make_judge()` for structured evaluation:

```python
judge = mlflow.genai.judges.make_judge(
    name="agent_planning_quality_judge",
    judge_kind="llm",
    instructions=get_judge_instructions(),
    result_type="DICT",
    scale=QUALITY_SCALE  # 5-point scale: excellent → very_poor
)

evaluation = judge.evaluate(
    input=task_goal,
    output=plan,
    resources=available_resources
)
```

## Running the Tutorial

```bash
# Run with default settings (Databricks)
uv run mlflow-agent-planning-judge

# Run with OpenAI
uv run mlflow-agent-planning-judge --provider openai --model gpt-4

# Specify MLflow backend
uv run mlflow-agent-planning-judge --backend sqlite --db-path ./mlflow.db

# View results in MLflow UI
mlflow ui
```

## Key Concepts Demonstrated

1. **Multi-Agent Planning**: Breaking complex tasks into executable steps
2. **LLM Function Calling**: Dynamic tool selection based on task requirements
3. **Context Management**: Passing results between execution steps
4. **MLflow Tracing**: Comprehensive observability for agent actions
5. **Evaluation as Code**: Reproducible quality assessment with MLflow judges
6. **Tool Abstraction**: Clean separation between tool interface and implementation

## File Structure

```
genai/agents/agent_planning/
├── __init__.py                      # Package exports
├── agent_planning_judge.py          # Main judge class
├── agent_planning_executor.py       # Plan execution engine
├── prompts.py                       # Prompt templates
├── README.md                        # This file
└── tools/                           # Tool implementations
    ├── __init__.py                  # Tool registry
    ├── base.py                      # BaseTool abstract class
    ├── utils.py                     # Fake data generation
    ├── flight_tools.py              # Flight search & booking
    ├── hotel_tools.py               # Hotel search
    ├── calendar_tools.py            # Calendar management
    └── email_tools.py               # Email sending
```

## Next Steps

- Implement custom tools by extending `BaseTool`
- Create custom evaluation criteria by modifying judge instructions
- Experiment with different LLM models for planning vs execution
- Add retry logic and error recovery
- Implement tool result caching for efficiency

## References

- MLflow Tracing: https://mlflow.org/docs/latest/llms/tracing/index.html
- MLflow Judges: https://mlflow.org/docs/latest/llms/llm-evaluate/index.html
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
