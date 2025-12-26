# Multi-Turn Conversation Evaluation with MLflow + DeepEval Integration

Tutorial demonstrating MLflow 3.8's DeepEval integration for session-level evaluation of multi-turn conversations.

## Overview

This tutorial showcases:
1. **MLflow Session Tracking**: Using `mlflow.update_current_trace()` to group conversation turns
2. **DeepEval Integration**: Using `mlflow.genai.scorers.deepeval` for industry-standard metrics
3. **Session-Level Evaluation**: Evaluating entire conversations with DeepEval metrics

## MLflow 3.8 Key Features

### DeepEval Scorer Integration

MLflow 3.8+ provides native integration with DeepEval metrics through `mlflow.genai.scorers.deepeval`:

```python
from mlflow.genai.scorers.deepeval import (
    ConversationCompleteness,
)

# Create DeepEval scorers
completeness_scorer = ConversationCompleteness(model="openai:/gpt-4o-mini")

# These are session-level scorers
print(completeness_scorer.is_session_level_scorer)  # True
```

### Session Tracking with update_current_trace()

```python
@mlflow.trace(span_type="CHAT_MODEL")
def handle_message(message, session_id):
    # CRITICAL: This links the trace to the session
    mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    response = generate_response(message)
    return response
```

### Evaluating Sessions with DeepEval

```python
# Search for all traces in a session
session_traces = mlflow.search_traces(
    experiment_ids=["<experiment-id>"],
    filter_string="metadata.`mlflow.trace.session` = 'session-123'"
)

# Evaluate with DeepEval scorers
eval_results = mlflow.genai.evaluate(
    data=session_traces,
    scorers=[completeness_scorer]
)

# Access results
result_df = eval_results.result_df
completeness_score = result_df['ConversationCompleteness/value'].iloc[0]
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ User sends message                                              │
│   Turn 1: "My printer won't turn on"                            │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ Agent handles message (@mlflow.trace + session metadata)        │
│   mlflow.update_current_trace(metadata={"session": "s-001"})    │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ Agent responds                                                  │
│   Turn 1 Response: "Let's troubleshoot. Is it plugged in?"      │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ User sends message                                              │
│   Turn 2: "Yes, it's plugged in securely"                       │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
      [Repeat for turns 3, 4...]
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ Session-Level Evaluation with DeepEval                          │
│   1. mlflow.search_traces(session="s-001") → All 4 traces       │
│   2. mlflow.genai.evaluate(scorers=[deepeval_scorers])          │
│   3. DeepEval metrics evaluate complete conversation            │
│   4. Results: Completeness, Recall, Retention scores            │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### DeepEval Scorers Used

**1. ConversationCompleteness**
- Evaluates whether the conversation satisfies user's needs
- Assesses if conversation reaches satisfactory conclusion
- Score: 0.0-1.0 (threshold configurable)

**2. ContextualRecall**
- Measures how well agent recalls context from earlier messages
- Evaluates retrieval and use of conversation history
- Score: 0.0-1.0

**3. KnowledgeRetention**
- Assesses agent's ability to retain information across turns
- Checks for consistent use of user-provided details
- Score: 0.0-1.0

All scorers are **session-level** (`is_session_level_scorer = True`), meaning they evaluate the entire conversation, not individual turns.

### Conversation Scenarios

**Scenario 1: Account Access**
- 4-turn conversation
- Password reset assistance
- Tests: conversation flow

## Running the Tutorial

### Setup

```bash
# Install dependencies (includes deepeval)
uv sync

# Verify MLflow version (requires 3.8.0+)
uv run python -c "import mlflow; print(mlflow.__version__)"
```

### Running with CLI

```bash
# Run with Databricks
export DATABRICKS_TOKEN='your-token'
export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'
uv run mlflow-deepeval-support

# Run with OpenAI
export OPENAI_API_KEY='sk-...'
uv run mlflow-deepeval-support --provider openai

# Run specific scenario
uv run mlflow-deepeval-support --scenario account

# Use different models
uv run mlflow-deepeval-support \
  --model databricks-gpt-5-2 \
  --judge-model gpt-4o-mini

# Enable debug output
uv run mlflow-deepeval-support --debug

# View results
mlflow ui
```

### Expected Output

```
======================================================================
TUTORIAL: MLflow + DeepEval Integration - Multi-Turn Conversations
======================================================================

[Step 1] MLflow tracing enabled
  └─ Experiment: customer-support-deepeval
  └─ View traces: mlflow ui

[Step 2] Configuring DeepEval environment
  └─ Provider: databricks
  └─ Judge Model: gpt-4o-mini

[Step 3] Initializing Customer Support Agent
  └─ Provider: databricks
  └─ Agent Model: databricks-gpt-5-2
  └─ Temperature: 1.0

[Step 4] Creating DeepEval Scorers (wrapped as MLflow Scorers)
  ✓ Created 3 DeepEval scorers:
    - ConversationCompleteness
    - ContextualRecall
    - KnowledgeRetention

[Step 5] Running Customer Support Conversations

──────────────────────────────────────────────────────────────────────
Scenario: Printer Troubleshooting
Description: User troubleshoots printer issue with good agent support
Session ID: session-printer-001
──────────────────────────────────────────────────────────────────────

======================================================================
Running 4-turn conversation (Session: session-printer-001)
======================================================================

Turn 1/4
  User: My HP LaserJet 3000 won't turn on at all...
  Agent: I'd be happy to help troubleshoot...

[... conversation continues ...]

[Step 6] Evaluating session 'session-printer-001' with DeepEval...
  └─ Running DeepEval evaluation...

======================================================================
DeepEval Evaluation Results: Printer Troubleshooting
======================================================================

📊 Conversation Completeness: 0.85
   Reason: The conversation successfully addressed the user's printer issue...

======================================================================

Tutorial Complete!
======================================================================

✓ Ran 2 conversation scenario(s)
✓ Each conversation evaluated with DeepEval scorers via MLflow

Key Features Demonstrated:
  1. MLflow session tracking
  2. DeepEval integration with mlflow.genai.scorers.deepeval
  3. Industry-standard evaluation metrics
  4. Seamless MLflow + DeepEval workflow

View detailed traces and evaluation results:
  mlflow ui
  → Open 'customer-support-deepeval' experiment
  → Explore traces with DeepEval assessments
```

## Key Concepts

### Why DeepEval with MLflow?

**DeepEval Benefits:**
- Industry-standard evaluation metrics
- Purpose-built conversational metrics
- Active community and updates
- Rich metric library

**MLflow Integration Benefits:**
- Unified evaluation interface
- Session-level evaluation support
- Experiment tracking and comparison
- Results persistence and visualization

### MLflow 3.8 + DeepEval Technical Details

1. **Native Integration**: `mlflow.genai.scorers.deepeval` module
2. **Automatic Session Handling**: Session-level scorers work with MLflow's session tracking
3. **Unified API**: Same `mlflow.genai.evaluate()` call as native scorers
4. **Results Format**: Metrics appear as `MetricName/value` and `MetricName/reason` columns

### Comparison: MLflow Native vs DeepEval Scorers

| Feature | MLflow Native Judges | DeepEval Scorers |
|---------|---------------------|------------------|
| Integration | Built-in | Via `mlflow.genai.scorers.deepeval` |
| Configuration | Template-based (`{{ conversation }}`) | Class-based initialization |
| Metrics | Custom-defined | Pre-built industry metrics |
| Session Support | ✅ Yes | ✅ Yes |
| Customization | High (any prompt) | Moderate (metric parameters) |
| Maintenance | MLflow team | DeepEval + MLflow teams |

## Implementation Example

### 1. Basic Usage

```python
from genai.agents.multi_turn import CustomerSupportAgentSimple
from genai.common.config import AgentConfig
from mlflow.genai.scorers.deepeval import ConversationCompleteness
import mlflow

# Initialize agent
config = AgentConfig(
    provider="openai",
    model="gpt-4o-mini",
    mlflow_experiment="my-support-bot"
)
agent = CustomerSupportAgentSimple(config)

# Run conversation
messages = [
    "My printer isn't working",
    "It's saying paper jam",
    "I checked, no paper stuck"
]

result = agent.run_conversation(
    messages=messages,
    session_id="conv-123"
)

# Create DeepEval scorer
scorer = ConversationCompleteness(model="openai:/gpt-4o-mini")

# Search traces
experiment = mlflow.get_experiment_by_name("my-support-bot")
traces = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    filter_string="metadata.`mlflow.trace.session` = 'conv-123'"
)

# Evaluate
results = mlflow.genai.evaluate(data=traces, scorers=[scorer])

# Extract score
score = results.result_df['ConversationCompleteness/mean'].iloc[0]
print(f"Completeness: {score}")
```


### 3. Using with Databricks

```python
import os

# Configure for Databricks
os.environ["OPENAI_API_KEY"] = os.environ.get("DATABRICKS_TOKEN", "")
os.environ["OPENAI_API_BASE"] = f"{databricks_host}/serving-endpoints"

# Create scorer with Databricks model
scorer = ConversationCompleteness(
    model="openai:/databricks-gpt-5-2"
)
```

## File Structure

```
genai/agents/deepeval/
├── __init__.py                            # Module exports
├── customer_support_deepeval_agent.py     # CLI entry point
└── README.md                              # This file
```

**Shared Components** (from `multi_turn/`):
- `customer_support_agent_simple.py` - Agent for conversations only
- `scenarios.py` - Conversation test cases
- `prompts.py` - System prompts (not judge instructions)

## Customization

### Add More DeepEval Metrics

```python
from mlflow.genai.scorers.deepeval import (
    ConversationCompleteness,
    RoleAdherence,
    TopicAdherence
)

# Add additional metrics
scorers = [
    ConversationCompleteness(model=model_uri),
    RoleAdherence(model=model_uri),
    TopicAdherence(model=model_uri, topics=["customer support", "technical help"])
]

eval_results = mlflow.genai.evaluate(data=traces, scorers=scorers)
```

### Configure Metric Thresholds

```python
# Set custom threshold
completeness_scorer = ConversationCompleteness(
    model="openai:/gpt-4o-mini",
    threshold=0.7  # Pass threshold at 0.7
)
```

### Combine with MLflow Native Judges

```python
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers.deepeval import ConversationCompleteness

# Mix DeepEval and native judges
native_judge = make_judge(
    name="custom_metric",
    instructions="Evaluate {{ conversation }} for politeness",
    feedback_value_type=bool,
    model="openai:/gpt-4o-mini"
)

deepeval_scorer = ConversationCompleteness(model="openai:/gpt-4o-mini")

# Use both
eval_results = mlflow.genai.evaluate(
    data=traces,
    scorers=[native_judge, deepeval_scorer]
)
```

## Version Requirements

- **MLflow >= 3.8.1** (for DeepEval integration)
- **deepeval >= 1.0.0** (for metrics library)
- Python >= 3.10
- OpenAI SDK or Databricks SDK

## Troubleshooting

### "ModuleNotFoundError: No module named 'mlflow.genai.scorers.deepeval'"

Ensure you have MLflow 3.8.0 or later:
```bash
pip install --upgrade "mlflow>=3.8.1"
```

### "cannot import name 'ConversationCompleteness'"

Check available DeepEval scorers:
```python
from mlflow.genai.scorers import deepeval
print(dir(deepeval))
```

### DeepEval scorers return NaN

**Common Causes:**
1. **Session-level scorer on individual traces**: DeepEval conversational metrics need multiple traces grouped as a session
2. **Model configuration**: Ensure model URI is correct (e.g., `"openai:/gpt-4o-mini"`)
3. **API access**: Verify API keys and endpoints are properly configured

**Debug Steps:**
```bash
# Run with debug flag
uv run mlflow-deepeval-support --debug

# Check trace search results
# Verify session_traces contains multiple traces
# Inspect result_df columns
```

### Empty assessments array

This is expected behavior with MLflow 3.8's DeepEval integration. Scores appear in columns like:
- `ConversationCompleteness/value`
- `ConversationCompleteness/reason`
- `ContextualRecall/value`
- etc.

Access them from `result_df` columns, not from `assessments` field.

### "No traces found for session"

Check that:
1. `mlflow.update_current_trace()` is called with session metadata
2. Session ID is consistent across all turns
3. Traces exist in the MLflow experiment

```python
# Verify traces exist
traces = mlflow.search_traces(
    experiment_ids=[experiment_id],
    filter_string="metadata.`mlflow.trace.session` = 'your-session-id'"
)
print(f"Found {len(traces)} traces")
```

## Available DeepEval Metrics

Check MLflow's DeepEval integration for available metrics:

```python
from mlflow.genai.scorers import deepeval

# List all available DeepEval scorers
available_scorers = [x for x in dir(deepeval) if not x.startswith('_')]
print("Available DeepEval scorers:")
for scorer in available_scorers:
    print(f"  - {scorer}")
```

Common metrics include:
- `ConversationCompleteness`
- `ContextualRecall`
- `ContextualPrecision`
- `ContextualRelevancy`
- `KnowledgeRetention`
- `RoleAdherence`
- `TopicAdherence`
- And more...

## Next Steps

1. **Explore More Metrics**: Try additional DeepEval metrics for your use case
2. **Custom Scenarios**: Add domain-specific conversation scenarios
3. **Hybrid Evaluation**: Combine DeepEval + MLflow native judges
4. **Production Integration**: Apply to your chatbot/agent systems
5. **Batch Evaluation**: Evaluate multiple sessions for quality monitoring
6. **Compare Approaches**: Benchmark DeepEval vs native judges for your needs

## References

- [MLflow DeepEval Integration](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/deepeval-scorers/)
- [DeepEval Documentation](https://docs.confident-ai.com/)
- [MLflow Session Tracking](https://mlflow.org/docs/latest/genai/tracing/track-users-sessions/)
- [MLflow GenAI Evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/)