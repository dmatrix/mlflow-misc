import os
from databricks.sdk import WorkspaceClient

DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST')
workspace_client = WorkspaceClient(profile="DEFAULT",
    host=DATABRICKS_HOST,
    token=DATABRICKS_TOKEN
)
openai_client = workspace_client.serving_endpoints.get_open_ai_client()

response = openai_client.chat.completions.create(
    model="databricks-gpt-5",
    messages=[
        {
            "role": "user",
            "content": "What is an LLM agent?"
        }
    ],
    max_tokens=5000
)

print(response.choices[0].message.content)