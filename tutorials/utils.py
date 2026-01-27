from databricks.sdk import WorkspaceClient
import os
from openai import OpenAI

def get_workspace_client():
    """Initialize Databricks workspace client."""
    databricks_token = os.environ.get("DATABRICKS_TOKEN")
    databricks_host = os.environ.get("DATABRICKS_HOST")
    return WorkspaceClient(
        profile="DEFAULT",
        host=databricks_host,
        token=databricks_token,
    )

def get_databricks_client():
    """Initialize Databricks client."""
    return get_workspace_client().serving_endpoints.get_open_ai_client()

def get_openai_client():
    """Initialize OpenAI client."""
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))