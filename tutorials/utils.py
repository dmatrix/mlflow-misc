from databricks.sdk import WorkspaceClient
from openai import OpenAI
from langchain_openai import ChatOpenAI
from databricks_langchain import ChatDatabricks
import os


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

def is_databricks_client():
    """Check if the client will a Databricks provider."""
    return os.environ.get("USE_DATABRICKS_CLIENT") == "True"

def is_openai_client():
    """Check if the client will a OpenAI provider."""
    return os.environ.get("USE_OPENAI_CLIENT") == "True"

def get_langchain_chat_openai_client(model_name: str, temperature: float = 1.0):
    """Initialize LangChain ChatOpenAI client."""
    return ChatOpenAI(model=model_name, temperature=temperature)

def get_databricks_langchain_chat_client(model_name: str, temperature: float = 1.0):
    """Initialize LangChain ChatDatabricks client."""
    return ChatDatabricks(endpoint=model_name, temperature=temperature)

def main():
    """Main function."""
    databricks_provider = is_databricks_client()
    print("✅ Using", "Databricks" if databricks_provider else "OpenAI", "as provider")
    if databricks_provider:
        llm = get_databricks_langchain_chat_client(model_name="databricks-gpt-5-2", temperature=1.0)
    else:
        llm = get_langchain_chat_openai_client(model_name="gpt-5-2i", temperature=1.0)
    print(llm.invoke({"role": "user", "content": "What is the capital of France?"}))

if __name__ == "__main__":
    main()