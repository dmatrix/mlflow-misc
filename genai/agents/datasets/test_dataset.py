import mlflow
from mlflow.genai.datasets import create_dataset
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Suresh Example Wikipedia")

# Create dataset from current experiment
dataset = create_dataset(
    name="suresh_example_wikipedia",
    experiment_id=mlflow.get_experiment_by_name("Suresh Example Wikipedia").experiment_id,
    tags={"stage": "validation", "domain": "wikipedia"},
)

# Create your wikipedia test cases
dataset_wikipedia = pd.DataFrame(

    [
        {
            "inputs": {
                "question": "What is MLflow?",
                "domain": "general",
            },
            "expectations": {
                "expected_answer": "MLflow is an open-source platform for ML",
                "must_mention": ["tracking", "experiments", "models"],
            },
        },
        {
            "inputs": {
                "question": "How do I track experiments?",
                "domain": "technical",
            },
            "expectations": {
                "expected_answer": "Use mlflow.start_run() and mlflow.log_params()",
                "must_mention": ["log_params", "log_metrics"],
            },
        },
        {
            "inputs": {
                "question": "Explain model versioning",
                "domain": "technical",
            },
            "expectations": {
                "expected_answer": "Model Registry provides versioning",
                "must_mention": ["Model Registry", "versions"],
            },
        },
    ]
)
dataset.merge_records(dataset_wikipedia)