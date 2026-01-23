
from mlflow.genai.datasets import get_dataset
import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
# Now fetch the dataset from the MLflow server using the dataset name
dataset = get_dataset(dataset_id="d-129dba30cfd94ef6847cde3458039692")
print(dataset.to_df())

print("--------------------------------")
print(dataset.to_evaluation_dataset())
