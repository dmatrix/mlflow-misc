"""
Simple MLflow Tracking Example

A minimal example showing the core MLflow workflow:
1. Use one model type
2. Run and log experiments  
3. Find best metric
4. Save the model
5. Load the model
6. Make inference
"""

import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Add utils to path
utils_path = Path(__file__).parent.parent / "utils"
sys.path.insert(0, str(utils_path))

from utils.loader import load_mlflow_setup, load_data_generation


def main():
    """Simple MLflow tracking workflow."""
    print("=" * 40)
    print("Simple MLflow Tracking")
    print("=" * 40)
    
    # 1. Setup MLflow
    mlflow_setup = load_mlflow_setup()
    mlflow_setup.setup_mlflow_tracking(
        experiment_name="simple-house-prediction",
        enable_autolog=True
    )
    
    # 2. Generate data
    data_gen = load_data_generation()
    df, feature_names = data_gen.generate_regression_data(n_samples=500, n_features=5)
    
    X = df[feature_names]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Run experiments with different hyperparameters
    experiments = [
        {'n_estimators': 50, 'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 10}, 
        {'n_estimators': 150, 'max_depth': 15}
    ]
    
    best_rmse = float('inf')
    best_run_id = None
    
    print(f"\nRunning {len(experiments)} experiments...")
    
    for i, params in enumerate(experiments, 1):
        with mlflow.start_run(run_name=f"experiment_{i}"):
            # Train model (autolog handles logging)
            model = RandomForestRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Calculate metrics for comparison (autolog handles logging)
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5  # RMSE
            r2 = r2_score(y_test, y_pred)
            
            print(f"Experiment {i}: RMSE={rmse:.3f}, R²={r2:.3f}")
            
            # Track best model
            if rmse < best_rmse:
                best_rmse = rmse
                best_run_id = mlflow.active_run().info.run_id
    
    # 4. Best model found
    print(f"\n🏆 Best model: Run {best_run_id[:8]} with RMSE={best_rmse:.3f}")
    
    # 5. Load the best model
    print("\n📦 Loading best model...")
    model_uri = f"runs:/{best_run_id}/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Model loaded from: {model_uri}")
    
    # 6. Make inference
    print("\n🔮 Making predictions...")
    sample_data = X_test.iloc[:3]  # Take 3 samples
    sample_targets = y_test.iloc[:3]
    predictions = loaded_model.predict(sample_data)
    
    print("Sample predictions:")
    for i, (predicted, actual) in enumerate(zip(predictions, sample_targets)):
        print(f"  Sample {i+1}: Predicted={predicted:.2f}, Actual={actual:.2f}")
    
    print(f"\n✅ Complete! View results: mlflow ui")
    print("=" * 40)


if __name__ == "__main__":
    main()