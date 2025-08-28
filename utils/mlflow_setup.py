"""
MLflow Setup Utilities

Common utilities for setting up MLflow tracking, experiments, and autolog configuration.
"""

import mlflow
import mlflow.sklearn
from typing import Optional, Dict, Any


def setup_mlflow_tracking(
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "default-experiment",
    enable_autolog: bool = True,
    autolog_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Configure MLflow tracking URI, experiment, and autolog settings.
    
    Args:
        tracking_uri: MLflow tracking URI (default: local file store)
        experiment_name: Name of the experiment to create/use
        enable_autolog: Whether to enable sklearn autolog
        autolog_config: Custom autolog configuration
        
    Returns:
        experiment_id: The ID of the created/existing experiment
    """
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    # Configure autolog if enabled
    if enable_autolog:
        default_autolog_config = {
            'log_input_examples': True,
            'log_model_signatures': True,
            'log_models': True,
            'disable': False,
            'exclusive': False,
            'silent': False
        }
        
        if autolog_config:
            default_autolog_config.update(autolog_config)
            
        mlflow.sklearn.autolog(**default_autolog_config)
        print("✅ MLflow sklearn autolog enabled")
    
    # Set or create experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"📝 Created new experiment: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"📂 Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    print(f"🎯 Active experiment: {experiment_name} (ID: {experiment_id})")
    
    return experiment_id


def setup_experiment_only(experiment_name: str) -> str:
    """
    Set up only the MLflow experiment (no tracking URI or autolog changes).
    
    Args:
        experiment_name: Name of the experiment to create/use
        
    Returns:
        experiment_id: The ID of the created/existing experiment
    """
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"📝 Created new experiment: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"📂 Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    return experiment_id


def enable_autolog_for_framework(
    framework: str = "sklearn",
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Enable autolog for specific ML framework.
    
    Args:
        framework: ML framework ('sklearn', 'tensorflow', 'pytorch', etc.)
        config: Custom autolog configuration
    """
    default_config = {
        'log_input_examples': True,
        'log_model_signatures': True,
        'log_models': True,
        'disable': False,
        'exclusive': False,
        'silent': False
    }
    
    if config:
        default_config.update(config)
    
    if framework.lower() == "sklearn":
        mlflow.sklearn.autolog(**default_config)
    elif framework.lower() == "tensorflow":
        mlflow.tensorflow.autolog(**default_config)
    elif framework.lower() == "pytorch":
        mlflow.pytorch.autolog(**default_config)
    elif framework.lower() == "xgboost":
        mlflow.xgboost.autolog(**default_config)
    elif framework.lower() == "lightgbm":
        mlflow.lightgbm.autolog(**default_config)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    
    print(f"✅ MLflow {framework} autolog enabled")


def get_current_experiment_info() -> Dict[str, Any]:
    """
    Get information about the current active experiment.
    
    Returns:
        Dictionary with experiment information
    """
    experiment = mlflow.get_experiment_by_name(mlflow.get_experiment(None).name)
    
    return {
        'experiment_id': experiment.experiment_id,
        'name': experiment.name,
        'artifact_location': experiment.artifact_location,
        'lifecycle_stage': experiment.lifecycle_stage,
        'creation_time': experiment.creation_time,
        'last_update_time': experiment.last_update_time
    }


def log_environment_info() -> None:
    """Log additional environment and context information."""
    import platform
    import sys
    from datetime import datetime
    
    mlflow.log_params({
        'python_version': sys.version,
        'platform': platform.platform(),
        'mlflow_version': mlflow.__version__,
        'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    print("📊 Environment information logged")

