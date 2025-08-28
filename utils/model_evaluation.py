"""
Model Evaluation Utilities

Common utilities for evaluating machine learning models and logging metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from typing import Dict, Any, List, Optional, Tuple
import mlflow
from datetime import datetime


def evaluate_regression_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    log_to_mlflow: bool = True,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Comprehensive evaluation of regression models.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        log_to_mlflow: Whether to log metrics to MLflow
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        f'{prefix}train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        f'{prefix}test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        f'{prefix}train_mae': mean_absolute_error(y_train, y_pred_train),
        f'{prefix}test_mae': mean_absolute_error(y_test, y_pred_test),
        f'{prefix}train_r2': r2_score(y_train, y_pred_train),
        f'{prefix}test_r2': r2_score(y_test, y_pred_test),
    }
    
    # Additional derived metrics
    metrics[f'{prefix}overfitting_ratio'] = (
        metrics[f'{prefix}train_rmse'] / metrics[f'{prefix}test_rmse'] 
        if metrics[f'{prefix}test_rmse'] > 0 else 1.0
    )
    
    metrics[f'{prefix}rmse_difference'] = abs(
        metrics[f'{prefix}train_rmse'] - metrics[f'{prefix}test_rmse']
    )
    
    metrics[f'{prefix}r2_difference'] = abs(
        metrics[f'{prefix}train_r2'] - metrics[f'{prefix}test_r2']
    )
    
    # Log to MLflow if requested
    if log_to_mlflow:
        mlflow.log_metrics(metrics)
        print(f"📊 Logged {len(metrics)} regression metrics to MLflow")
    
    return metrics


def evaluate_classification_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    average: str = 'weighted',
    log_to_mlflow: bool = True,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Comprehensive evaluation of classification models.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        average: Averaging strategy for multi-class metrics
        log_to_mlflow: Whether to log metrics to MLflow
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate basic metrics
    metrics = {
        f'{prefix}train_accuracy': accuracy_score(y_train, y_pred_train),
        f'{prefix}test_accuracy': accuracy_score(y_test, y_pred_test),
        f'{prefix}train_precision': precision_score(y_train, y_pred_train, average=average, zero_division=0),
        f'{prefix}test_precision': precision_score(y_test, y_pred_test, average=average, zero_division=0),
        f'{prefix}train_recall': recall_score(y_train, y_pred_train, average=average, zero_division=0),
        f'{prefix}test_recall': recall_score(y_test, y_pred_test, average=average, zero_division=0),
        f'{prefix}train_f1': f1_score(y_train, y_pred_train, average=average, zero_division=0),
        f'{prefix}test_f1': f1_score(y_test, y_pred_test, average=average, zero_division=0),
    }
    
    # Add ROC AUC for binary classification
    if len(np.unique(y_test)) == 2:
        try:
            if hasattr(model, 'predict_proba'):
                y_proba_train = model.predict_proba(X_train)[:, 1]
                y_proba_test = model.predict_proba(X_test)[:, 1]
                metrics[f'{prefix}train_auc'] = roc_auc_score(y_train, y_proba_train)
                metrics[f'{prefix}test_auc'] = roc_auc_score(y_test, y_proba_test)
        except Exception as e:
            print(f"⚠️ Could not calculate AUC: {e}")
    
    # Additional derived metrics
    metrics[f'{prefix}accuracy_difference'] = abs(
        metrics[f'{prefix}train_accuracy'] - metrics[f'{prefix}test_accuracy']
    )
    
    # Log to MLflow if requested
    if log_to_mlflow:
        mlflow.log_metrics(metrics)
        print(f"📊 Logged {len(metrics)} classification metrics to MLflow")
    
    return metrics


def compare_model_results(
    results: List[Dict[str, Any]],
    metric_name: str = "test_rmse",
    ascending: bool = True
) -> pd.DataFrame:
    """
    Compare results from multiple model runs.
    
    Args:
        results: List of result dictionaries
        metric_name: Metric to sort by
        ascending: Sort order (True for ascending)
        
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for i, result in enumerate(results):
        row = {'run_id': result.get('run_id', f'run_{i+1}')}
        
        # Add hyperparameters
        if 'hyperparams' in result:
            row.update(result['hyperparams'])
        
        # Add metrics
        if 'metrics' in result:
            row.update(result['metrics'])
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by specified metric if it exists
    if metric_name in df.columns:
        df = df.sort_values(metric_name, ascending=ascending)
        df = df.reset_index(drop=True)
    
    return df


def log_model_summary(
    model: Any,
    hyperparams: Dict[str, Any],
    metrics: Dict[str, float],
    feature_names: List[str],
    run_id: str,
    model_type: str = "Unknown"
) -> str:
    """
    Create and log a comprehensive model summary.
    
    Args:
        model: Trained model
        hyperparams: Model hyperparameters
        metrics: Evaluation metrics
        feature_names: List of feature names
        run_id: MLflow run ID
        model_type: Type of model
        
    Returns:
        Path to the summary file
    """
    summary_file = f"model_summary_{run_id[:8]}.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"=== MLflow Model Summary ===\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run ID: {run_id}\n\n")
        
        f.write(f"Hyperparameters:\n")
        for key, value in hyperparams.items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\nEvaluation Metrics:\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.6f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            f.write(f"\nFeature Importance (Top 10):\n")
            importance = model.feature_importances_
            top_indices = np.argsort(importance)[::-1][:10]
            for idx in top_indices:
                f.write(f"  {feature_names[idx]}: {importance[idx]:.6f}\n")
        
        f.write(f"\nModel Parameters:\n")
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
    
    # Log as artifact
    mlflow.log_artifact(summary_file)
    print(f"📄 Model summary logged: {summary_file}")
    
    return summary_file


def calculate_feature_statistics(
    X: np.ndarray,
    feature_names: List[str],
    log_to_mlflow: bool = True
) -> Dict[str, float]:
    """
    Calculate and log feature statistics.
    
    Args:
        X: Feature matrix
        feature_names: Names of features
        log_to_mlflow: Whether to log to MLflow
        
    Returns:
        Dictionary of feature statistics
    """
    stats = {}
    
    for i, name in enumerate(feature_names):
        feature_data = X[:, i] if X.ndim > 1 else X
        stats[f'feature_{name}_mean'] = float(np.mean(feature_data))
        stats[f'feature_{name}_std'] = float(np.std(feature_data))
        stats[f'feature_{name}_min'] = float(np.min(feature_data))
        stats[f'feature_{name}_max'] = float(np.max(feature_data))
        stats[f'feature_{name}_median'] = float(np.median(feature_data))
    
    # Overall statistics
    stats['n_features'] = len(feature_names)
    stats['n_samples'] = X.shape[0]
    
    if log_to_mlflow:
        mlflow.log_params(stats)
        print(f"📊 Logged feature statistics for {len(feature_names)} features")
    
    return stats


def log_dataset_info(
    df: pd.DataFrame,
    target_column: str = 'target',
    log_to_mlflow: bool = True
) -> Dict[str, Any]:
    """
    Log comprehensive dataset information.
    
    Args:
        df: Dataset DataFrame
        target_column: Name of target column
        log_to_mlflow: Whether to log to MLflow
        
    Returns:
        Dictionary of dataset information
    """
    info = {
        'dataset_shape_rows': df.shape[0],
        'dataset_shape_cols': df.shape[1],
        'dataset_memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'missing_values_total': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
    }
    
    # Target statistics if target column exists
    if target_column in df.columns:
        target_data = df[target_column]
        info.update({
            f'{target_column}_mean': float(target_data.mean()),
            f'{target_column}_std': float(target_data.std()),
            f'{target_column}_min': float(target_data.min()),
            f'{target_column}_max': float(target_data.max()),
            f'{target_column}_unique_values': int(target_data.nunique()),
        })
    
    # Feature column info
    feature_cols = [col for col in df.columns if col != target_column]
    info['n_feature_columns'] = len(feature_cols)
    
    if log_to_mlflow:
        mlflow.log_params(info)
        print(f"📊 Logged dataset information")
    
    return info

