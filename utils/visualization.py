"""
Visualization Utilities

Common utilities for creating plots and visualizations for MLflow experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import os


def create_regression_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_importance: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    save_path: str = "regression_analysis.png",
    figsize: Tuple[int, int] = (15, 10)
) -> str:
    """
    Create comprehensive regression analysis plots.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        feature_importance: Feature importance values
        feature_names: Names of features
        save_path: Path to save the plot
        figsize: Figure size (width, height)
        
    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Predictions vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Predictions vs Actual Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add R² score to the plot
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes, 
                   bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # 2. Residuals plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature importance (if provided)
    if feature_importance is not None and feature_names is not None:
        n_features = min(10, len(feature_importance))  # Top 10 features
        indices = np.argsort(feature_importance)[::-1][:n_features]
        
        axes[1, 0].bar(range(n_features), feature_importance[indices], color='orange')
        axes[1, 0].set_xticks(range(n_features))
        axes[1, 0].set_xticklabels([feature_names[i] for i in indices], rotation=45)
        axes[1, 0].set_title(f'Top {n_features} Feature Importances')
        axes[1, 0].set_ylabel('Importance')
    else:
        axes[1, 0].text(0.5, 0.5, 'Feature importance\nnot available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Feature Importance')
    
    # 4. Residuals histogram
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Residuals')
    axes[1, 1].axvline(x=0, color='r', linestyle='--')
    
    # Add statistics
    axes[1, 1].text(0.05, 0.95, f'Mean: {residuals.mean():.3f}\nStd: {residuals.std():.3f}', 
                   transform=axes[1, 1].transAxes, 
                   bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Regression analysis plot saved: {save_path}")
    return save_path


def create_classification_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    save_path: str = "classification_analysis.png",
    figsize: Tuple[int, int] = (15, 10)
) -> str:
    """
    Create comprehensive classification analysis plots.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (for ROC curve)
        class_names: Names of classes
        save_path: Path to save the plot
        figsize: Figure size (width, height)
        
    Returns:
        Path to saved plot
    """
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. Class Distribution
    unique_classes, counts_true = np.unique(y_true, return_counts=True)
    unique_classes_pred, counts_pred = np.unique(y_pred, return_counts=True)
    
    x = np.arange(len(unique_classes))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, counts_true, width, label='True', alpha=0.7)
    axes[0, 1].bar(x + width/2, counts_pred, width, label='Predicted', alpha=0.7)
    axes[0, 1].set_xlabel('Classes')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Class Distribution')
    axes[0, 1].set_xticks(x)
    if class_names:
        axes[0, 1].set_xticklabels(class_names)
    axes[0, 1].legend()
    
    # 3. ROC Curve (for binary classification)
    if y_proba is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend(loc="lower right")
    else:
        axes[1, 0].text(0.5, 0.5, 'ROC Curve\n(Binary classification only)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # 4. Classification Report (as text)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Create a simple text representation
    axes[1, 1].axis('off')
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    axes[1, 1].text(0.1, 0.9, report_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title('Classification Report')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Classification analysis plot saved: {save_path}")
    return save_path


def create_training_history_plot(
    history: Dict[str, List[float]],
    save_path: str = "training_history.png",
    figsize: Tuple[int, int] = (12, 8)
) -> str:
    """
    Create training history plots for metrics over epochs.
    
    Args:
        history: Dictionary with metric names as keys and lists of values
        save_path: Path to save the plot
        figsize: Figure size (width, height)
        
    Returns:
        Path to saved plot
    """
    n_metrics = len(history)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (metric_name, values) in enumerate(history.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, 'b-', linewidth=2)
        ax.set_title(f'{metric_name.replace("_", " ").title()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training history plot saved: {save_path}")
    return save_path


def create_hyperparameter_comparison_plot(
    results: List[Dict[str, Any]],
    metric_name: str = "test_rmse",
    param_name: str = "n_estimators",
    save_path: str = "hyperparameter_comparison.png",
    figsize: Tuple[int, int] = (10, 6)
) -> str:
    """
    Create hyperparameter vs metric comparison plot.
    
    Args:
        results: List of experiment results with hyperparams and metrics
        metric_name: Name of the metric to plot
        param_name: Name of the hyperparameter to plot
        save_path: Path to save the plot
        figsize: Figure size (width, height)
        
    Returns:
        Path to saved plot
    """
    param_values = []
    metric_values = []
    
    for result in results:
        if param_name in result['hyperparams'] and metric_name in result['metrics']:
            param_values.append(result['hyperparams'][param_name])
            metric_values.append(result['metrics'][metric_name])
    
    plt.figure(figsize=figsize)
    plt.scatter(param_values, metric_values, alpha=0.7, s=100)
    plt.plot(param_values, metric_values, 'r--', alpha=0.5)
    
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} vs {param_name.replace("_", " ").title()}')
    plt.grid(True, alpha=0.3)
    
    # Annotate points
    for i, (x, y) in enumerate(zip(param_values, metric_values)):
        plt.annotate(f'Run {i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Hyperparameter comparison plot saved: {save_path}")
    return save_path


def cleanup_plot_files(*file_paths: str) -> None:
    """
    Clean up temporary plot files.
    
    Args:
        *file_paths: Paths to files to delete
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Cleaned up: {file_path}")
        except Exception as e:
            print(f"⚠️ Could not remove {file_path}: {e}")

