"""
Data Generation Utilities

Common utilities for generating synthetic datasets for MLflow experiments.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, make_classification, make_blobs
from typing import Tuple, List, Optional, Dict, Any


def generate_regression_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 8,
    noise: float = 0.1,
    random_state: int = 42,
    feature_prefix: str = "feature"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate synthetic regression dataset.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_informative: Number of informative features
        noise: Standard deviation of gaussian noise
        random_state: Random seed for reproducibility
        feature_prefix: Prefix for feature names
        
    Returns:
        Tuple of (DataFrame with features and target, list of feature names)
    """
    print(f"🔢 Generating regression dataset: {n_samples} samples, {n_features} features")
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f"{feature_prefix}_{i}" for i in range(X.shape[1])]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"✅ Generated dataset shape: {df.shape}")
    return df, feature_names


def generate_classification_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 8,
    n_classes: int = 2,
    random_state: int = 42,
    feature_prefix: str = "feature"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate synthetic classification dataset.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_informative: Number of informative features
        n_classes: Number of classes
        random_state: Random seed for reproducibility
        feature_prefix: Prefix for feature names
        
    Returns:
        Tuple of (DataFrame with features and target, list of feature names)
    """
    print(f"🎯 Generating classification dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f"{feature_prefix}_{i}" for i in range(X.shape[1])]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"✅ Generated dataset shape: {df.shape}")
    return df, feature_names


def generate_clustering_data(
    n_samples: int = 1000,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0,
    random_state: int = 42,
    feature_prefix: str = "feature"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate synthetic clustering dataset.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        centers: Number of cluster centers
        cluster_std: Standard deviation of clusters
        random_state: Random seed for reproducibility
        feature_prefix: Prefix for feature names
        
    Returns:
        Tuple of (DataFrame with features and cluster labels, list of feature names)
    """
    print(f"🎪 Generating clustering dataset: {n_samples} samples, {n_features} features, {centers} centers")
    
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = [f"{feature_prefix}_{i}" for i in range(X.shape[1])]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['cluster'] = y
    
    print(f"✅ Generated dataset shape: {df.shape}")
    return df, feature_names


def load_sample_data(
    dataset_name: str = "boston",
    return_feature_names: bool = True
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """
    Load sample datasets for experimentation.
    
    Args:
        dataset_name: Name of the dataset ('boston', 'iris', 'wine', 'digits')
        return_feature_names: Whether to return feature names
        
    Returns:
        Tuple of (DataFrame, feature names if requested)
    """
    from sklearn.datasets import load_iris, load_wine, load_digits
    
    print(f"📂 Loading {dataset_name} dataset...")
    
    if dataset_name.lower() == "iris":
        data = load_iris()
    elif dataset_name.lower() == "wine":
        data = load_wine()
    elif dataset_name.lower() == "digits":
        data = load_digits()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    feature_names = list(data.feature_names) if return_feature_names else None
    
    print(f"✅ Loaded {dataset_name} dataset shape: {df.shape}")
    return df, feature_names


def add_noise_to_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    noise_level: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Add gaussian noise to specified columns.
    
    Args:
        df: Input DataFrame
        feature_columns: Columns to add noise to
        noise_level: Standard deviation of noise relative to feature std
        random_state: Random seed
        
    Returns:
        DataFrame with noise added
    """
    print(f"🔊 Adding {noise_level} noise to {len(feature_columns)} features")
    
    np.random.seed(random_state)
    df_noisy = df.copy()
    
    for col in feature_columns:
        if col in df.columns:
            noise = np.random.normal(0, df[col].std() * noise_level, len(df))
            df_noisy[col] = df[col] + noise
    
    return df_noisy


def create_train_test_split_info(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Create train/test split information for logging.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Dictionary with split information
    """
    n_total = len(df)
    n_test = int(n_total * test_size)
    n_train = n_total - n_test
    
    return {
        'total_samples': n_total,
        'train_samples': n_train,
        'test_samples': n_test,
        'test_ratio': test_size,
        'train_ratio': 1 - test_size,
        'random_state': random_state
    }

