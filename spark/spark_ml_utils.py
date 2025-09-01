"""
Spark ML Utilities for MLflow Integration

This module provides common ML functions used across Spark MLflow examples,
including model training, evaluation, and inference demonstration.
"""

import mlflow
import mlflow.spark
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from typing import Optional


def prepare_features(df, feature_cols: list = None, target_col: str = "tip_amount", return_feature_cols: bool = False):
    """
    Prepare features using VectorAssembler.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names. If None, will auto-detect for NYC taxi data
        target_col: Target column name
        return_feature_cols: Whether to return feature column names along with DataFrame
        
    Returns:
        DataFrame with features vector and target column, optionally with feature_cols list
    """
    from pyspark.sql.functions import col
    from pyspark.sql.types import DoubleType
    
    # Auto-detect feature columns for NYC taxi data if not provided
    if feature_cols is None:
        feature_cols = [
            'trip_distance',
            'fare_amount', 
            'passenger_count',
            'trip_duration_minutes',
            'avg_speed_mph',
            'pickup_hour',
            'pickup_day_of_week',
            'pickup_month',
            'pickup_day'
        ]
        
        # Ensure all feature columns exist and are numeric
        available_feature_cols = []
        for col_name in feature_cols:
            if col_name in df.columns:
                # Cast to double to ensure numeric type
                df = df.withColumn(col_name, col(col_name).cast(DoubleType()))
                available_feature_cols.append(col_name)
            else:
                print(f"⚠️  Warning: Feature column '{col_name}' not found in data")
        
        feature_cols = available_feature_cols
        print(f"🔧 Using features: {feature_cols}")
    
    # Assemble features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_assembled = assembler.transform(df).select("features", target_col)
    
    if return_feature_cols:
        return df_assembled, feature_cols
    else:
        return df_assembled


def train_random_forest_model(
    train_df, 
    test_df, 
    feature_cols: list, 
    target_col: str = "tip_amount",
    data_source: str = "Unknown",
    model_name: str = "random_forest_model",
    num_trees: int = 50,
    max_depth: int = 15,
    min_instances_per_node: int = 10,
    seed: int = 42,
    show_feature_importance: bool = False
):
    """
    Train RandomForest model with MLflow logging.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        data_source: Description of data source for logging
        model_name: Name for the logged model
        num_trees: Number of trees in the forest
        max_depth: Maximum depth of trees
        min_instances_per_node: Minimum instances per node
        seed: Random seed for reproducibility
        show_feature_importance: Whether to display and log feature importance
        
    Returns:
        Tuple of (trained_model, predictions_dataframe)
    """
    # Configure RandomForest
    rf = RandomForestRegressor(
        labelCol=target_col,
        featuresCol="features",
        numTrees=num_trees,
        maxDepth=max_depth,
        minInstancesPerNode=min_instances_per_node,
        seed=seed
    )
    
    # Log additional context (autolog will handle model parameters)
    mlflow.log_params({
        "feature_count": len(feature_cols),
        "features": ", ".join(feature_cols),
        "data_source": data_source
    })
    
    print(f"🌲 Training RandomForest model on {data_source.lower()}...")
    
    # Train model (autolog will capture parameters and model)
    model = rf.fit(train_df)
    
    # Make predictions
    predictions = model.transform(test_df)
    
    # Evaluate model
    evaluator = RegressionEvaluator(
        labelCol=target_col,
        predictionCol="prediction",
        metricName="rmse"
    )
    
    rmse = evaluator.evaluate(predictions)
    
    # Calculate additional metrics
    evaluator.setMetricName("r2")
    r2 = evaluator.evaluate(predictions)
    
    evaluator.setMetricName("mae")
    mae = evaluator.evaluate(predictions)
    
    # Log model parameters manually (Spark ML models need manual logging)
    mlflow.log_params({
        "numTrees": rf.getNumTrees(),
        "maxDepth": rf.getMaxDepth(),
        "minInstancesPerNode": rf.getMinInstancesPerNode(),
        "seed": rf.getSeed()
    })
    
    # Log custom metrics
    mlflow.log_metrics({
        "test_rmse": rmse,
        "test_r2": r2,
        "test_mae": mae
    })
    
    # Log the Spark ML model manually
    mlflow.spark.log_model(model, model_name)
    
    print(f"📊 Model Performance on {data_source}:")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: ${mae:.2f}")
    
    # Feature importance (if requested)
    if show_feature_importance:
        try:
            feature_importance = model.featureImportances.toArray()
            importance_dict = {f"feature_importance_{feature_cols[i]}": float(feature_importance[i]) 
                              for i in range(len(feature_cols))}
            
            print(f"\n🎯 Feature Importance:")
            for i, feature in enumerate(feature_cols):
                importance = feature_importance[i]
                print(f"  {feature}: {importance:.4f}")
            
            # Log feature importance as parameters
            mlflow.log_params(importance_dict)
            
        except Exception as e:
            print(f"⚠️  Could not extract feature importance: {e}")
    
    return model, predictions


def demonstrate_model_inference(
    model, 
    test_df, 
    data_source: str = "Data",
    num_samples: int = 5
):
    """
    Demonstrate model inference with detailed output.
    
    Args:
        model: Trained model
        test_df: Test DataFrame
        data_source: Description of data source for display
        num_samples: Number of samples to show
    """
    # Get sample predictions
    sample_predictions = model.transform(test_df.limit(num_samples))
    
    print(f"\n🔮 Model Inference Demonstration")
    print(f"=" * 50)
    
    # Show raw predictions table
    print(f"\n🚖 Sample Predictions on {data_source} (first {num_samples} rows):")
    sample_predictions.select("features", "tip_amount", "prediction").show(truncate=False)
    
    # Collect data for detailed analysis
    predictions_data = sample_predictions.select("tip_amount", "prediction").collect()
    
    print(f"\n📊 Detailed Prediction Analysis:")
    print(f"{'#':<3} {'Actual Tip':<12} {'Predicted Tip':<14} {'Difference':<12} {'Error %':<10}")
    print(f"{'-'*3:<3} {'-'*12:<12} {'-'*14:<14} {'-'*12:<12} {'-'*10:<10}")
    
    total_error = 0
    total_actual = 0
    for i, row in enumerate(predictions_data, 1):
        actual = row['tip_amount']
        predicted = row['prediction']
        difference = predicted - actual
        error_pct = abs(difference / actual * 100) if actual != 0 else 0
        
        total_error += abs(difference)
        total_actual += actual
        
        print(f"{i:<3} ${actual:<11.2f} ${predicted:<13.2f} ${difference:<11.2f} {error_pct:<9.1f}%")
    
    # Summary statistics
    avg_error = total_error / len(predictions_data)
    avg_actual = total_actual / len(predictions_data)
    
    print(f"\n📈 Inference Summary:")
    print(f"  • Average Actual Tip: ${avg_actual:.2f}")
    print(f"  • Average Absolute Error: ${avg_error:.2f}")
    print(f"  • Sample Size: {len(predictions_data)} predictions")
    print(f"  • Model Type: RandomForest Regression")
    print(f"  • Data Source: {data_source}")
    
    # Show prediction quality insights
    accurate_predictions = sum(1 for row in predictions_data 
                             if abs(row['prediction'] - row['tip_amount']) / max(row['tip_amount'], 0.01) < 0.5)
    accuracy_rate = accurate_predictions / len(predictions_data) * 100
    
    print(f"\n💡 Prediction Quality Insights:")
    print(f"  • {accurate_predictions}/{len(predictions_data)} predictions within 50% of actual ({accuracy_rate:.1f}%)")
    
    return predictions_data


def get_default_rf_params():
    """
    Get default RandomForest parameters optimized for taxi tip prediction.
    
    Returns:
        Dictionary of default parameters
    """
    return {
        "num_trees": 50,
        "max_depth": 15,
        "min_instances_per_node": 10,
        "seed": 42
    }


def log_dataset_info(df, feature_cols: list, train_df, test_df, **additional_params):
    """
    Log common dataset information to MLflow.
    
    Args:
        df: Original DataFrame
        feature_cols: List of feature columns
        train_df: Training DataFrame
        test_df: Test DataFrame
        **additional_params: Additional parameters to log
    """
    base_params = {
        "dataset_rows": df.count(),
        "feature_columns": len(feature_cols),
        "train_rows": train_df.count(),
        "test_rows": test_df.count()
    }
    
    # Merge with additional parameters
    base_params.update(additional_params)
    
    mlflow.log_params(base_params)
