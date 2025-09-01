"""
MLflow Spark RandomForest Regression Example

This example demonstrates:
1. Spark MLflow integration with autologging
2. RandomForest regression on distributed data
3. Feature engineering with VectorAssembler
4. Model training, evaluation, and logging
5. Reusable utility functions for MLflow setup

Requirements:
- Apache Spark (pyspark)
- MLflow with Spark support
- Sample dataset (can use synthetic data if NYC taxi data unavailable)
"""

import argparse
import sys
from pathlib import Path

# Add utils to path for imports
sys.path.append(str(Path(__file__).parent.parent / "utils"))

import mlflow
import mlflow.sklearn
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when, lit
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Import utility functions
from loader import load_mlflow_setup
from spark_utils import create_mlflow_spark_session
from .spark_ml_utils import prepare_features, train_random_forest_model, demonstrate_model_inference, log_dataset_info


def create_synthetic_taxi_data(spark: SparkSession, num_rows: int = 10000):
    """
    Create synthetic taxi-like dataset for demonstration.
    
    Args:
        spark: Spark session
        num_rows: Number of rows to generate
        
    Returns:
        Spark DataFrame with synthetic taxi data
    """
    # Define schema
    schema = StructType([
        StructField("trip_distance", DoubleType(), True),
        StructField("passenger_count", IntegerType(), True),
        StructField("fare_amount", DoubleType(), True),
        StructField("tip_amount", DoubleType(), True),
        StructField("pickup_hour", IntegerType(), True),
        StructField("day_of_week", IntegerType(), True)
    ])
    
    # Generate synthetic data
    df = spark.range(num_rows).select(
        (rand() * 20 + 0.5).alias("trip_distance"),  # 0.5 to 20.5 miles
        (rand() * 5 + 1).cast("int").alias("passenger_count"),  # 1 to 6 passengers
        (rand() * 50 + 5).alias("fare_amount"),  # $5 to $55
        (rand() * 15).alias("base_tip"),  # Base tip amount
        (rand() * 24).cast("int").alias("pickup_hour"),  # 0 to 23 hours
        (rand() * 7).cast("int").alias("day_of_week")  # 0 to 6 days
    )
    
    # Create realistic tip amounts based on fare and other factors
    df = df.withColumn(
        "tip_amount",
        when(col("fare_amount") > 30, col("base_tip") + col("fare_amount") * 0.2)
        .when(col("fare_amount") > 15, col("base_tip") + col("fare_amount") * 0.15)
        .otherwise(col("base_tip") + col("fare_amount") * 0.1)
    ).drop("base_tip")
    
    return df





def main():
    """Main execution function."""
    # Set environment variable for Spark autologging compatibility (must be set before Spark session)
    import os
    os.environ['PYSPARK_PIN_THREAD'] = 'false'
    
    parser = argparse.ArgumentParser(description="Spark MLflow Synthetic NYC Taxi Example")
    parser.add_argument("--backend", choices=["file", "sqlite"], default="file",
                       help="MLflow backend store type")
    parser.add_argument("--db-path", default="./mlflow.db",
                       help="Database path for SQLite backend")
    parser.add_argument("--tracking-uri", 
                       help="MLflow tracking URI (overrides backend settings)")
    parser.add_argument("--experiment-name", default="synthetic-nyc-taxi-regression",
                       help="MLflow experiment name")
    parser.add_argument("--num-rows", type=int, default=500000,
                       help="Number of synthetic data rows to generate")
    parser.add_argument("--data-path", 
                       help="Path to existing Parquet data file (optional)")
    
    args = parser.parse_args()
    
    # End any active MLflow runs
    mlflow.end_run()
    
    # Setup MLflow using utility function
    mlflow_setup_module = load_mlflow_setup()
    
    # Determine tracking URI
    if args.tracking_uri:
        tracking_uri = args.tracking_uri
    elif args.backend == "sqlite":
        db_path = Path(args.db_path).resolve()
        tracking_uri = f"sqlite:///{db_path}"
    else:
        tracking_uri = "file:./mlruns"
    
    # Setup MLflow
    mlflow_setup_module.setup_mlflow_tracking(
        tracking_uri=tracking_uri,
        experiment_name=args.experiment_name,
        enable_autolog=False  # We'll enable Spark autolog separately
    )
    
    # Create Spark session
    print("🚀 Initializing Spark Session for Synthetic NYC Taxi Data...")
    spark = create_mlflow_spark_session("MLflow-Spark-Synthetic-NYC-Taxi")
    
    try:
        # Enable MLflow autologging (for any sklearn components)
        mlflow.sklearn.autolog()
        print("✅ MLflow autologging enabled")
        
        # Load or create data
        if args.data_path and Path(args.data_path).exists():
            print(f"📊 Loading data from {args.data_path}...")
            df = spark.read.parquet(args.data_path)
        else:
            print(f"🎲 Creating synthetic NYC taxi dataset ({args.num_rows:,} rows)...")
            df = create_synthetic_taxi_data(spark, args.num_rows)
        
        print(f"Dataset shape: {df.count():,} rows, {len(df.columns)} columns")
        
        # Show sample data
        print(f"\n🚖 Sample Synthetic NYC Taxi Data:")
        df.select('trip_distance', 'fare_amount', 'tip_amount', 'passenger_count', 
                 'pickup_hour', 'day_of_week').show(5)
        
        # Prepare features
        feature_cols = ["trip_distance", "passenger_count", "fare_amount", "pickup_hour", "day_of_week"]
        print(f"🔧 Preparing features: {feature_cols}")
        
        prepared_df = prepare_features(df, feature_cols, "tip_amount")
        
        # Split data
        print("🔄 Splitting data (80% train, 20% test)...")
        train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)
        
        print(f"Training set: {train_df.count():,} rows")
        print(f"Test set: {test_df.count():,} rows")
        
        # Train model with MLflow tracking
        print("\n🌲 Training RandomForest model...")
        with mlflow.start_run():
            # Log dataset info
            log_dataset_info(df, feature_cols, train_df, test_df,
                           dataset_source="Synthetic Data",
                           num_rows_generated=args.num_rows)
            
            model, predictions = train_random_forest_model(
                train_df, test_df, feature_cols, "tip_amount",
                data_source="Synthetic NYC Taxi Data",
                model_name="synthetic_taxi_rf_model"
            )
            
            # Demonstrate inference
            demonstrate_model_inference(model, test_df, "Synthetic NYC Taxi Data")
            
            # Add synthetic data specific insights
            print(f"\n💡 Synthetic Data Insights:")
            print(f"  • Controlled testing environment for model behavior")
            print(f"  • Features: trip_distance, passenger_count, fare_amount, pickup_hour, day_of_week")
            print(f"  • Generated patterns simulate realistic taxi scenarios")
            
            # Log run info
            run = mlflow.active_run()
            print(f"\n✅ MLflow Run completed: {run.info.run_id}")
            print(f"🔗 View in MLflow UI: {mlflow.get_tracking_uri()}")
    
    finally:
        # Clean up Spark session
        print("\n🛑 Stopping Spark session...")
        spark.stop()
        print("✅ Spark session stopped")


if __name__ == "__main__":
    main()
