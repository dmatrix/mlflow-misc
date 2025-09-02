"""
MLflow Spark NYC Taxi Real Data Example

This example demonstrates:
1. Downloading and processing real NYC taxi public dataset
2. RandomForest regression on real-world data
3. Feature engineering with real taxi trip data
4. MLflow tracking with production-scale datasets
5. Data quality checks and preprocessing

Requirements:
- Apache Spark (pyspark)
- MLflow with Spark support
- Internet connection for data download
- ~2GB disk space for dataset
"""

import argparse
import sys
from pathlib import Path
import requests
import os
from typing import Optional

# Add utils to path for imports
sys.path.append(str(Path(__file__).parent.parent / "utils"))

import mlflow
import mlflow.sklearn
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, month, dayofmonth, hour, dayofweek, unix_timestamp
from pyspark.sql.types import DoubleType, IntegerType

# Import utility functions
from loader import load_mlflow_setup
from spark_utils import create_mlflow_spark_session
from .spark_ml_utils import prepare_features, train_random_forest_model, demonstrate_model_inference, log_dataset_info


def download_nyc_taxi_data(data_dir: Path, year: int = 2023, month: int = 1) -> Path:
    """
    Download NYC taxi data from the official NYC TLC website.
    
    Args:
        data_dir: Directory to store downloaded data
        year: Year of data to download
        month: Month of data to download
        
    Returns:
        Path to downloaded parquet file
    """
    data_dir.mkdir(exist_ok=True)
    
    # NYC TLC data URL pattern
    filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
    local_path = data_dir / filename
    
    if local_path.exists():
        print(f"📁 Data already exists: {local_path}")
        return local_path
    
    print(f"📥 Downloading NYC taxi data: {filename}")
    print(f"🔗 URL: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📊 Progress: {progress:.1f}% ({downloaded:,} / {total_size:,} bytes)", end="")
        
        print(f"\n✅ Download completed: {local_path}")
        return local_path
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading data: {e}")
        print("💡 Tip: Check your internet connection or try a different year/month")
        raise


def load_and_clean_taxi_data(spark: SparkSession, data_path: Path, sample_fraction: Optional[float] = None):
    """
    Load and clean NYC taxi data with quality checks.
    
    Args:
        spark: Spark session
        data_path: Path to parquet file
        sample_fraction: Optional fraction to sample (for faster processing)
        
    Returns:
        Cleaned Spark DataFrame
    """
    print(f"📊 Loading NYC taxi data from: {data_path}")
    df = spark.read.parquet(str(data_path))
    
    print(f"📈 Original dataset: {df.count():,} rows, {len(df.columns)} columns")
    
    # Sample data if requested
    if sample_fraction and 0 < sample_fraction < 1:
        df = df.sample(fraction=sample_fraction, seed=42)
        print(f"📉 Sampled dataset: {df.count():,} rows ({sample_fraction*100:.1f}% sample)")
    
    # Show original schema
    print("\n📋 Original Schema:")
    df.printSchema()
    
    # Data quality checks and cleaning
    print("\n🧹 Cleaning data...")
    
    # Remove rows with null values in critical columns
    numeric_cols = ['trip_distance', 'fare_amount', 'tip_amount']
    datetime_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
    
    # Handle numeric columns (check for null and NaN)
    for col_name in numeric_cols:
        if col_name in df.columns:
            initial_count = df.count()
            df = df.filter(col(col_name).isNotNull() & ~isnan(col(col_name)))
            final_count = df.count()
            if initial_count != final_count:
                print(f"  🗑️  Removed {initial_count - final_count:,} rows with null/NaN {col_name}")
    
    # Handle datetime columns (only check for null)
    for col_name in datetime_cols:
        if col_name in df.columns:
            initial_count = df.count()
            df = df.filter(col(col_name).isNotNull())
            final_count = df.count()
            if initial_count != final_count:
                print(f"  🗑️  Removed {initial_count - final_count:,} rows with null {col_name}")
    
    # Remove unrealistic values
    df = df.filter(
        (col('trip_distance') > 0) & (col('trip_distance') < 100) &  # Reasonable trip distance
        (col('fare_amount') > 0) & (col('fare_amount') < 500) &      # Reasonable fare
        (col('tip_amount') >= 0) & (col('tip_amount') < 100) &       # Non-negative tips
        (col('passenger_count') > 0) & (col('passenger_count') <= 6) # Reasonable passenger count
    )
    
    # Calculate trip duration in minutes
    df = df.withColumn(
        'trip_duration_minutes',
        (unix_timestamp('tpep_dropoff_datetime') - unix_timestamp('tpep_pickup_datetime')) / 60
    )
    
    # Remove trips with unrealistic duration
    df = df.filter((col('trip_duration_minutes') > 1) & (col('trip_duration_minutes') < 300))  # 1 min to 5 hours
    
    # Extract time-based features
    df = df.withColumn('pickup_hour', hour('tpep_pickup_datetime')) \
           .withColumn('pickup_day_of_week', dayofweek('tpep_pickup_datetime')) \
           .withColumn('pickup_month', month('tpep_pickup_datetime')) \
           .withColumn('pickup_day', dayofmonth('tpep_pickup_datetime'))
    
    # Calculate speed (mph)
    df = df.withColumn(
        'avg_speed_mph',
        col('trip_distance') / (col('trip_duration_minutes') / 60)
    )
    
    # Remove trips with unrealistic speeds
    df = df.filter((col('avg_speed_mph') > 0) & (col('avg_speed_mph') < 80))  # Max 80 mph
    
    print(f"✅ Cleaned dataset: {df.count():,} rows")
    
    return df








def main():
    """Main execution function."""
    # Set environment variable for Spark autologging compatibility (must be set before Spark session)
    import os
    os.environ['PYSPARK_PIN_THREAD'] = 'false'
    
    parser = argparse.ArgumentParser(description="Spark MLflow NYC Taxi Real Data Example")
    parser.add_argument("--backend", choices=["file", "sqlite"], default="file",
                       help="MLflow backend store type")
    parser.add_argument("--db-path", default="./mlflow.db",
                       help="Database path for SQLite backend")
    parser.add_argument("--tracking-uri", 
                       help="MLflow tracking URI (overrides backend settings)")
    parser.add_argument("--experiment-name", default="nyc-taxi-real-data",
                       help="MLflow experiment name")
    parser.add_argument("--data-dir", default="./data",
                       help="Directory to store downloaded data")
    parser.add_argument("--year", type=int, default=2023,
                       help="Year of NYC taxi data to download")
    parser.add_argument("--month", type=int, default=1,
                       help="Month of NYC taxi data to download")
    parser.add_argument("--sample-fraction", type=float,
                       help="Fraction of data to sample (0.0-1.0) for faster processing")
    parser.add_argument("--local-data", 
                       help="Path to local parquet file (skips download)")
    
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
        enable_autolog=False  # Using manual logging
    )
    
    # Create Spark session
    print("🚀 Initializing Spark Session for NYC Taxi Data...")
    spark = create_mlflow_spark_session("MLflow-Spark-Real-NYC-Taxi")
    
    try:
        # Enable MLflow autologging (for any sklearn components)
        mlflow.sklearn.autolog()
        print("✅ MLflow autologging enabled")
        
        # Download or load data
        if args.local_data and Path(args.local_data).exists():
            data_path = Path(args.local_data)
            print(f"📁 Using local data: {data_path}")
        else:
            data_dir = Path(args.data_dir)
            data_path = download_nyc_taxi_data(data_dir, args.year, args.month)
        
        # Load and clean data
        df = load_and_clean_taxi_data(spark, data_path, args.sample_fraction)
        
        # Show sample data
        print(f"\n🚖 Sample NYC Taxi Data:")
        df.select('trip_distance', 'fare_amount', 'tip_amount', 'passenger_count', 
                 'trip_duration_minutes', 'avg_speed_mph', 'pickup_hour', 'pickup_day_of_week').show(5)
        
        # Prepare features
        print("🔧 Preparing features for tip prediction...")
        prepared_df, feature_cols = prepare_features(df, target_col="tip_amount", return_feature_cols=True)
        
        # Split data
        print("🔄 Splitting data (80% train, 20% test)...")
        train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=42)
        
        print(f"Training set: {train_df.count():,} rows")
        print(f"Test set: {test_df.count():,} rows")
        
        # Train model with MLflow tracking
        with mlflow.start_run():
            # Log dataset info
            log_dataset_info(df, feature_cols, train_df, test_df,
                           dataset_source="NYC TLC Official Data",
                           dataset_year=args.year,
                           dataset_month=args.month,
                           sample_fraction=args.sample_fraction or 1.0,
                           data_file=data_path.name)
            
            model, predictions = train_random_forest_model(
                train_df, test_df, feature_cols, "tip_amount",
                data_source="NYC TLC Public Data",
                model_name="nyc_taxi_rf_model",
                show_feature_importance=True
            )
            
            # Demonstrate inference
            demonstrate_model_inference(model, test_df, "Real NYC Taxi Data")
            
            # Add NYC taxi specific insights
            print(f"\n🗽 NYC Taxi Context:")
            print(f"  • Data represents actual taxi trips in New York City")
            print(f"  • Tips are only recorded for credit card payments")
            print(f"  • Model learns from real passenger tipping behavior")
            print(f"  • Zero-tip rides are common (cash tips not recorded)")
            
            # Log run info
            run = mlflow.active_run()
            print(f"\n✅ MLflow Run completed: {run.info.run_id}")
            print(f"🔗 View in MLflow UI: {mlflow.get_tracking_uri()}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    
    finally:
        # Clean up Spark session
        print("\n🛑 Stopping Spark session...")
        spark.stop()
        print("✅ Spark session stopped")


if __name__ == "__main__":
    main()
