# Spark MLflow Integration Examples

Educational examples demonstrating MLflow integration with Apache Spark for distributed machine learning workflows.

## Overview

Apache Spark provides a unified analytics engine for large-scale data processing. These examples show how to integrate MLflow with Spark for experiment tracking, model versioning, and distributed ML workflows.

## Examples

### `spark_synthetic_nyc_taxi_data.py`

Synthetic data example with configurable dataset size:

- RandomForest regression with Spark MLlib
- Synthetic NYC taxi-like dataset generation (500K rows default)
- Feature engineering with VectorAssembler
- MLflow hybrid logging (sklearn autolog + manual Spark ML logging)
- Model evaluation with RMSE, R², and MAE metrics
- Integration with reusable utility modules
- Support for file store and SQLite backends

### `spark_real_nyc_taxi_data.py`

Real-world data example using NYC taxi public dataset:

- Automatic download of NYC TLC public data (47MB+ files, 3M+ records)
- Data quality checks and preprocessing for real-world messiness
- Configurable data sampling for memory management
- Advanced feature engineering (time-based, speed, duration)
- Feature importance analysis for model interpretability
- Demonstrates memory management challenges with large datasets
- Educational focus on real data processing patterns

**⚠️ Memory Management**: The full NYC dataset contains millions of records. For local development, use `--sample-fraction 0.01` (1%) or `0.05` (5%) to avoid memory issues. On a distributed Spark cluster, you can process the full dataset.

### `spark_langchain_simple.py`

Simplified NLP example integrating Spark, LangChain, and MLflow:

- **Clean Integration**: Focused on core Spark + LangChain + MLflow workflow
- **Distributed Processing**: Uses Spark UDFs for true distributed text processing across cluster
- **Sentiment Analysis**: Simple LLM-powered text classification using LangChain
- **MLflow Autologging**: Automatic experiment tracking with minimal manual logging
- **Utility Functions**: Uses existing project utilities for clean, maintainable code
- **Mock LLM Support**: Works without OpenAI API key for development and testing
- **Educational Focus**: Streamlined example without complex feature engineering
- **No TF-IDF**: Modern LLM approach without traditional NLP preprocessing
- **Production Ready**: UDF-based architecture scales to large datasets

**🎯 Simplified Approach**: Demonstrates the essential integration patterns without the complexity of traditional feature engineering, making it perfect for learning the core concepts.

**⚡ UDF Architecture**: Uses Spark User Defined Functions (UDFs) to distribute LangChain sentiment analysis across the cluster, enabling processing of large text datasets efficiently.

## Identical Logic Structure

Both examples share **identical logic structure** with only the data source being different:

✅ **Same Spark Configuration**: Identical session setup with optimized settings  
✅ **Same RandomForest Parameters**: 50 trees, depth 15, minInstancesPerNode 10  
✅ **Same MLflow Integration**: Hybrid logging approach (sklearn autolog + manual Spark ML)  
✅ **Same Feature Engineering**: VectorAssembler with consistent feature preparation  
✅ **Same Training Function**: Identical `train_random_forest_model()` signatures  
✅ **Same Evaluation Metrics**: RMSE, R², MAE logged in same format  
✅ **Same Model Logging**: Manual Spark ML parameter and model logging  
✅ **Same Output Format**: Consistent display with emojis and styling  

This design makes it easy to compare synthetic vs real data processing patterns while learning the same MLflow and Spark concepts.

## Quick Start

### Prerequisites
```bash
# Install dependencies (Spark only)
uv sync --extra spark

# Install with LangChain support
uv sync --extra spark --extra langchain

# Verify installation
uv run python -c "import pyspark, mlflow; print('✅ All dependencies ready')"
```

### Run Examples

#### **Synthetic Data Example**
```bash
# Basic run with default settings (500K rows)
uv run mlflow-spark-synthetic

# Custom configuration
uv run mlflow-spark-synthetic --num-rows 100000 --experiment-name "my-synthetic-test"
```

#### **Real NYC Taxi Data Example**
```bash
# Small sample for testing (recommended for local development)
uv run mlflow-spark-nyc-taxi --sample-fraction 0.01

# Custom configuration
uv run mlflow-spark-nyc-taxi --sample-fraction 0.005 --experiment-name "nyc-taxi-test"
```

#### **LangChain Sentiment Analysis Example**
```bash
# Basic run with mock LLM (no API key required)
uv run mlflow-spark-langchain --num-samples 50

# With OpenAI API (requires OPENAI_API_KEY environment variable)
export OPENAI_API_KEY="your-api-key-here"
uv run mlflow-spark-langchain --num-samples 100 --use-openai --experiment-name "sentiment-openai"
```

### View Results
```bash
# Start MLflow UI
uv run mlflow ui 

# Open browser to http://localhost:5000
```

## Utility Modules

### `spark_ml_utils.py` - Common ML Functions

This module contains shared functions used by both examples:

- **`prepare_features()`**: Unified feature preparation with VectorAssembler
  - Auto-detects NYC taxi features when `feature_cols=None`
  - Supports explicit feature lists for synthetic data
  - Automatic type casting to DoubleType
  - Column validation with warnings
  
- **`train_random_forest_model()`**: Common RandomForest training
  - Configurable parameters (trees, depth, etc.)
  - MLflow parameter and metric logging
  - Feature importance extraction (optional)
  - Consistent model performance display
  
- **`demonstrate_model_inference()`**: Standardized inference demo
  - Detailed prediction analysis table
  - Error calculation and statistics
  - Data source-specific insights
  - Professional formatted output
  
- **`log_dataset_info()`**: Consistent dataset logging
  - Standard parameters (rows, features, splits)
  - Extensible with additional metadata

## Synthetic vs Real Data: Key Differences

### Synthetic Data Example (`spark_synthetic_nyc_taxi_data.py`)

**Purpose**: Learning Spark MLflow integration without external dependencies

**Characteristics**:
- **Data Generation**: Creates clean, predictable taxi-like data in-memory
- **Dataset Size**: Configurable (default 500K rows), generated instantly
- **Data Quality**: Perfect - no missing values, outliers, or inconsistencies
- **Performance**: Consistent and predictable across runs
- **Memory Usage**: Controlled and scalable based on `--num-rows` parameter
- **Use Case**: Learning Spark concepts, testing configurations, development

**Typical Results**:
- RMSE: ~$4.43, R²: ~0.26, MAE: ~$3.82
- Execution time: ~15-20 seconds for 500K rows
- Memory usage: Moderate and predictable

### Real Data Example (`spark_real_nyc_taxi_data.py`)

**Purpose**: Understanding real-world data challenges and production patterns

**Characteristics**:
- **Data Source**: NYC TLC official public dataset (47MB+ files)
- **Dataset Size**: 3+ million records per month, requires download
- **Data Quality**: Real-world messiness - missing values, outliers, invalid records
- **Performance**: Variable based on data quality and system resources
- **Memory Usage**: Significant - requires sampling for local development
- **Use Case**: Learning data preprocessing, handling production-scale challenges

**Typical Results** (0.1% sample):
- RMSE: ~$2.95, R²: ~0.33, MAE: ~$1.93
- Execution time: ~30-45 seconds (including data cleaning)
- Memory usage: High - can cause OOM errors without sampling

### When to Use Which

**Use Synthetic Data When**:
- Learning Spark MLflow basics
- Testing different configurations
- Developing on resource-constrained machines
- Need consistent, reproducible results
- Focusing on MLflow tracking concepts

**Use Real Data When**:
- Learning data preprocessing techniques
- Understanding production challenges
- Practicing memory management
- Working with feature engineering on messy data
- Preparing for real-world ML projects

## Quick Start

### Prerequisites

Install Spark dependencies:

```bash
# Install with Spark support
uv sync --extra spark

# Or install PySpark directly
pip install pyspark>=3.4.0
```

### Basic Usage

```bash
# Run synthetic data example with default settings
uv run mlflow-spark-synthetic

# Run with SQLite backend
python spark/spark_synthetic_nyc_taxi_data.py --backend sqlite --db-path ./spark_mlflow.db

# Run with custom experiment name and more data
python spark/spark_synthetic_nyc_taxi_data.py \
    --experiment-name "taxi-tip-prediction" \
    --num-rows 1000000

# Use existing Parquet data
python spark/spark_synthetic_nyc_taxi_data.py \
    --data-path /path/to/your/data.parquet

# Run NYC taxi real data example
uv run mlflow-spark-nyc-taxi

# NYC taxi with custom year/month
python spark/spark_real_nyc_taxi_data.py \
    --year 2023 --month 6 \
    --experiment-name "nyc-taxi-june-2023"

# NYC taxi with data sampling for local development
python spark/spark_real_nyc_taxi_data.py \
    --sample-fraction 0.01 \
    --experiment-name "nyc-taxi-1percent-sample"

# Recommended maximum for MacBook Pro (tested safe limit)
python spark/spark_real_nyc_taxi_data.py \
    --sample-fraction 0.01 \
    --experiment-name "nyc-taxi-optimal-sample"
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--backend` | Backend store type (`file`, `sqlite`) | `file` |
| `--db-path` | SQLite database path | `./mlflow.db` |
| `--tracking-uri` | Custom MLflow tracking URI | Auto-determined |
| `--experiment-name` | MLflow experiment name | `synthetic-nyc-taxi-regression` |
| `--num-rows` | Synthetic data rows to generate | `500000` |
| `--data-path` | Path to existing Parquet data | None (uses synthetic) |

## What Gets Logged

MLflow hybrid logging (sklearn autolog + manual Spark ML) captures:

### Model Artifacts
- Spark ML RandomForest model with transformers and estimators
- Model serialization for deployment and inference
- Model metadata and configuration

### Metrics
- Test set evaluation metrics (RMSE, R², MAE)
- Model performance statistics
- Custom evaluation results

### Parameters
- Model hyperparameters (numTrees, maxDepth, seed, etc.)
- Dataset information (rows, features, splits)
- Training configuration (label column, features column)

## Synthetic Data

The example generates realistic taxi-like data with:

- Trip distance (0.5 to 20.5 miles)
- Passenger count (1 to 6 passengers)  
- Fare amount ($5 to $55)
- Pickup hour (0 to 23)
- Day of week (0 to 6)
- Tip amount (calculated based on fare and other factors)

## Model Performance

The RandomForest model typically achieves:

**Synthetic Data**:
- RMSE: ~$4.43, R²: ~0.26, MAE: ~$3.82

**Real NYC Taxi Data** (0.1% sample):
- RMSE: ~$2.95, R²: ~0.33, MAE: ~$1.93

## MLflow UI

View your Spark experiments in the MLflow UI:

```bash
# For file store
mlflow ui --backend-store-uri file:./mlruns

# For SQLite store  
mlflow ui --backend-store-uri sqlite:///$(pwd)/spark_mlflow.db --port 5001
```

Navigate to `http://localhost:5000` (or `:5001` for SQLite) to explore:

- Experiment runs with all logged parameters and metrics
- Model artifacts including the full Spark pipeline
- Model comparison across different hyperparameters
- Feature importance visualizations

## Spark Configuration

The example includes optimized Spark settings:

```python
SparkSession.builder \
    .appName("MLflow-Spark-RandomForest") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()
```

These settings enable:
- Adaptive Query Execution for better performance
- Automatic partition coalescing to reduce small files
- Optimized resource utilization

## Integration with Utils

The examples leverage utility functions from multiple sources:

### Global Utils (`../utils/`)
- `load_mlflow_setup()`: Dynamic MLflow configuration
- `create_mlflow_spark_session()`: Optimized Spark session creation

### Local Spark Utils (`spark_ml_utils.py`)
- `prepare_features()`: Unified feature preparation with VectorAssembler
- `train_random_forest_model()`: Common RandomForest training with configurable parameters
- `demonstrate_model_inference()`: Standardized inference demonstration with detailed output
- `log_dataset_info()`: Consistent dataset information logging

### Benefits of Refactored Architecture
- **DRY Principle**: Eliminated ~320+ lines of duplicate code
- **Single Source of Truth**: Common ML logic maintained in one place
- **Flexible Interface**: Functions adapt to different data sources (synthetic vs real)
- **Consistent Behavior**: Identical training and evaluation logic across examples
- **Easy Maintenance**: Updates to ML logic automatically apply to all examples

## Learning Path

1. Start with the synthetic data example (`spark_synthetic_nyc_taxi_data.py`)
2. Experiment with different hyperparameters and data sizes
3. Try the real NYC taxi data example (`spark_real_nyc_taxi_data.py`)
4. Use your own Parquet files with `--data-path`
5. Scale up: The default 500K rows demonstrate Spark's distributed processing
6. Adapt the patterns for your specific learning projects

## Troubleshooting

### Memory Issues with NYC Taxi Data

#### **MacBook Pro Environment (Tested Configuration)**

Based on extensive memory testing with the real NYC taxi dataset:

**✅ Safe Sample Fractions**:
```bash
# Conservative (recommended for all MacBook Pro users)
python spark/spark_real_nyc_taxi_data.py --sample-fraction 0.01    # 1.0% = ~30K rows

# Optimal performance (maximum recommended limit)
python spark/spark_real_nyc_taxi_data.py --sample-fraction 0.01    # 1.0% = ~30K rows

# Quick testing
python spark/spark_real_nyc_taxi_data.py --sample-fraction 0.005   # 0.5% = ~15K rows
```

**❌ Memory Limits Exceeded**:
- **1.3%+ samples** (37K+ rows): `OutOfMemoryError: Java heap space`
- **Task binary sizes** exceed ~17 MiB causing JVM crashes
- **Symptoms**: RandomForest training fails during tree construction

**Memory Characteristics**:
- **Safe Zone**: Up to 30,000 rows (1.0% sample)
- **Caution Zone**: 30K-34K rows (1.0%-1.2% sample) - may work but risky
- **Danger Zone**: Above 37,000 rows (1.3%+ sample)  
- **Breaking Point**: Task binaries reach ~17.8 MiB limit

#### **Other Local Development Environments**:
```bash
# Start conservative and increase gradually
python spark/spark_real_nyc_taxi_data.py --sample-fraction 0.005   # 0.5%
python spark/spark_real_nyc_taxi_data.py --sample-fraction 0.01    # 1.0%

# Increase Spark driver memory if needed
export SPARK_DRIVER_MEMORY=4g
```

#### **Distributed Spark Cluster**:
```bash
# Can handle full dataset (3M+ records)
python spark/spark_real_nyc_taxi_data.py  # No sampling needed
```

#### **Troubleshooting Out of Memory Errors**:
1. **Reduce sample size**: Use `--sample-fraction 0.005` (0.5%) or smaller
2. **Increase JVM memory**: `export SPARK_DRIVER_MEMORY=8g`  
3. **Reduce model complexity**: Modify `numTrees=20` and `maxDepth=10` in code
4. **Monitor task binaries**: Watch for "Broadcasting large task binary" warnings > 15 MiB

#### **Memory Testing Results (MacBook Pro)**

Comprehensive testing revealed these memory limits:

| Sample % | Rows | Status | Max Task Binary | Notes |
|----------|------|--------|-----------------|-------|
| 0.1% | ~3K | ✅ Works | 2.4 MiB | Very safe |
| 0.5% | ~15K | ✅ Works | 9.2 MiB | Safe |
| 1.0% | ~30K | ✅ Works | 15.1 MiB | Recommended |
| 1.2% | ~34K | ⚠️ Risky | 16.7 MiB | **Caution zone** |
| 1.3% | ~37K | ❌ OOM | 17.8 MiB | **Exceeds limit** |
| 1.5%+ | 45K+ | ❌ Crash | N/A | JVM crashes |

**Key Findings**:
- **Memory wall**: ~17 MiB task binary size
- **Failure point**: RandomForest training phase  
- **Safe recommendation**: Stay at or below 1.0% sample fraction for reliable operation

### Spark Session Issues
```bash
# If you get Java/Spark errors, ensure Java 8+ is installed
java -version

# Set JAVA_HOME if needed
export JAVA_HOME=/path/to/java
```

### MLflow UI Not Showing Runs
- Ensure the tracking URI matches between training and UI
- Check that the backend store path is correct
- Verify the experiment name matches

## Next Steps

- Hyperparameter tuning with Spark ML's CrossValidator
- Model deployment with MLflow's Spark UDF functionality  
- Streaming ML with Spark Structured Streaming
- Advanced features like feature stores and model serving

---

Start with the synthetic data example, then progress to real NYC taxi data. These examples provide a foundation for understanding distributed machine learning with Spark and MLflow.
