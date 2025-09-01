"""
Spark Utilities for MLflow Integration

This module provides reusable Spark utilities for MLflow experiments,
including configurable Spark session creation with optimized settings.
"""

from typing import Dict, Optional
from pyspark.sql import SparkSession


def create_spark_session(
    app_name: str = "MLflow-Spark-Session",
    configs: Optional[Dict[str, str]] = None,
    enable_adaptive_query: bool = True,
    enable_adaptive_coalesce: bool = True,
    serializer: str = "org.apache.spark.serializer.KryoSerializer",
    enable_skew_join: bool = True
) -> SparkSession:
    """
    Create and configure Spark session for MLflow integration.
    
    Args:
        app_name: Name for the Spark application
        configs: Additional Spark configurations as key-value pairs
        enable_adaptive_query: Enable Adaptive Query Execution (AQE)
        enable_adaptive_coalesce: Enable automatic partition coalescing
        serializer: Spark serializer to use (default: KryoSerializer)
        enable_skew_join: Enable skew join optimization
        
    Returns:
        Configured SparkSession instance
        
    Example:
        # Basic usage
        spark = create_spark_session("My-MLflow-App")
        
        # With custom configurations
        custom_configs = {
            "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128MB",
            "spark.driver.memory": "4g"
        }
        spark = create_spark_session(
            app_name="Custom-App",
            configs=custom_configs,
            enable_adaptive_query=True
        )
    """
    # Start with the base session builder
    builder = SparkSession.builder.appName(app_name)
    
    # Apply standard optimized configurations
    if enable_adaptive_query:
        builder = builder.config("spark.sql.adaptive.enabled", "true")
    
    if enable_adaptive_coalesce:
        builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    
    if serializer:
        builder = builder.config("spark.serializer", serializer)
    
    if enable_skew_join:
        builder = builder.config("spark.sql.adaptive.skewJoin.enabled", "true")
    
    # Apply any additional custom configurations
    if configs:
        for key, value in configs.items():
            builder = builder.config(key, value)
    
    return builder.getOrCreate()


def get_optimized_spark_configs() -> Dict[str, str]:
    """
    Get a dictionary of recommended Spark configurations for MLflow workloads.
    
    Returns:
        Dictionary of optimized Spark configuration key-value pairs
    """
    return {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true", 
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128MB",
        "spark.sql.adaptive.coalescePartitions.minPartitionSize": "1MB"
    }


def get_memory_optimized_configs(driver_memory: str = "4g", executor_memory: str = "2g") -> Dict[str, str]:
    """
    Get memory-optimized Spark configurations for local development.
    
    Args:
        driver_memory: Driver memory allocation (e.g., "4g", "8g")
        executor_memory: Executor memory allocation (e.g., "2g", "4g")
        
    Returns:
        Dictionary of memory-optimized Spark configuration key-value pairs
    """
    return {
        "spark.driver.memory": driver_memory,
        "spark.executor.memory": executor_memory,
        "spark.driver.maxResultSize": "2g",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "64MB",
        "spark.sql.adaptive.coalescePartitions.minPartitionSize": "1MB"
        # Removed deprecated: spark.sql.adaptive.coalescePartitions.initialPartitionNum
    }


def create_mlflow_spark_session(
    app_name: str,
    memory_optimized: bool = False,
    driver_memory: str = "4g"
) -> SparkSession:
    """
    Convenience function to create a Spark session optimized for MLflow workloads.
    
    Args:
        app_name: Name for the Spark application
        memory_optimized: Whether to apply memory optimization settings
        driver_memory: Driver memory allocation if memory_optimized is True
        
    Returns:
        Configured SparkSession instance optimized for MLflow
    """
    base_configs = get_optimized_spark_configs()
    
    if memory_optimized:
        memory_configs = get_memory_optimized_configs(driver_memory=driver_memory)
        base_configs.update(memory_configs)
    
    return create_spark_session(
        app_name=app_name,
        configs=base_configs
    )
