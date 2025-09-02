"""
Simplified Spark + MLflow + LangChain Integration: Sentiment Analysis

A streamlined example demonstrating:
1. Spark for distributed text processing
2. LangChain for LLM-powered sentiment analysis  
3. MLflow autologging for experiment tracking
4. Using existing utility functions for clean, maintainable code

Focus: Core integration without complex feature engineering
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import random

# Add utils to path for imports
sys.path.append(str(Path(__file__).parent.parent / "utils"))

import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# LangChain imports with fallback
try:
    from langchain_community.llms import FakeListLLM
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not available. Install with: uv sync --extra langchain")

# Import our utility functions
from loader import load_mlflow_setup
from spark_utils import create_mlflow_spark_session


def create_sample_text_data(spark: SparkSession, num_samples: int = 100):
    """Generate simple text data for sentiment analysis demonstration."""
    
    print(f"📝 Creating sample text data ({num_samples} samples)...")
    
    # Simple sample texts with known sentiments
    sample_texts = [
        ("I love this product! It's amazing.", "positive"),
        ("This is terrible. I hate it.", "negative"), 
        ("It's okay, nothing special.", "neutral"),
        ("Fantastic quality and great service!", "positive"),
        ("Poor quality, very disappointed.", "negative"),
        ("Average product, meets expectations.", "neutral"),
        ("Outstanding! Highly recommend.", "positive"),
        ("Waste of money, completely useless.", "negative"),
        ("Decent value for the price.", "neutral"),
        ("Excellent experience overall!", "positive")
    ]
    
    # Generate data by sampling from templates
    data = []
    for i in range(num_samples):
        text, sentiment = random.choice(sample_texts)
        data.append({
            'id': i + 1,
            'text': text,
            'true_sentiment': sentiment
        })
    
    # Create Spark DataFrame
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("text", StringType(), True),
        StructField("true_sentiment", StringType(), True)
    ])
    
    df = spark.createDataFrame(data, schema)
    print(f"✅ Created dataset: {df.count()} rows")
    
    return df


def setup_langchain_llm(use_openai: bool = False, api_key: str = None):
    """Setup LangChain LLM with fallback to mock."""
    
    if not LANGCHAIN_AVAILABLE:
        return None
        
    if use_openai and api_key:
        print("🤖 Using OpenAI ChatGPT for sentiment analysis")
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=api_key
        )
    else:
        print("🎭 Using mock LLM for demonstration")
        # Simple mock responses for demo
        mock_responses = [
            "positive", "negative", "neutral", 
            "positive", "negative", "neutral"
        ] * 50  # Repeat for multiple calls
        
        return FakeListLLM(responses=mock_responses)


def analyze_sentiment_with_langchain(text: str, llm) -> str:
    """Simple sentiment analysis using LangChain."""
    
    if not llm:
        # Fallback rule-based analysis
        text_lower = text.lower()
        if any(word in text_lower for word in ['love', 'amazing', 'fantastic', 'excellent', 'outstanding']):
            return 'positive'
        elif any(word in text_lower for word in ['hate', 'terrible', 'poor', 'waste', 'useless']):
            return 'negative'
        else:
            return 'neutral'
    
    try:
        prompt = f"Analyze the sentiment of this text and respond with only 'positive', 'negative', or 'neutral': {text}"
        
        if hasattr(llm, 'invoke'):
            response = llm.invoke(prompt)
        else:
            response = llm(prompt)
            
        # Clean response
        sentiment = str(response).strip().lower()
        if 'positive' in sentiment:
            return 'positive'
        elif 'negative' in sentiment:
            return 'negative'
        else:
            return 'neutral'
            
    except Exception as e:
        print(f"⚠️  LangChain error: {e}, using fallback")
        return analyze_sentiment_with_langchain(text, None)  # Fallback


def process_texts_with_spark_and_langchain(spark, df, llm):
    """Process texts using Spark UDFs for distributed LangChain integration."""
    
    print("🔄 Processing texts with Spark UDFs + LangChain...")
    
    # Create a UDF wrapper for sentiment analysis with error handling
    def sentiment_analysis_udf(text: str) -> str:
        """
        UDF wrapper for LangChain sentiment analysis.
        
        This function will be executed on each Spark executor,
        enabling true distributed processing of text data.
        """
        try:
            if text is None or text.strip() == "":
                return "neutral"
            return analyze_sentiment_with_langchain(text, llm)
        except Exception as e:
            # Log error and return neutral as fallback
            print(f"⚠️  UDF error processing text: {e}")
            return "neutral"
    
    # Register the UDF with Spark (optimized for string processing)
    sentiment_udf = udf(sentiment_analysis_udf, StringType())
    
    print("🌐 Applying UDF across Spark cluster for distributed processing...")
    
    # Apply UDF to process texts in a distributed manner
    # Each partition will be processed on different executors
    results_df = df.withColumn("predicted_sentiment", sentiment_udf(col("text")))
    
    # Cache the result for better performance if accessed multiple times
    results_df.cache()
    
    # Force evaluation to trigger the distributed processing
    count = results_df.count()
    print(f"✅ Processed {count} texts using distributed UDFs across Spark cluster")
    
    return results_df


def calculate_accuracy(results_df):
    """Calculate simple accuracy metrics."""
    
    pandas_df = results_df.toPandas()
    
    correct = (pandas_df['true_sentiment'] == pandas_df['predicted_sentiment']).sum()
    total = len(pandas_df)
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct_predictions': correct,
        'total_predictions': total
    }


def main():
    """Main function demonstrating simplified Spark + MLflow + LangChain integration."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Simplified Spark + MLflow + LangChain Integration')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of text samples (default: 50)')
    parser.add_argument('--experiment-name', default='spark-langchain-simple',
                       help='MLflow experiment name')
    parser.add_argument('--use-openai', action='store_true',
                       help='Use OpenAI API (requires OPENAI_API_KEY)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Simplified Spark + MLflow + LangChain Integration")
    print("=" * 60)
    print(f"📊 Samples: {args.num_samples}")
    print(f"🤖 LangChain: {'Available' if LANGCHAIN_AVAILABLE else 'Not available'}")
    print(f"🔑 OpenAI: {'Yes' if args.use_openai else 'Mock/Demo'}")
    print("=" * 60)
    
    # End any existing MLflow run
    mlflow.end_run()
    
    # 1. Setup MLflow using our utility
    mlflow_setup = load_mlflow_setup()
    experiment_id = mlflow_setup.setup_mlflow_tracking(
        tracking_uri="file:./mlruns",
        experiment_name=args.experiment_name,
        enable_autolog=True  # Use autolog for any sklearn components
    )
    
    # 2. Initialize Spark using our utility
    print("🚀 Initializing Spark session...")
    spark = create_mlflow_spark_session(
        app_name="MLflow-Spark-LangChain-Simple"
    )
    
    try:
        with mlflow.start_run(run_name="simple_langchain_sentiment"):
            
            # 3. Create sample data using Spark
            df = create_sample_text_data(spark, args.num_samples)
            
            # Log basic parameters
            mlflow.log_param("num_samples", args.num_samples)
            mlflow.log_param("use_openai", args.use_openai)
            mlflow.log_param("langchain_available", LANGCHAIN_AVAILABLE)
            
            # Show sample data
            print("\n📝 Sample Data:")
            df.show(5, truncate=False)
            
            # 4. Setup LangChain LLM
            import os
            api_key = os.getenv('OPENAI_API_KEY') if args.use_openai else None
            llm = setup_langchain_llm(args.use_openai, api_key)
            
            # 5. Process texts with Spark UDFs + LangChain (distributed processing)
            results_df = process_texts_with_spark_and_langchain(spark, df, llm)
            
            # 6. Calculate and log metrics
            metrics = calculate_accuracy(results_df)
            
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("correct_predictions", metrics['correct_predictions'])
            mlflow.log_metric("total_predictions", metrics['total_predictions'])
            
            # 7. Display results
            print(f"\n📊 Results:")
            print(f"  ✅ Accuracy: {metrics['accuracy']:.3f}")
            print(f"  ✅ Correct: {metrics['correct_predictions']}/{metrics['total_predictions']}")
            
            print(f"\n🔍 Sample Predictions:")
            results_df.show(10, truncate=False)
            
            print(f"\n✅ MLflow Run completed: {mlflow.active_run().info.run_id}")
            print(f"🔗 View in MLflow UI: file:./mlruns")
    
    finally:
        # Clean up
        print("\n🛑 Stopping Spark session...")
        spark.stop()
        print("✅ Spark session stopped")


if __name__ == "__main__":
    main()
