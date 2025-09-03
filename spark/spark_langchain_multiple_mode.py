"""
Spark + MLflow + LangChain Integration: Multiple LLM Mode Support
================================================================

A flexible example demonstrating multiple LLM integration options:
1. Spark for distributed text processing
2. LangChain with multiple LLM backends (Mock/Ollama/OpenAI)
3. MLflow autologging for experiment tracking
4. Using existing utility functions for clean, maintainable code

Focus: Educational example showing how to support multiple LLM types

LLM Options:
- Mock: FakeListLLM (no dependencies, for testing)
- Ollama: Local LLM processing (privacy-first)
- OpenAI: Cloud LLM API (requires API key)

Usage:
    python spark_langchain_multiple_mode.py --llm-type mock --num-samples 50
    python spark_langchain_multiple_mode.py --llm-type ollama --ollama-model llama3.2
    python spark_langchain_multiple_mode.py --llm-type openai  # requires OPENAI_API_KEY
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import random

# Add utils to path for imports
sys.path.append(str(Path(__file__).parent.parent / "utils"))

import mlflow
import mlflow.langchain
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# LangChain imports with fallback
try:
    from langchain_community.llms import FakeListLLM
    from langchain_openai import ChatOpenAI
    from langchain_ollama import OllamaLLM
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


def setup_langchain_llm(llm_type: str = "mock", model_name: str = "llama3.2", api_key: str = None, ollama_base_url: str = "http://localhost:11434"):
    """Setup LangChain LLM with support for Ollama, OpenAI, or mock."""
    
    if not LANGCHAIN_AVAILABLE:
        return None
    
    if llm_type == "ollama":
        print(f"🦙 Using Ollama model '{model_name}' for sentiment analysis")
        print(f"🔗 Connecting to Ollama at: {ollama_base_url}")
        try:
            # Use ChatOpenAI client pointing to Ollama for better MLflow autolog integration
            llm = ChatOpenAI(
                model=model_name,
                openai_api_base=f"{ollama_base_url}/v1",  # Ollama OpenAI-compatible endpoint
                openai_api_key="ollama",  # Dummy key (Ollama doesn't require real key)
                temperature=0.1,
                max_tokens=10,  # Short responses for sentiment analysis
                timeout=30
            )
            
            # Test connection
            print("🧪 Testing Ollama connection...")
            test_response = llm.invoke("Hello")
            print(f"✅ Ollama connection successful! Test response: {str(test_response.content)[:50]}...")
            
            return llm
        except Exception as e:
            print(f"⚠️  Failed to connect to Ollama: {e}")
            print("💡 Make sure Ollama is running and the model is installed")
            print(f"   Try: ollama pull {model_name}")
            return None
            
    elif llm_type == "openai" and api_key:
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
        prompt = f"Analyze the sentiment of this text. Respond with ONLY one word: positive, negative, or neutral.\n\nText: \"{text}\"\n\nSentiment:"
        
        if hasattr(llm, 'invoke'):
            response = llm.invoke(prompt)
            # Handle ChatOpenAI response format
            if hasattr(response, 'content'):
                sentiment = str(response.content).strip().lower()
            else:
                sentiment = str(response).strip().lower()
        else:
            response = llm(prompt)
            sentiment = str(response).strip().lower()
            
        # Extract sentiment from response
        if 'positive' in sentiment:
            return 'positive'
        elif 'negative' in sentiment:
            return 'negative'
        elif 'neutral' in sentiment:
            return 'neutral'
        else:
            return 'neutral'
            
    except Exception as e:
        print(f"⚠️  LangChain error: {e}, using fallback")
        return analyze_sentiment_with_langchain(text, None)  # Fallback


def process_texts_with_spark_and_langchain(spark, df, llm_config):
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
            
            # Create LLM instance based on configuration
            llm = None
            if llm_config and llm_config['type'] == 'ollama':
                try:
                    from langchain_openai import ChatOpenAI
                    # Use ChatOpenAI client pointing to Ollama for better MLflow integration
                    llm = ChatOpenAI(
                        model=llm_config['model'],
                        openai_api_base=f"{llm_config['base_url']}/v1",
                        openai_api_key="ollama",
                        temperature=0.1,
                        max_tokens=10,
                        timeout=15
                    )
                except ImportError:
                    pass
            elif llm_config and llm_config['type'] == 'openai':
                try:
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0.1,
                        openai_api_key=llm_config['api_key']
                    )
                except ImportError:
                    pass
            elif llm_config and llm_config['type'] == 'mock':
                try:
                    from langchain_community.llms import FakeListLLM
                    mock_responses = ["positive", "negative", "neutral"] * 50
                    llm = FakeListLLM(responses=mock_responses)
                except ImportError:
                    pass
            
            return analyze_sentiment_with_langchain(text, llm)
            
        except Exception as e:
            # Log error and return neutral as fallback
            print(f"⚠️  UDF error processing text: {e}")
            return analyze_sentiment_with_langchain(text, None)
    
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
    parser = argparse.ArgumentParser(description='Spark + MLflow + LangChain Integration: Multiple LLM Mode Support')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of text samples (default: 50)')
    parser.add_argument('--experiment-name', default='spark-langchain-multiple-mode',
                       help='MLflow experiment name')
    parser.add_argument('--llm-type', choices=['mock', 'ollama', 'openai'], default='mock',
                       help='LLM type to use (default: mock)')
    parser.add_argument('--ollama-model', default='llama3.2',
                       help='Ollama model name (default: llama3.2)')
    parser.add_argument('--ollama-url', default='http://localhost:11434',
                       help='Ollama base URL (default: http://localhost:11434)')
    parser.add_argument('--use-openai', action='store_true',
                       help='Use OpenAI API (requires OPENAI_API_KEY) - DEPRECATED: use --llm-type openai')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Spark + MLflow + LangChain: Multiple LLM Mode Support")
    print("=" * 60)
    print(f"📊 Samples: {args.num_samples}")
    print(f"🤖 LangChain: {'Available' if LANGCHAIN_AVAILABLE else 'Not available'}")
    # Handle deprecated --use-openai flag
    if args.use_openai:
        args.llm_type = 'openai'
    
    print(f"🤖 LLM Type: {args.llm_type}")
    if args.llm_type == 'ollama':
        print(f"🦙 Ollama Model: {args.ollama_model}")
        print(f"🔗 Ollama URL: {args.ollama_url}")
    print("=" * 60)
    
    # End any existing MLflow run
    mlflow.end_run()
    
    # 1. Setup MLflow using our utility
    mlflow_setup = load_mlflow_setup()
    experiment_id = mlflow_setup.setup_mlflow_tracking(
        tracking_uri="file:./mlruns",
        experiment_name=args.experiment_name,
        enable_autolog=True
    )
    
    # Enable MLflow LangChain autologging if using Ollama
    if args.llm_type == 'ollama':
        mlflow.langchain.autolog(silent=True)
        print("✅ MLflow LangChain autologging enabled for Ollama")
    
    # 2. Initialize Spark using our utility
    print("🚀 Initializing Spark session...")
    spark = create_mlflow_spark_session(
        app_name="MLflow-Spark-LangChain-Multiple-Mode"
    )
    
    try:
        with mlflow.start_run(run_name="multiple_mode_langchain_sentiment"):
            
            # 3. Create sample data using Spark
            df = create_sample_text_data(spark, args.num_samples)
            
            # Log basic parameters
            mlflow.log_param("num_samples", args.num_samples)
            mlflow.log_param("llm_type", args.llm_type)
            mlflow.log_param("langchain_available", LANGCHAIN_AVAILABLE)
            if args.llm_type == 'ollama':
                mlflow.log_param("ollama_model", args.ollama_model)
                mlflow.log_param("ollama_url", args.ollama_url)
                mlflow.log_param("langchain_autolog_enabled", True)
            elif args.llm_type == 'openai':
                mlflow.log_param("use_openai", True)
                mlflow.log_param("langchain_autolog_enabled", False)
            else:  # mock
                mlflow.log_param("langchain_autolog_enabled", False)
            
            # Show sample data
            print("\n📝 Sample Data:")
            df.show(5, truncate=False)
            
            # 4. Setup LangChain LLM configuration
            import os
            api_key = os.getenv('OPENAI_API_KEY') if args.llm_type == 'openai' else None
            
            # Test LLM connection (for logging purposes)
            llm = setup_langchain_llm(
                llm_type=args.llm_type,
                model_name=args.ollama_model,
                api_key=api_key,
                ollama_base_url=args.ollama_url
            )
            
            # Create configuration for UDFs (serializable)
            llm_config = None
            if args.llm_type == 'ollama':
                llm_config = {
                    'type': 'ollama',
                    'model': args.ollama_model,
                    'base_url': args.ollama_url
                }
            elif args.llm_type == 'openai':
                llm_config = {
                    'type': 'openai',
                    'api_key': api_key
                }
            elif args.llm_type == 'mock':
                llm_config = {
                    'type': 'mock'
                }
            
            # 5. Process texts with Spark UDFs + LangChain (distributed processing)
            results_df = process_texts_with_spark_and_langchain(spark, df, llm_config)
            
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
