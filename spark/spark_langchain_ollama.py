"""
Spark + MLflow + LangChain + Ollama Integration: Sentiment Analysis
==================================================================

A streamlined example demonstrating Ollama-focused LangChain integration:
1. Spark for distributed text processing
2. Ollama (local LLM) via LangChain for privacy-first sentiment analysis  
3. MLflow LangChain autologging for comprehensive experiment tracking
4. Using existing utility functions for clean, maintainable code

Focus: Production-ready Ollama + LangChain integration - no external API keys required

Key Features:
- Uses ChatOpenAI client pointing to Ollama for better MLflow compatibility
- MLflow LangChain autologging captures individual LLM traces automatically
- Distributed Spark processing with UDF-based LLM integration
- Complete privacy with local LLM processing

Prerequisites:
1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama3.2`
3. Make sure Ollama is running: `ollama serve`

Usage:
    python spark_langchain_ollama.py --ollama-model llama3.2 --num-samples 100
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

# LangChain imports for Ollama
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not available. Install with: uv sync --extra langchain")

# Import our utility functions
from loader import load_mlflow_setup
from spark_utils import create_mlflow_spark_session


def create_sample_text_data(spark: SparkSession, num_samples: int = 100):
    """Generate diverse text data for sentiment analysis demonstration."""
    
    print(f"📝 Creating sample text data ({num_samples} samples)...")
    
    # Expanded sample texts with varied expressions
    positive_texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "Fantastic quality and exceptional service! Highly recommend.",
        "Outstanding experience! This exceeded all my expectations.",
        "Brilliant! This is exactly what I was looking for.",
        "Excellent value for money. Very satisfied with my purchase.",
        "Perfect! Works like a charm and looks great too.",
        "Amazing customer support and fast delivery. Five stars!",
        "This product is incredible! Best purchase I've made this year.",
        "Wonderful experience from start to finish. Thank you!",
        "Superb quality and attention to detail. Love it!"
    ]
    
    negative_texts = [
        "This is terrible. I hate it and want my money back.",
        "Poor quality, very disappointed. Complete waste of money.",
        "Awful experience. Nothing worked as advertised.",
        "Horrible customer service. Would not recommend to anyone.",
        "Completely useless. Broke after one day of use.",
        "Worst purchase ever. Save your money and buy something else.",
        "Terrible quality control. Product arrived damaged.",
        "Disappointing performance. Does not meet basic expectations.",
        "Frustrating experience. Nothing but problems from day one.",
        "Unacceptable quality for the price. Very unsatisfied."
    ]
    
    neutral_texts = [
        "It's okay, nothing special. Does what it's supposed to do.",
        "Average product, meets basic expectations.",
        "Decent value for the price. Neither great nor terrible.",
        "Standard quality. Works fine but nothing extraordinary.",
        "It's alright. Some good points, some not so good.",
        "Acceptable performance. Gets the job done adequately.",
        "Fair quality for the price point. Could be better.",
        "Reasonable option. Not the best but not the worst either.",
        "Satisfactory experience overall. No major complaints.",
        "Ordinary product. Functions as expected, nothing more."
    ]
    
    # Generate balanced dataset
    data = []
    samples_per_category = num_samples // 3
    remainder = num_samples % 3
    
    categories = [
        (positive_texts, "positive", samples_per_category + (1 if remainder > 0 else 0)),
        (negative_texts, "negative", samples_per_category + (1 if remainder > 1 else 0)),
        (neutral_texts, "neutral", samples_per_category)
    ]
    
    id_counter = 1
    for texts, sentiment, count in categories:
        for i in range(count):
            text = random.choice(texts)
            data.append({
                'id': id_counter,
                'text': text,
                'true_sentiment': sentiment
            })
            id_counter += 1
    
    # Shuffle the data
    random.shuffle(data)
    
    # Create Spark DataFrame
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("text", StringType(), True),
        StructField("true_sentiment", StringType(), True)
    ])
    
    df = spark.createDataFrame(data, schema)
    print(f"✅ Created balanced dataset: {df.count()} rows")
    
    return df


def setup_ollama_llm(model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
    """Setup Ollama LLM using ChatOpenAI client for better MLflow integration."""
    
    if not LANGCHAIN_AVAILABLE:
        print("⚠️  LangChain not available. Using fallback rule-based analysis.")
        return None
    
    print(f"🦙 Using Ollama model '{model_name}' for sentiment analysis")
    print(f"🔗 Connecting to Ollama at: {base_url}")
    
    try:
        # Use ChatOpenAI client pointing to Ollama for better MLflow integration
        llm = ChatOpenAI(
            model=model_name,
            openai_api_base=f"{base_url}/v1",  # Ollama OpenAI-compatible endpoint
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


def analyze_sentiment_with_ollama(text: str, llm) -> str:
    """Analyze sentiment using Ollama via ChatOpenAI client.
    
    Uses a recursive fallback pattern: if Ollama fails or returns unclear results,
    falls back to rule-based keyword matching by calling itself with llm=None.
    """
    
    if not llm:
        # Fallback rule-based analysis
        text_lower = text.lower()
        positive_words = ['love', 'amazing', 'fantastic', 'excellent', 'outstanding', 'brilliant', 'perfect', 'wonderful', 'superb']
        negative_words = ['hate', 'terrible', 'poor', 'awful', 'horrible', 'worst', 'disappointing', 'frustrating', 'unacceptable']
        
        if any(word in text_lower for word in positive_words):
            return 'positive'
        elif any(word in text_lower for word in negative_words):
            return 'negative'
        else:
            return 'neutral'
    
    try:
        prompt = f"Analyze the sentiment of this text. Respond with ONLY one word: positive, negative, or neutral.\n\nText: \"{text}\"\n\nSentiment:"
        
        # Use ChatOpenAI invoke method
        response = llm.invoke(prompt)
        
        # Extract content from ChatOpenAI response
        sentiment = str(response.content).strip().lower()
        
        # Extract sentiment from response
        if 'positive' in sentiment:
            return 'positive'
        elif 'negative' in sentiment:
            return 'negative'
        elif 'neutral' in sentiment:
            return 'neutral'
        else:
            # If response is unclear, use fallback
            return analyze_sentiment_with_ollama(text, None)
            
    except Exception as e:
        print(f"⚠️  Ollama error: {e}, using fallback")
        return analyze_sentiment_with_ollama(text, None)  # Fallback


def process_texts_with_spark_and_ollama(spark, df, model_name: str, base_url: str):
    """Process texts using Spark UDFs for distributed Ollama integration."""
    
    print("🔄 Processing texts with Spark UDFs + Ollama...")
    
    # Create a UDF wrapper for Ollama sentiment analysis
    def sentiment_analysis_udf(text: str) -> str:
        """
        UDF wrapper for Ollama sentiment analysis.
        
        Each Spark executor creates its own ChatOpenAI-Ollama connection
        for better MLflow integration and distributed processing.
        """
        try:
            if text is None or text.strip() == "":
                return "neutral"
            
            # Import inside UDF to ensure it's available on executors
            try:
                from langchain_openai import ChatOpenAI
                
                # Create a local ChatOpenAI instance pointing to Ollama
                local_llm = ChatOpenAI(
                    model=model_name,
                    openai_api_base=f"{base_url}/v1",
                    openai_api_key="ollama",
                    temperature=0.1,
                    max_tokens=10,
                    timeout=15  # Shorter timeout for UDF
                )
                
                return analyze_sentiment_with_ollama(text, local_llm)
                
            except ImportError:
                # Fallback if LangChain not available on executor
                return analyze_sentiment_with_ollama(text, None)
                
        except Exception as e:
            print(f"⚠️  UDF error processing text '{text[:20]}...': {e}")
            # Use fallback rule-based analysis
            return analyze_sentiment_with_ollama(text, None)
    
    # Register the UDF with Spark
    sentiment_udf = udf(sentiment_analysis_udf, StringType())
    
    print("🌐 Applying UDF across Spark cluster for distributed processing...")
    print("🦙 Each executor will create its own Ollama connection")
    
    # Apply UDF to process texts in a distributed manner
    results_df = df.withColumn("predicted_sentiment", sentiment_udf(col("text")))
    
    # Cache the result for better performance
    results_df.cache()
    
    # Force evaluation to trigger the distributed processing
    count = results_df.count()
    print(f"✅ Processed {count} texts using distributed UDFs with Ollama")
    
    return results_df


def calculate_detailed_metrics(results_df):
    """Calculate comprehensive accuracy metrics."""
    
    pandas_df = results_df.toPandas()
    
    # Overall accuracy
    correct = (pandas_df['true_sentiment'] == pandas_df['predicted_sentiment']).sum()
    total = len(pandas_df)
    accuracy = correct / total if total > 0 else 0
    
    # Per-class accuracy
    class_metrics = {}
    for sentiment in ['positive', 'negative', 'neutral']:
        class_data = pandas_df[pandas_df['true_sentiment'] == sentiment]
        if len(class_data) > 0:
            class_correct = (class_data['true_sentiment'] == class_data['predicted_sentiment']).sum()
            class_accuracy = class_correct / len(class_data)
            class_metrics[f'{sentiment}_accuracy'] = class_accuracy
            class_metrics[f'{sentiment}_count'] = len(class_data)
        else:
            class_metrics[f'{sentiment}_accuracy'] = 0.0
            class_metrics[f'{sentiment}_count'] = 0
    
    return {
        'accuracy': accuracy,
        'correct_predictions': correct,
        'total_predictions': total,
        **class_metrics
    }


def main():
    """Main function demonstrating Spark + MLflow + Ollama integration."""
    
    parser = argparse.ArgumentParser(description='Spark + MLflow + Ollama Integration')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of text samples (default: 50)')
    parser.add_argument('--experiment-name', default='spark-langchain-ollama',
                       help='MLflow experiment name')
    parser.add_argument('--ollama-model', default='llama3.2',
                       help='Ollama model name (default: llama3.2)')
    parser.add_argument('--ollama-url', default='http://localhost:11434',
                       help='Ollama base URL (default: http://localhost:11434)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🦙 Spark + MLflow + Ollama Integration")
    print("=" * 70)
    print(f"📊 Samples: {args.num_samples}")
    print(f"🦙 Ollama Model: {args.ollama_model}")
    print(f"🔗 Ollama URL: {args.ollama_url}")
    print(f"🤖 LangChain: {'Available' if LANGCHAIN_AVAILABLE else 'Not available'}")
    print("=" * 70)
    
    # End any existing MLflow run
    mlflow.end_run()
    
    # 1. Setup MLflow with LangChain autologging
    mlflow_setup = load_mlflow_setup()
    experiment_id = mlflow_setup.setup_mlflow_tracking(
        tracking_uri="file:./mlruns",
        experiment_name=args.experiment_name,
        enable_autolog=False  # We'll enable LangChain autolog separately
    )
    
    # Enable MLflow LangChain autologging for automatic LLM tracking
    mlflow.langchain.autolog(silent=True)
    print("✅ MLflow LangChain autologging enabled")
    
    # 2. Initialize Spark
    print("🚀 Initializing Spark session...")
    spark = create_mlflow_spark_session(
        app_name="MLflow-Spark-LangChain-Ollama"
    )
    
    try:
        with mlflow.start_run(run_name="ollama_sentiment_analysis"):
            
            # 3. Create sample data
            df = create_sample_text_data(spark, args.num_samples)
            
            # Log parameters (autolog will handle LangChain operations)
            mlflow.log_param("num_samples", args.num_samples)
            mlflow.log_param("llm_type", "ollama")
            mlflow.log_param("ollama_model", args.ollama_model)
            mlflow.log_param("ollama_url", args.ollama_url)
            mlflow.log_param("langchain_available", LANGCHAIN_AVAILABLE)
            mlflow.log_param("autolog_enabled", True)
            
            # Show sample data
            print("\n📝 Sample Data:")
            df.show(5, truncate=False)
            
            # 4. Setup Ollama LLM (this will be auto-logged)
            llm = setup_ollama_llm(args.ollama_model, args.ollama_url)
            
            if llm is None:
                print("⚠️  Proceeding with rule-based fallback analysis...")
                mlflow.log_param("ollama_status", "failed_fallback_used")
            else:
                mlflow.log_param("ollama_status", "connected")
                
                # Test the LLM with a sample (this will be auto-logged by MLflow)
                print("🧪 Testing LLM with sample text (auto-logged)...")
                sample_text = "This product is amazing and works perfectly!"
                sample_result = analyze_sentiment_with_ollama(sample_text, llm)
                print(f"   Sample: '{sample_text}' → {sample_result}")
                mlflow.log_param("sample_test_result", sample_result)
            
            # 5. Process texts with Spark + Ollama (autolog will track LLM calls)
            results_df = process_texts_with_spark_and_ollama(spark, df, args.ollama_model, args.ollama_url)
            
            # 6. Calculate and log detailed metrics
            metrics = calculate_detailed_metrics(results_df)
            
            # Log all metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # 7. Display results
            print(f"\n📊 Results Summary:")
            print(f"  ✅ Overall Accuracy: {metrics['accuracy']:.3f}")
            print(f"  ✅ Correct Predictions: {metrics['correct_predictions']}/{metrics['total_predictions']}")
            
            print(f"\n📈 Per-Class Performance:")
            for sentiment in ['positive', 'negative', 'neutral']:
                accuracy = metrics[f'{sentiment}_accuracy']
                count = metrics[f'{sentiment}_count']
                print(f"  {sentiment.title()}: {accuracy:.3f} ({count} samples)")
            
            print(f"\n🔍 Sample Predictions:")
            results_df.select("text", "true_sentiment", "predicted_sentiment").show(10, truncate=False)
            
            print(f"\n✅ MLflow Run completed: {mlflow.active_run().info.run_id}")
            print(f"🔗 View in MLflow UI: file:./mlruns")
            print(f"🤖 LangChain operations automatically logged by MLflow autolog!")
    
    finally:
        print("\n🛑 Stopping Spark session...")
        spark.stop()
        print("✅ Spark session stopped")


if __name__ == "__main__":
    main()
