"""
Spark + LangChain + MLflow Integration Example: Sentiment Analysis

This example demonstrates:
1. Distributed text processing with Apache Spark
2. LangChain integration for LLM-powered sentiment analysis
3. MLflow experiment tracking for NLP workflows
4. Scalable text analytics with proper error handling
5. Comparison between rule-based and LLM-based sentiment analysis

Features:
- Synthetic text data generation for demonstration
- Distributed sentiment analysis using LangChain
- MLflow tracking of NLP experiments
- Performance metrics and model comparison
- Configurable batch processing for large datasets

Requirements:
- Apache Spark (pyspark)
- MLflow with LangChain support
- LangChain and OpenAI API (optional, falls back to mock)
- Internet connection for LLM API calls (optional)
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import random
import pandas as pd

# Add utils to path for imports
sys.path.append(str(Path(__file__).parent.parent / "utils"))

import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit, count, avg, when, isnan, isnull
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, ArrayType

# LangChain imports with fallback handling
try:
    from langchain.schema import BaseMessage, HumanMessage
    from langchain_community.callbacks.manager import get_openai_callback
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import FakeListLLM
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  LangChain not available. Using mock sentiment analysis.")

# Import utility functions
from loader import load_mlflow_setup
from spark_utils import create_mlflow_spark_session
# Note: Using direct MLflow logging instead of spark_ml_utils for NLP-specific workflow


def create_synthetic_text_data(spark: SparkSession, num_samples: int = 1000) -> "DataFrame":
    """
    Create synthetic text data for sentiment analysis demonstration.
    
    Args:
        spark: Spark session
        num_samples: Number of text samples to generate
        
    Returns:
        Spark DataFrame with text and true sentiment labels
    """
    print(f"📝 Generating synthetic text dataset ({num_samples:,} samples)...")
    
    # Sample texts with known sentiments
    positive_texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is the best service I've ever experienced. Highly recommended!",
        "Outstanding quality and excellent customer support. Five stars!",
        "Incredible value for money. I'm so happy with this purchase.",
        "Fantastic experience from start to finish. Will definitely buy again.",
        "Superb quality and fast delivery. Exceeded my expectations!",
        "Amazing product that solved all my problems. Thank you so much!",
        "Perfect in every way. Great design and functionality.",
        "Excellent service and wonderful staff. Truly impressed!",
        "This product is a game-changer. Absolutely brilliant!"
    ]
    
    negative_texts = [
        "This product is terrible. Complete waste of money.",
        "Worst customer service ever. Very disappointed and frustrated.",
        "Poor quality and doesn't work as advertised. Avoid at all costs.",
        "Horrible experience. The product broke after one day of use.",
        "Terrible value for money. I want my money back immediately.",
        "Awful quality and very poor customer support. Never again!",
        "This is the worst purchase I've ever made. Completely useless.",
        "Disappointing product that doesn't meet basic expectations.",
        "Poor service and rude staff. Would not recommend to anyone.",
        "Defective product with no support. Total disaster!"
    ]
    
    neutral_texts = [
        "The product is okay. It works but nothing special about it.",
        "Average service. Not bad but not great either. It's fine.",
        "The product does what it's supposed to do. Nothing more, nothing less.",
        "Standard quality for the price. No complaints but no praise either.",
        "It's an acceptable product. Works as expected without issues.",
        "The service was adequate. Got what I paid for, nothing extra.",
        "Decent product with basic functionality. Meets minimum requirements.",
        "Fair quality and reasonable price. It's an okay choice.",
        "The product is functional but not particularly impressive.",
        "Standard experience. Neither good nor bad, just average."
    ]
    
    # Generate random samples
    data = []
    for i in range(num_samples):
        sentiment_type = random.choice(['positive', 'negative', 'neutral'])
        
        if sentiment_type == 'positive':
            text = random.choice(positive_texts)
            sentiment_score = random.uniform(0.6, 1.0)
            sentiment_label = 'positive'
        elif sentiment_type == 'negative':
            text = random.choice(negative_texts)
            sentiment_score = random.uniform(-1.0, -0.6)
            sentiment_label = 'negative'
        else:  # neutral
            text = random.choice(neutral_texts)
            sentiment_score = random.uniform(-0.3, 0.3)
            sentiment_label = 'neutral'
        
        # Add some variation to the text
        variations = [
            text,
            text + " Really!",
            text + " Honestly.",
            "I think " + text.lower(),
            "In my opinion, " + text.lower()
        ]
        
        final_text = random.choice(variations)
        
        data.append({
            'id': i + 1,
            'text': final_text,
            'true_sentiment': sentiment_label,
            'true_sentiment_score': round(sentiment_score, 3),
            'text_length': len(final_text),
            'word_count': len(final_text.split())
        })
    
    # Create Spark DataFrame
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("text", StringType(), True),
        StructField("true_sentiment", StringType(), True),
        StructField("true_sentiment_score", DoubleType(), True),
        StructField("text_length", IntegerType(), True),
        StructField("word_count", IntegerType(), True)
    ])
    
    df = spark.createDataFrame(data, schema)
    print(f"✅ Generated dataset: {df.count():,} rows, {len(df.columns)} columns")
    
    return df


def create_langchain_sentiment_analyzer(use_openai: bool = False, api_key: Optional[str] = None):
    """
    Create a LangChain-based sentiment analyzer.
    
    Args:
        use_openai: Whether to use OpenAI API (requires API key)
        api_key: OpenAI API key (optional)
        
    Returns:
        LangChain LLM instance
    """
    if not LANGCHAIN_AVAILABLE:
        return None
    
    if use_openai and api_key:
        print("🤖 Initializing OpenAI ChatGPT for sentiment analysis...")
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=api_key,
            max_tokens=50
        )
    else:
        print("🎭 Using mock LLM for sentiment analysis demonstration...")
        # Mock responses for demonstration
        responses = [
            "positive: 0.8",
            "negative: -0.7", 
            "neutral: 0.1",
            "positive: 0.9",
            "negative: -0.6",
            "neutral: -0.1",
            "positive: 0.7",
            "negative: -0.8",
            "neutral: 0.2"
        ]
        return FakeListLLM(responses=responses * 1000)  # Repeat for multiple calls


def analyze_sentiment_with_langchain(text: str, llm) -> Dict[str, Any]:
    """
    Analyze sentiment of text using LangChain LLM.
    
    Args:
        text: Text to analyze
        llm: LangChain LLM instance
        
    Returns:
        Dictionary with sentiment analysis results
    """
    if not LANGCHAIN_AVAILABLE or llm is None:
        # Fallback to simple rule-based analysis
        return analyze_sentiment_rule_based(text)
    
    try:
        prompt = f"""
        Analyze the sentiment of the following text and respond with only the sentiment (positive, negative, or neutral) 
        and a confidence score between -1.0 and 1.0, formatted as "sentiment: score".
        
        Text: "{text}"
        
        Response format: sentiment: score
        """
        
        if hasattr(llm, 'invoke'):
            response = llm.invoke(prompt)
        else:
            response = llm(prompt)
        
        # Parse response
        response_text = str(response).strip().lower()
        
        if 'positive' in response_text:
            sentiment = 'positive'
            # Extract score or default
            try:
                score = float(response_text.split(':')[1].strip())
            except:
                score = 0.7
        elif 'negative' in response_text:
            sentiment = 'negative'
            try:
                score = float(response_text.split(':')[1].strip())
            except:
                score = -0.7
        else:
            sentiment = 'neutral'
            try:
                score = float(response_text.split(':')[1].strip())
            except:
                score = 0.0
        
        return {
            'predicted_sentiment': sentiment,
            'predicted_score': score,
            'confidence': abs(score),
            'method': 'langchain_llm'
        }
        
    except Exception as e:
        print(f"⚠️  LangChain analysis failed: {e}")
        return analyze_sentiment_rule_based(text)


def analyze_sentiment_rule_based(text: str) -> Dict[str, Any]:
    """
    Simple rule-based sentiment analysis as fallback.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment analysis results
    """
    text_lower = text.lower()
    
    positive_words = ['love', 'amazing', 'excellent', 'great', 'fantastic', 'wonderful', 
                     'perfect', 'outstanding', 'superb', 'brilliant', 'best', 'happy']
    negative_words = ['terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointed', 
                     'poor', 'bad', 'useless', 'disaster', 'broken', 'defective']
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        sentiment = 'positive'
        score = min(0.8, 0.3 + positive_count * 0.2)
    elif negative_count > positive_count:
        sentiment = 'negative'
        score = max(-0.8, -0.3 - negative_count * 0.2)
    else:
        sentiment = 'neutral'
        score = 0.0
    
    return {
        'predicted_sentiment': sentiment,
        'predicted_score': score,
        'confidence': abs(score),
        'method': 'rule_based'
    }


def process_batch_with_langchain(texts: List[str], llm, batch_id: int) -> List[Dict[str, Any]]:
    """
    Process a batch of texts with LangChain sentiment analysis.
    
    Args:
        texts: List of texts to analyze
        llm: LangChain LLM instance
        batch_id: Batch identifier for logging
        
    Returns:
        List of sentiment analysis results
    """
    print(f"🔄 Processing batch {batch_id} ({len(texts)} texts)...")
    
    results = []
    start_time = time.time()
    
    for i, text in enumerate(texts):
        try:
            result = analyze_sentiment_with_langchain(text, llm)
            result['batch_id'] = batch_id
            result['text_id'] = i
            results.append(result)
        except Exception as e:
            print(f"⚠️  Error processing text {i} in batch {batch_id}: {e}")
            # Add fallback result
            fallback = analyze_sentiment_rule_based(text)
            fallback['batch_id'] = batch_id
            fallback['text_id'] = i
            fallback['error'] = str(e)
            results.append(fallback)
    
    processing_time = time.time() - start_time
    print(f"✅ Batch {batch_id} completed in {processing_time:.2f}s ({len(texts)/processing_time:.1f} texts/sec)")
    
    return results


def calculate_sentiment_metrics(df: "DataFrame") -> Dict[str, float]:
    """
    Calculate sentiment analysis performance metrics.
    
    Args:
        df: DataFrame with true and predicted sentiments
        
    Returns:
        Dictionary with performance metrics
    """
    print("📊 Calculating sentiment analysis metrics...")
    
    # Convert to Pandas for easier metric calculation
    pandas_df = df.toPandas()
    
    # Accuracy
    correct_predictions = (pandas_df['true_sentiment'] == pandas_df['predicted_sentiment']).sum()
    total_predictions = len(pandas_df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Sentiment distribution
    true_dist = pandas_df['true_sentiment'].value_counts(normalize=True).to_dict()
    pred_dist = pandas_df['predicted_sentiment'].value_counts(normalize=True).to_dict()
    
    # Score correlation (if available)
    score_correlation = 0.0
    try:
        if 'true_sentiment_score' in pandas_df.columns and 'predicted_score' in pandas_df.columns:
            score_correlation = pandas_df['true_sentiment_score'].corr(pandas_df['predicted_score'])
            if pd.isna(score_correlation):
                score_correlation = 0.0
    except:
        pass
    
    # Average confidence
    avg_confidence = pandas_df['confidence'].mean() if 'confidence' in pandas_df.columns else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'total_samples': total_predictions,
        'correct_predictions': correct_predictions,
        'score_correlation': score_correlation,
        'avg_confidence': avg_confidence,
        'true_positive_ratio': true_dist.get('positive', 0.0),
        'true_negative_ratio': true_dist.get('negative', 0.0),
        'true_neutral_ratio': true_dist.get('neutral', 0.0),
        'pred_positive_ratio': pred_dist.get('positive', 0.0),
        'pred_negative_ratio': pred_dist.get('negative', 0.0),
        'pred_neutral_ratio': pred_dist.get('neutral', 0.0)
    }
    
    return metrics


def main():
    """Main execution function for Spark + LangChain + MLflow sentiment analysis."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Spark + LangChain + MLflow Sentiment Analysis')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of text samples to generate (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing (default: 100)')
    parser.add_argument('--experiment-name', default='spark-langchain-sentiment',
                       help='MLflow experiment name')
    parser.add_argument('--use-openai', action='store_true',
                       help='Use OpenAI API (requires OPENAI_API_KEY env var)')
    parser.add_argument('--openai-api-key', 
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Get OpenAI API key
    api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    
    print("=" * 60)
    print("🚀 Spark + LangChain + MLflow Sentiment Analysis")
    print("=" * 60)
    print(f"📊 Samples: {args.num_samples:,}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🤖 LangChain available: {LANGCHAIN_AVAILABLE}")
    print(f"🔑 OpenAI API: {'Yes' if (args.use_openai and api_key) else 'Mock/Fallback'}")
    print("=" * 60)
    
    # End any existing MLflow run
    mlflow.end_run()
    
    # 1. Setup MLflow
    mlflow_setup = load_mlflow_setup()
    experiment_id = mlflow_setup.setup_mlflow_tracking(
        tracking_uri="file:./mlruns",
        experiment_name=args.experiment_name,
        enable_autolog=False  # Manual logging for NLP
    )
    
    # 2. Initialize Spark Session
    print("🚀 Initializing Spark Session for LangChain Integration...")
    spark = create_mlflow_spark_session(
        app_name="MLflow-Spark-LangChain-Sentiment"
    )
    
    try:
        with mlflow.start_run(run_name="langchain_sentiment_analysis"):
            
            # 3. Generate synthetic text data
            df = create_synthetic_text_data(spark, args.num_samples)
            
            # Log dataset info
            mlflow.log_param("total_rows", df.count())
            mlflow.log_param("feature_count", len(df.columns) - 2)
            mlflow.log_param("data_source", "Synthetic Text Data")
            mlflow.log_param("task_type", "sentiment_analysis")
            
            # Show sample data
            print("\n📝 Sample Text Data:")
            df.select("id", "text", "true_sentiment", "word_count").show(5, truncate=False)
            
            # 4. Initialize LangChain sentiment analyzer
            llm = create_langchain_sentiment_analyzer(
                use_openai=args.use_openai,
                api_key=api_key
            )
            
            # 5. Process texts in batches using Spark
            print(f"\n🔄 Processing {args.num_samples:,} texts in batches of {args.batch_size}...")
            
            # Collect texts for processing (in practice, you'd use Spark UDFs for true distributed processing)
            texts_data = df.select("id", "text", "true_sentiment", "true_sentiment_score").collect()
            
            all_results = []
            total_batches = (len(texts_data) + args.batch_size - 1) // args.batch_size
            
            start_time = time.time()
            
            for batch_num in range(total_batches):
                start_idx = batch_num * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(texts_data))
                
                batch_texts = [row['text'] for row in texts_data[start_idx:end_idx]]
                batch_results = process_batch_with_langchain(batch_texts, llm, batch_num + 1)
                
                # Add original data to results
                for i, result in enumerate(batch_results):
                    original_row = texts_data[start_idx + i]
                    result.update({
                        'id': original_row['id'],
                        'text': original_row['text'],
                        'true_sentiment': original_row['true_sentiment'],
                        'true_sentiment_score': original_row['true_sentiment_score']
                    })
                
                all_results.extend(batch_results)
            
            total_time = time.time() - start_time
            
            # 6. Create results DataFrame
            results_schema = StructType([
                StructField("id", IntegerType(), True),
                StructField("text", StringType(), True),
                StructField("true_sentiment", StringType(), True),
                StructField("true_sentiment_score", DoubleType(), True),
                StructField("predicted_sentiment", StringType(), True),
                StructField("predicted_score", DoubleType(), True),
                StructField("confidence", DoubleType(), True),
                StructField("method", StringType(), True),
                StructField("batch_id", IntegerType(), True)
            ])
            
            results_data = []
            for result in all_results:
                results_data.append((
                    result['id'],
                    result['text'],
                    result['true_sentiment'],
                    result['true_sentiment_score'],
                    result['predicted_sentiment'],
                    result['predicted_score'],
                    result['confidence'],
                    result['method'],
                    result['batch_id']
                ))
            
            results_df = spark.createDataFrame(results_data, results_schema)
            
            # 7. Calculate metrics
            metrics = calculate_sentiment_metrics(results_df)
            
            # 8. Log parameters and metrics to MLflow
            mlflow.log_param("num_samples", args.num_samples)
            mlflow.log_param("batch_size", args.batch_size)
            mlflow.log_param("use_openai", args.use_openai)
            mlflow.log_param("langchain_available", LANGCHAIN_AVAILABLE)
            mlflow.log_param("total_batches", total_batches)
            mlflow.log_param("processing_method", "langchain_llm" if LANGCHAIN_AVAILABLE else "rule_based")
            
            # Log performance metrics
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("total_samples", metrics['total_samples'])
            mlflow.log_metric("score_correlation", metrics['score_correlation'])
            mlflow.log_metric("avg_confidence", metrics['avg_confidence'])
            mlflow.log_metric("processing_time_seconds", total_time)
            mlflow.log_metric("texts_per_second", args.num_samples / total_time)
            
            # Log sentiment distributions
            mlflow.log_metric("true_positive_ratio", metrics['true_positive_ratio'])
            mlflow.log_metric("true_negative_ratio", metrics['true_negative_ratio'])
            mlflow.log_metric("true_neutral_ratio", metrics['true_neutral_ratio'])
            mlflow.log_metric("pred_positive_ratio", metrics['pred_positive_ratio'])
            mlflow.log_metric("pred_negative_ratio", metrics['pred_negative_ratio'])
            mlflow.log_metric("pred_neutral_ratio", metrics['pred_neutral_ratio'])
            
            # 9. Display results
            print(f"\n📊 Sentiment Analysis Results:")
            print(f"  ✅ Accuracy: {metrics['accuracy']:.3f}")
            print(f"  📈 Score Correlation: {metrics['score_correlation']:.3f}")
            print(f"  🎯 Average Confidence: {metrics['avg_confidence']:.3f}")
            print(f"  ⏱️  Processing Time: {total_time:.2f}s")
            print(f"  🚀 Throughput: {args.num_samples/total_time:.1f} texts/sec")
            
            print(f"\n📊 Sentiment Distribution:")
            print(f"  True:      Pos={metrics['true_positive_ratio']:.2f}, Neg={metrics['true_negative_ratio']:.2f}, Neu={metrics['true_neutral_ratio']:.2f}")
            print(f"  Predicted: Pos={metrics['pred_positive_ratio']:.2f}, Neg={metrics['pred_negative_ratio']:.2f}, Neu={metrics['pred_neutral_ratio']:.2f}")
            
            # Show sample predictions
            print(f"\n🔍 Sample Predictions:")
            results_df.select("text", "true_sentiment", "predicted_sentiment", "confidence").show(10, truncate=False)
            
            print(f"\n✅ MLflow Run completed: {mlflow.active_run().info.run_id}")
            print(f"🔗 View in MLflow UI: file:./mlruns")
    
    finally:
        # Clean up Spark session
        print("\n🛑 Stopping Spark session...")
        spark.stop()
        print("✅ Spark session stopped")


if __name__ == "__main__":
    main()
