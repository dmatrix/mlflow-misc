"""
Spark + LangChain Utilities for MLflow Integration

This module provides common utilities for integrating LangChain with Spark
and MLflow for distributed NLP workflows.
"""

import time
from typing import List, Dict, Any, Optional, Callable
from functools import wraps

# LangChain imports with fallback handling
try:
    from langchain.schema import BaseMessage, HumanMessage
    from langchain.callbacks import get_openai_callback
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import FakeListLLM
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def langchain_available_check(func):
    """Decorator to check if LangChain is available before executing function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not available. Install with: uv sync --extra langchain"
            )
        return func(*args, **kwargs)
    return wrapper


@langchain_available_check
def create_openai_llm(
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.1,
    max_tokens: int = 100,
    api_key: Optional[str] = None
) -> ChatOpenAI:
    """
    Create an OpenAI LLM instance for LangChain.
    
    Args:
        model: OpenAI model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        api_key: OpenAI API key
        
    Returns:
        ChatOpenAI instance
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key
    )


def create_mock_llm(responses: List[str]) -> "FakeListLLM":
    """
    Create a mock LLM for testing and demonstration.
    
    Args:
        responses: List of mock responses
        
    Returns:
        FakeListLLM instance
    """
    if not LANGCHAIN_AVAILABLE:
        return None
    
    return FakeListLLM(responses=responses)


def create_sentiment_mock_llm() -> "FakeListLLM":
    """Create a mock LLM specifically for sentiment analysis."""
    sentiment_responses = [
        "positive: 0.8",
        "negative: -0.7", 
        "neutral: 0.1",
        "positive: 0.9",
        "negative: -0.6",
        "neutral: -0.1",
        "positive: 0.7",
        "negative: -0.8",
        "neutral: 0.2",
        "positive: 0.6",
        "negative: -0.9",
        "neutral: 0.0"
    ]
    return create_mock_llm(sentiment_responses * 100)  # Repeat for multiple calls


def batch_process_with_llm(
    texts: List[str],
    llm,
    processing_func: Callable,
    batch_size: int = 50,
    delay_between_batches: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Process texts in batches with an LLM to avoid rate limits.
    
    Args:
        texts: List of texts to process
        llm: LangChain LLM instance
        processing_func: Function to process each text
        batch_size: Size of each batch
        delay_between_batches: Delay between batches in seconds
        
    Returns:
        List of processing results
    """
    all_results = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        print(f"🔄 Processing batch {batch_num + 1}/{total_batches} ({len(batch_texts)} texts)...")
        
        batch_start_time = time.time()
        batch_results = []
        
        for i, text in enumerate(batch_texts):
            try:
                result = processing_func(text, llm)
                result['batch_id'] = batch_num + 1
                result['text_index'] = start_idx + i
                batch_results.append(result)
            except Exception as e:
                print(f"⚠️  Error processing text {start_idx + i}: {e}")
                batch_results.append({
                    'error': str(e),
                    'batch_id': batch_num + 1,
                    'text_index': start_idx + i
                })
        
        batch_time = time.time() - batch_start_time
        print(f"✅ Batch {batch_num + 1} completed in {batch_time:.2f}s")
        
        all_results.extend(batch_results)
        
        # Delay between batches to respect rate limits
        if batch_num < total_batches - 1 and delay_between_batches > 0:
            time.sleep(delay_between_batches)
    
    return all_results


def parse_llm_sentiment_response(response: str) -> Dict[str, Any]:
    """
    Parse LLM response for sentiment analysis.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Parsed sentiment data
    """
    response_text = str(response).strip().lower()
    
    # Default values
    sentiment = 'neutral'
    score = 0.0
    confidence = 0.5
    
    try:
        # Look for sentiment keywords
        if 'positive' in response_text:
            sentiment = 'positive'
            confidence = 0.8
        elif 'negative' in response_text:
            sentiment = 'negative'
            confidence = 0.8
        elif 'neutral' in response_text:
            sentiment = 'neutral'
            confidence = 0.6
        
        # Try to extract numerical score
        if ':' in response_text:
            try:
                score_part = response_text.split(':')[1].strip()
                score = float(score_part)
                confidence = abs(score)
            except (ValueError, IndexError):
                pass
        
        # Ensure score is in valid range
        score = max(-1.0, min(1.0, score))
        confidence = max(0.0, min(1.0, confidence))
        
    except Exception as e:
        print(f"⚠️  Error parsing response '{response}': {e}")
    
    return {
        'sentiment': sentiment,
        'score': score,
        'confidence': confidence,
        'raw_response': str(response)
    }


def create_sentiment_prompt(text: str, format_instructions: bool = True) -> str:
    """
    Create a standardized prompt for sentiment analysis.
    
    Args:
        text: Text to analyze
        format_instructions: Whether to include format instructions
        
    Returns:
        Formatted prompt string
    """
    base_prompt = f"""Analyze the sentiment of the following text:

Text: "{text}"

Determine if the sentiment is positive, negative, or neutral."""
    
    if format_instructions:
        base_prompt += """

Respond with only the sentiment and a confidence score between -1.0 and 1.0, formatted as:
sentiment: score

Examples:
positive: 0.8
negative: -0.7
neutral: 0.1"""
    
    return base_prompt


def calculate_text_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate basic text metrics for a collection of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary with text metrics
    """
    if not texts:
        return {}
    
    text_lengths = [len(text) for text in texts]
    word_counts = [len(text.split()) for text in texts]
    
    return {
        'total_texts': len(texts),
        'avg_text_length': sum(text_lengths) / len(text_lengths),
        'min_text_length': min(text_lengths),
        'max_text_length': max(text_lengths),
        'avg_word_count': sum(word_counts) / len(word_counts),
        'min_word_count': min(word_counts),
        'max_word_count': max(word_counts),
        'total_characters': sum(text_lengths),
        'total_words': sum(word_counts)
    }


@langchain_available_check
def track_openai_usage(func):
    """
    Decorator to track OpenAI API usage with LangChain callbacks.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with usage tracking
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with get_openai_callback() as cb:
            result = func(*args, **kwargs)
            
            # Add usage info to result if it's a dict
            if isinstance(result, dict):
                result['openai_usage'] = {
                    'total_tokens': cb.total_tokens,
                    'prompt_tokens': cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens,
                    'total_cost': cb.total_cost
                }
            
            print(f"💰 OpenAI Usage - Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
            
            return result
    
    return wrapper


def create_text_classification_prompt(
    text: str,
    categories: List[str],
    include_confidence: bool = True
) -> str:
    """
    Create a prompt for text classification tasks.
    
    Args:
        text: Text to classify
        categories: List of possible categories
        include_confidence: Whether to request confidence scores
        
    Returns:
        Formatted classification prompt
    """
    categories_str = ", ".join(categories)
    
    prompt = f"""Classify the following text into one of these categories: {categories_str}

Text: "{text}"

Choose the most appropriate category."""
    
    if include_confidence:
        prompt += f"""

Respond with the category and a confidence score between 0.0 and 1.0, formatted as:
category: confidence

Example: {categories[0]}: 0.85"""
    
    return prompt


def validate_langchain_setup() -> Dict[str, Any]:
    """
    Validate LangChain setup and return status information.
    
    Returns:
        Dictionary with setup status
    """
    status = {
        'langchain_available': LANGCHAIN_AVAILABLE,
        'components': {}
    }
    
    if LANGCHAIN_AVAILABLE:
        try:
            # Test basic imports
            from langchain.schema import BaseMessage
            status['components']['schema'] = True
        except ImportError:
            status['components']['schema'] = False
        
        try:
            from langchain_openai import ChatOpenAI
            status['components']['openai'] = True
        except ImportError:
            status['components']['openai'] = False
        
        try:
            from langchain_community.llms import FakeListLLM
            status['components']['community'] = True
        except ImportError:
            status['components']['community'] = False
        
        try:
            from langchain.callbacks import get_openai_callback
            status['components']['callbacks'] = True
        except ImportError:
            status['components']['callbacks'] = False
    
    return status
