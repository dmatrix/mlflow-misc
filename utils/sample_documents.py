"""
Sample Document Generator for RAG Testing

This module provides utilities to create sample documents for testing
RAG (Retrieval-Augmented Generation) systems with various document formats.
"""

from pathlib import Path
from typing import Dict, List


def get_sample_documents() -> Dict[str, str]:
    """
    Get a collection of sample documents for RAG testing.
    
    Returns:
        Dictionary mapping filename to document content
    """
    
    return {
        "machine_learning_basics.md": """# Machine Learning Basics

## What is Machine Learning?

Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.

## Types of Machine Learning

### Supervised Learning
- Uses labeled training data
- Examples: Classification, Regression
- Algorithms: Linear Regression, Random Forest, SVM

### Unsupervised Learning
- Works with unlabeled data
- Examples: Clustering, Dimensionality Reduction
- Algorithms: K-Means, PCA, DBSCAN

### Reinforcement Learning
- Learns through interaction with environment
- Uses rewards and penalties
- Applications: Game playing, Robotics

## Key Concepts

**Training Data**: The dataset used to train the model
**Features**: Input variables used to make predictions
**Labels**: The target variable we want to predict
**Model**: The algorithm that learns patterns from data
**Overfitting**: When a model performs well on training data but poorly on new data
""",

        "python_programming.txt": """Python Programming Guide

Python is a high-level, interpreted programming language known for its simplicity and readability.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support
- Excellent for data science and machine learning

Basic Syntax:
Variables are created by assignment: x = 5
Functions are defined with 'def' keyword
Indentation is used for code blocks
Comments start with # symbol

Data Types:
- Numbers: int, float, complex
- Strings: text data in quotes
- Lists: ordered, mutable collections
- Tuples: ordered, immutable collections
- Dictionaries: key-value pairs
- Sets: unordered collections of unique elements

Control Structures:
- if/elif/else statements for conditions
- for loops for iteration
- while loops for repeated execution
- try/except for error handling

Popular Libraries:
- NumPy: Numerical computing
- Pandas: Data manipulation
- Matplotlib: Data visualization
- Scikit-learn: Machine learning
- TensorFlow/PyTorch: Deep learning
""",

        "data_science_workflow.md": """# Data Science Workflow

## Overview

The data science workflow is a systematic approach to solving problems using data. It involves several key stages that help ensure reproducible and reliable results.

## Stages of Data Science Workflow

### 1. Problem Definition
- Clearly define the business problem
- Identify success metrics
- Determine project scope and constraints

### 2. Data Collection
- Identify relevant data sources
- Gather data from various sources (databases, APIs, files)
- Ensure data quality and completeness

### 3. Data Exploration and Analysis
- Examine data structure and characteristics
- Identify patterns, trends, and anomalies
- Perform statistical analysis
- Create visualizations

### 4. Data Preprocessing
- Clean and prepare data for modeling
- Handle missing values
- Remove outliers
- Feature engineering and selection
- Data transformation and normalization

### 5. Model Development
- Select appropriate algorithms
- Train multiple models
- Tune hyperparameters
- Cross-validation for model selection

### 6. Model Evaluation
- Assess model performance using appropriate metrics
- Test on unseen data
- Compare different models
- Validate business impact

### 7. Deployment and Monitoring
- Deploy model to production environment
- Monitor model performance over time
- Update model as needed
- Maintain model documentation

## Best Practices

- Document all steps and decisions
- Use version control for code and data
- Implement proper testing procedures
- Ensure reproducibility
- Collaborate effectively with stakeholders
""",

        "mlflow_guide.txt": """MLflow: Machine Learning Lifecycle Management

MLflow is an open-source platform for managing the machine learning lifecycle, including experimentation, reproducibility, deployment, and a central model registry.

Core Components:

MLflow Tracking:
- Log parameters, metrics, and artifacts
- Compare experiment runs
- Organize experiments by project
- Track model lineage and versions

MLflow Projects:
- Package ML code in reusable format
- Specify dependencies and entry points
- Enable reproducible runs across environments
- Support for different execution backends

MLflow Models:
- Standard format for packaging ML models
- Support for multiple frameworks (scikit-learn, TensorFlow, PyTorch)
- Easy deployment to various platforms
- Model signature and input/output schema

MLflow Model Registry:
- Centralized model store
- Model versioning and stage transitions
- Collaborative model management
- Integration with CI/CD pipelines

Key Benefits:
- Experiment tracking and comparison
- Reproducible ML workflows
- Easy model deployment
- Collaboration across teams
- Integration with popular ML libraries

Common Use Cases:
- Hyperparameter tuning experiments
- Model performance comparison
- A/B testing of models
- Model deployment automation
- Regulatory compliance and audit trails

Getting Started:
1. Install MLflow: pip install mlflow
2. Start tracking server: mlflow ui
3. Log experiments in your code
4. Compare results in the web UI
5. Deploy best models to production
""",

        "spark_fundamentals.md": """# Apache Spark Fundamentals

## What is Apache Spark?

Apache Spark is a unified analytics engine for large-scale data processing. It provides high-level APIs in Java, Scala, Python, and R, and an optimized engine that supports general execution graphs.

## Key Features

### Speed
- In-memory computing capabilities
- Advanced DAG execution engine
- Code generation and optimization

### Ease of Use
- Simple APIs for complex operations
- Interactive shells for Python, Scala, and R
- Rich set of higher-level tools

### Generality
- Combines SQL, streaming, and complex analytics
- Unified engine for diverse workloads
- Rich ecosystem of libraries

### Runs Everywhere
- Hadoop clusters, Kubernetes, standalone
- Cloud providers (AWS, Azure, GCP)
- Local development environments

## Core Concepts

### RDD (Resilient Distributed Dataset)
- Fundamental data structure in Spark
- Immutable distributed collection
- Fault-tolerant through lineage
- Lazy evaluation

### DataFrame and Dataset
- Higher-level abstraction over RDD
- Structured data with schema
- Catalyst optimizer for query optimization
- Type-safe operations (Dataset)

### Spark SQL
- Module for working with structured data
- SQL queries on DataFrames
- Integration with Hive and other data sources
- Catalyst optimizer for performance

### Spark Streaming
- Scalable, high-throughput, fault-tolerant stream processing
- Micro-batch processing model
- Integration with Kafka, Flume, HDFS
- Structured Streaming for continuous applications

## Architecture

### Driver Program
- Contains the main function
- Creates SparkContext
- Coordinates execution

### Cluster Manager
- Manages resources across cluster
- Types: Standalone, YARN, Mesos, Kubernetes

### Executors
- Run tasks and store data
- Report results back to driver
- Provide in-memory storage for cached data

## Common Operations

### Transformations
- map(), filter(), groupBy()
- join(), union(), distinct()
- Lazy evaluation - not executed immediately

### Actions
- collect(), count(), save()
- reduce(), foreach(), take()
- Trigger execution of transformations

## Best Practices

- Use DataFrames/Datasets over RDDs when possible
- Cache frequently accessed data
- Avoid shuffles when possible
- Tune partition sizes appropriately
- Monitor and optimize resource usage
"""
    }


def create_sample_documents(docs_path: str = "./documents") -> List[str]:
    """
    Create sample documents for RAG testing.
    
    Args:
        docs_path: Directory path where documents will be created
        
    Returns:
        List of created filenames
    """
    
    docs_dir = Path(docs_path)
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Creating sample documents in {docs_path}...")
    
    sample_docs = get_sample_documents()
    created_files = []
    
    for filename, content in sample_docs.items():
        file_path = docs_dir / filename
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            created_files.append(filename)
    
    if created_files:
        print(f"✅ Created {len(created_files)} sample documents:")
        for filename in created_files:
            print(f"   - {filename}")
    else:
        print("✅ Sample documents already exist")
    
    return created_files


def list_available_documents() -> List[str]:
    """
    List all available sample document templates.
    
    Returns:
        List of available document filenames
    """
    return list(get_sample_documents().keys())


def get_document_info() -> Dict[str, Dict[str, str]]:
    """
    Get information about available sample documents.
    
    Returns:
        Dictionary mapping filename to document metadata
    """
    
    sample_docs = get_sample_documents()
    doc_info = {}
    
    for filename, content in sample_docs.items():
        file_type = "Markdown" if filename.endswith('.md') else "Text"
        word_count = len(content.split())
        char_count = len(content)
        
        doc_info[filename] = {
            "type": file_type,
            "word_count": str(word_count),
            "char_count": str(char_count),
            "topic": filename.split('.')[0].replace('_', ' ').title()
        }
    
    return doc_info


def demo_sample_documents() -> None:
    """Demonstrate the sample document generator."""
    
    print("📚 Sample Document Generator Demo")
    print("=" * 50)
    
    # Show available documents
    print("\n📋 Available Documents:")
    doc_info = get_document_info()
    for filename, info in doc_info.items():
        print(f"  - {filename} ({info['type']}, {info['word_count']} words)")
        print(f"    Topic: {info['topic']}")
    
    # Create documents
    print(f"\n📁 Creating documents...")
    created = create_sample_documents("./demo_documents")
    
    print(f"\n✅ Demo completed! Created {len(created)} documents in ./demo_documents/")


if __name__ == "__main__":
    demo_sample_documents()
