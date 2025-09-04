"""
Spark + MLflow + LlamaIndex Integration: RAG Document Q&A
========================================================

A comprehensive example demonstrating LlamaIndex RAG integration:
1. Spark for distributed document processing and chunking
2. LlamaIndex for RAG (Retrieval-Augmented Generation) with multiple LLM backends
3. MLflow for experiment tracking and artifact storage
4. Support for TXT and MD document formats
5. Using existing utility functions for clean, maintainable code

Focus: Document Q&A system with distributed processing and RAG

LLM Options:
- Mock: Simple keyword-based responses (no dependencies, for testing)
- Ollama: Local LLM processing via LlamaIndex (privacy-first)
- OpenAI: Cloud LLM API with LlamaIndex integration

Document Formats Supported:
- TXT: Plain text files
- MD: Markdown files

Usage:
    python spark_llamaindex_rag.py --llm-type mock --docs-path ./documents/
    python spark_llamaindex_rag.py --llm-type ollama --ollama-model llama3.2 --docs-path ./documents/
    python spark_llamaindex_rag.py --llm-type openai --docs-path ./documents/  # requires OPENAI_API_KEY
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add utils to path for imports
sys.path.append(str(Path(__file__).parent.parent / "utils"))

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, input_file_name
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType

# LlamaIndex imports with fallback
try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama
    from llama_index.llms.openai import OpenAI
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    print("⚠️  LlamaIndex not available. Install with: pip install llama-index llama-index-embeddings-huggingface llama-index-llms-ollama llama-index-llms-openai")

# Import our utility functions
from loader import load_mlflow_setup
from spark_utils import create_mlflow_spark_session
from sample_documents import create_sample_documents


def load_documents_with_spark(spark: SparkSession, docs_path: str):
    """Load TXT and MD documents using Spark for distributed processing."""
    
    print(f"📚 Loading documents from {docs_path}...")
    
    docs_dir = Path(docs_path)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_path}")
    
    # Find TXT and MD files
    txt_files = list(docs_dir.glob("*.txt"))
    md_files = list(docs_dir.glob("*.md"))
    
    if not txt_files and not md_files:
        raise FileNotFoundError(f"No TXT or MD files found in {docs_path}")
    
    print(f"   Found {len(txt_files)} TXT files and {len(md_files)} MD files")
    
    # Load documents using Spark
    all_files = [str(f) for f in txt_files + md_files]
    
    # Read all text files into a single DataFrame
    df = spark.read.text(all_files, wholetext=True)
    
    # Add filename and file type columns
    df = df.withColumn("filename", input_file_name())
    
    def extract_file_info(filepath: str) -> tuple:
        """Extract filename and file type from full path."""
        path = Path(filepath)
        return path.name, path.suffix.lower()
    
    # Register UDF for file info extraction
    file_info_udf = spark.udf.register("extract_file_info", extract_file_info, 
                                      StructType([
                                          StructField("name", StringType(), True),
                                          StructField("type", StringType(), True)
                                      ]))
    
    # Apply UDF to extract file information
    df = df.withColumn("file_info", file_info_udf(col("filename"))) \
           .withColumn("doc_name", col("file_info.name")) \
           .withColumn("file_type", col("file_info.type")) \
           .drop("file_info", "filename")
    
    # Add document ID
    df = df.withColumn("doc_id", col("doc_name"))
    
    print(f"✅ Loaded {df.count()} documents into Spark DataFrame")
    
    return df


def chunk_documents_with_spark(spark: SparkSession, docs_df, chunk_size: int = 512, chunk_overlap: int = 50):
    """Chunk documents using Spark UDFs for distributed processing."""
    
    print(f"✂️  Chunking documents (size: {chunk_size}, overlap: {chunk_overlap})...")
    
    def chunk_text(text: str, doc_id: str, file_type: str) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces with overlap.
        
        This UDF will be executed on each Spark executor for distributed processing.
        """
        if not text or text.strip() == "":
            return []
        
        # Simple sentence-based chunking
        sentences = text.replace('\n\n', ' [PARAGRAPH] ').split('. ')
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'doc_id': doc_id,
                    'chunk_id': f"{doc_id}_chunk_{chunk_id}",
                    'chunk_text': current_chunk.strip(),
                    'file_type': file_type,
                    'chunk_index': chunk_id,
                    'chunk_length': len(current_chunk.strip())
                })
                
                # Start new chunk with overlap
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    current_chunk = current_chunk[-chunk_overlap:] + " " + sentence
                else:
                    current_chunk = sentence
                chunk_id += 1
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk if it exists
        if current_chunk.strip():
            chunks.append({
                'doc_id': doc_id,
                'chunk_id': f"{doc_id}_chunk_{chunk_id}",
                'chunk_text': current_chunk.strip(),
                'file_type': file_type,
                'chunk_index': chunk_id,
                'chunk_length': len(current_chunk.strip())
            })
        
        return chunks
    
    # Define return schema for the UDF
    chunk_schema = ArrayType(StructType([
        StructField("doc_id", StringType(), True),
        StructField("chunk_id", StringType(), True),
        StructField("chunk_text", StringType(), True),
        StructField("file_type", StringType(), True),
        StructField("chunk_index", IntegerType(), True),
        StructField("chunk_length", IntegerType(), True)
    ]))
    
    # Register UDF
    chunk_udf = spark.udf.register("chunk_text", chunk_text, chunk_schema)
    
    # Apply chunking UDF
    chunked_df = docs_df.withColumn("chunks", chunk_udf(col("value"), col("doc_id"), col("file_type")))
    
    # Explode chunks to create one row per chunk
    from pyspark.sql.functions import explode
    chunks_df = chunked_df.select(explode(col("chunks")).alias("chunk_data")) \
                          .select("chunk_data.*")
    
    # Cache for better performance
    chunks_df.cache()
    
    chunk_count = chunks_df.count()
    print(f"✅ Created {chunk_count} chunks from documents")
    
    return chunks_df


def setup_llamaindex_llm(llm_type: str = "mock", model_name: str = "llama3.2", 
                        api_key: str = None, ollama_base_url: str = "http://localhost:11434"):
    """Setup LlamaIndex LLM with support for Mock, Ollama, or OpenAI."""
    
    if not LLAMAINDEX_AVAILABLE:
        print("⚠️  LlamaIndex not available. Using mock responses.")
        return None
    
    if llm_type == "ollama":
        print(f"🦙 Using Ollama model '{model_name}' via LlamaIndex")
        print(f"🔗 Connecting to Ollama at: {ollama_base_url}")
        try:
            llm = Ollama(model=model_name, base_url=ollama_base_url, request_timeout=30.0)
            
            # Test connection
            print("🧪 Testing Ollama connection...")
            test_response = llm.complete("Hello, respond with just 'OK'")
            print(f"✅ Ollama connection successful! Test response: {str(test_response)[:50]}...")
            
            return llm
        except Exception as e:
            print(f"⚠️  Failed to connect to Ollama: {e}")
            print("💡 Make sure Ollama is running and the model is installed")
            print(f"   Try: ollama pull {model_name}")
            return None
            
    elif llm_type == "openai" and api_key:
        print("🤖 Using OpenAI GPT via LlamaIndex")
        try:
            llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0.1)
            
            # Test connection
            print("🧪 Testing OpenAI connection...")
            test_response = llm.complete("Hello, respond with just 'OK'")
            print(f"✅ OpenAI connection successful! Test response: {str(test_response)[:50]}...")
            
            return llm
        except Exception as e:
            print(f"⚠️  Failed to connect to OpenAI: {e}")
            return None
    else:
        print("🎭 Using mock LLM for demonstration")
        return None


class MockQueryEngine:
    """Mock query engine for demonstration when LlamaIndex is not available."""
    
    def __init__(self, chunks_data: List[Dict]):
        self.chunks_data = chunks_data
        
    def query(self, question: str) -> str:
        """Simple keyword-based mock responses."""
        question_lower = question.lower()
        
        # Simple keyword matching for demo
        if any(word in question_lower for word in ['machine learning', 'ml', 'model']):
            return "Machine learning is a subset of AI that enables computers to learn from data. It includes supervised, unsupervised, and reinforcement learning approaches."
        elif any(word in question_lower for word in ['python', 'programming']):
            return "Python is a high-level programming language known for its simplicity and extensive libraries for data science and machine learning."
        elif any(word in question_lower for word in ['spark', 'apache']):
            return "Apache Spark is a unified analytics engine for large-scale data processing with in-memory computing capabilities."
        elif any(word in question_lower for word in ['mlflow', 'tracking']):
            return "MLflow is an open-source platform for managing the machine learning lifecycle, including experiment tracking and model deployment."
        elif any(word in question_lower for word in ['data science', 'workflow']):
            return "Data science workflow includes problem definition, data collection, exploration, preprocessing, modeling, evaluation, and deployment."
        else:
            return f"Based on the available documents, I found information related to your question about '{question}'. The documents contain details about machine learning, Python programming, Spark, MLflow, and data science workflows."


def create_llamaindex_rag(chunks_df, llm, embed_model_name: str = "BAAI/bge-small-en-v1.5"):
    """Create LlamaIndex RAG system from Spark DataFrame chunks."""
    
    print("🔍 Creating LlamaIndex RAG system...")
    
    if not LLAMAINDEX_AVAILABLE or llm is None:
        print("⚠️  Using mock query engine")
        # Convert Spark DataFrame to list for mock engine
        chunks_data = chunks_df.collect()
        return MockQueryEngine([row.asDict() for row in chunks_data])
    
    try:
        # Setup embedding model (local HuggingFace model)
        print(f"📊 Loading embedding model: {embed_model_name}")
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        
        # Configure LlamaIndex settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        # Convert Spark DataFrame to LlamaIndex Documents
        print("📄 Converting chunks to LlamaIndex documents...")
        chunks_data = chunks_df.collect()
        
        documents = []
        for row in chunks_data:
            doc = Document(
                text=row.chunk_text,
                metadata={
                    'doc_id': row.doc_id,
                    'chunk_id': row.chunk_id,
                    'file_type': row.file_type,
                    'chunk_index': row.chunk_index,
                    'chunk_length': row.chunk_length
                }
            )
            documents.append(doc)
        
        print(f"✅ Created {len(documents)} LlamaIndex documents")
        
        # Create vector index
        print("🔮 Building vector index...")
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        
        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=3,  # Return top 3 most relevant chunks
            response_mode="compact"  # Compact response mode
        )
        
        print("✅ RAG system ready!")
        return query_engine
        
    except Exception as e:
        print(f"⚠️  Error creating RAG system: {e}")
        print("   Falling back to mock query engine")
        chunks_data = chunks_df.collect()
        return MockQueryEngine([row.asDict() for row in chunks_data])


def generate_sample_questions() -> List[str]:
    """Generate sample questions for testing the RAG system."""
    
    return [
        "What is machine learning and what are its main types?",
        "How does Python support data science and machine learning?",
        "What are the key features of Apache Spark?",
        "What is MLflow and how does it help with ML lifecycle management?",
        "What are the main stages of a data science workflow?",
        "What is the difference between supervised and unsupervised learning?",
        "How does Spark handle distributed data processing?",
        "What are the core components of MLflow?",
        "What are RDDs in Apache Spark?",
        "What are some best practices for data science projects?"
    ]


def process_questions_with_rag(query_engine, questions: List[str]):
    """Process questions using the RAG system and return results."""
    
    print(f"❓ Processing {len(questions)} questions with RAG...")
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"   Question {i}/{len(questions)}: {question[:60]}...")
        
        try:
            if hasattr(query_engine, 'query'):
                response = query_engine.query(question)
                
                # Handle different response types
                if hasattr(response, 'response'):
                    answer = str(response.response)
                    # Get source information if available
                    sources = []
                    if hasattr(response, 'source_nodes'):
                        for node in response.source_nodes[:2]:  # Top 2 sources
                            if hasattr(node, 'metadata'):
                                sources.append(node.metadata.get('doc_id', 'Unknown'))
                else:
                    answer = str(response)
                    sources = []
            else:
                answer = "Error: Query engine not properly initialized"
                sources = []
            
            results.append({
                'question': question,
                'answer': answer,
                'sources': sources,
                'answer_length': len(answer)
            })
            
        except Exception as e:
            print(f"      ⚠️  Error processing question: {e}")
            results.append({
                'question': question,
                'answer': f"Error processing question: {str(e)}",
                'sources': [],
                'answer_length': 0
            })
    
    print(f"✅ Processed {len(results)} questions")
    return results


def calculate_rag_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate metrics for RAG system performance."""
    
    if not results:
        return {'total_questions': 0, 'avg_answer_length': 0, 'success_rate': 0}
    
    successful_answers = [r for r in results if not r['answer'].startswith('Error')]
    
    avg_answer_length = sum(r['answer_length'] for r in successful_answers) / len(successful_answers) if successful_answers else 0
    success_rate = len(successful_answers) / len(results)
    
    # Count questions with sources
    questions_with_sources = sum(1 for r in results if r['sources'])
    source_retrieval_rate = questions_with_sources / len(results)
    
    return {
        'total_questions': len(results),
        'successful_answers': len(successful_answers),
        'avg_answer_length': avg_answer_length,
        'success_rate': success_rate,
        'source_retrieval_rate': source_retrieval_rate
    }


def main():
    """Main function demonstrating Spark + MLflow + LlamaIndex RAG integration."""
    
    parser = argparse.ArgumentParser(description='Spark + MLflow + LlamaIndex RAG Integration')
    parser.add_argument('--docs-path', default='./documents',
                       help='Path to documents directory (default: ./documents)')
    parser.add_argument('--experiment-name', default='spark-llamaindex-rag',
                       help='MLflow experiment name')
    parser.add_argument('--llm-type', choices=['mock', 'ollama', 'openai'], default='mock',
                       help='LLM type to use (default: mock)')
    parser.add_argument('--ollama-model', default='llama3.2',
                       help='Ollama model name (default: llama3.2)')
    parser.add_argument('--ollama-url', default='http://localhost:11434',
                       help='Ollama base URL (default: http://localhost:11434)')
    parser.add_argument('--chunk-size', type=int, default=512,
                       help='Document chunk size (default: 512)')
    parser.add_argument('--chunk-overlap', type=int, default=50,
                       help='Chunk overlap size (default: 50)')
    parser.add_argument('--embed-model', default='BAAI/bge-small-en-v1.5',
                       help='Embedding model name (default: BAAI/bge-small-en-v1.5)')
    parser.add_argument('--create-sample-docs', action='store_true',
                       help='Create sample documents if they don\'t exist')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🦙 Spark + MLflow + LlamaIndex RAG Integration")
    print("=" * 70)
    print(f"📁 Documents Path: {args.docs_path}")
    print(f"🤖 LLM Type: {args.llm_type}")
    if args.llm_type == 'ollama':
        print(f"🦙 Ollama Model: {args.ollama_model}")
        print(f"🔗 Ollama URL: {args.ollama_url}")
    print(f"✂️  Chunk Size: {args.chunk_size}")
    print(f"🔄 Chunk Overlap: {args.chunk_overlap}")
    print(f"📊 Embedding Model: {args.embed_model}")
    print(f"🦙 LlamaIndex: {'Available' if LLAMAINDEX_AVAILABLE else 'Not available'}")
    print("=" * 70)
    
    # Create sample documents if requested
    if args.create_sample_docs:
        create_sample_documents(args.docs_path)
    
    # End any existing MLflow run
    mlflow.end_run()
    
    # 1. Setup MLflow tracking (without sklearn autolog since we're doing RAG, not ML training)
    mlflow_setup = load_mlflow_setup()
    experiment_id = mlflow_setup.setup_mlflow_tracking(
        tracking_uri="file:./mlruns",
        experiment_name=args.experiment_name,
        enable_autolog=False  # sklearn autolog not needed for RAG operations
    )
    
    # 2. Enable framework-specific autologging for RAG operations
    autolog_enabled = False
    if LLAMAINDEX_AVAILABLE and args.llm_type != 'mock':
        try:
            mlflow.llama_index.autolog()
            autolog_enabled = True
            print("✅ MLflow LlamaIndex autologging enabled (captures RAG traces)")
        except AttributeError:
            print("⚠️  MLflow LlamaIndex autolog not available in this version")
        except Exception as e:
            print(f"⚠️  Could not enable LlamaIndex autolog: {e}")
    else:
        print("ℹ️  LlamaIndex autolog skipped (mock mode or LlamaIndex unavailable)")
    
    print(f"📊 Autolog Status: sklearn=False, llamaindex={autolog_enabled}")
    
    # 3. Initialize Spark
    print("🚀 Initializing Spark session...")
    spark = create_mlflow_spark_session(
        app_name="MLflow-Spark-LlamaIndex-RAG"
    )
    
    try:
        with mlflow.start_run(run_name="llamaindex_rag_qa"):
            
            # Log parameters
            mlflow.log_param("docs_path", args.docs_path)
            mlflow.log_param("llm_type", args.llm_type)
            mlflow.log_param("chunk_size", args.chunk_size)
            mlflow.log_param("chunk_overlap", args.chunk_overlap)
            mlflow.log_param("embed_model", args.embed_model)
            mlflow.log_param("llamaindex_available", LLAMAINDEX_AVAILABLE)
            
            # Log autolog status for both frameworks
            mlflow.log_param("sklearn_autolog_enabled", False)  # Disabled for RAG use case
            mlflow.log_param("llamaindex_autolog_enabled", autolog_enabled)
            
            if args.llm_type == 'ollama':
                mlflow.log_param("ollama_model", args.ollama_model)
                mlflow.log_param("ollama_url", args.ollama_url)
            
            # 4. Load documents with Spark
            docs_df = load_documents_with_spark(spark, args.docs_path)
            
            # Show sample documents
            print("\n📚 Sample Documents:")
            docs_df.select("doc_name", "file_type").show(5, truncate=False)
            
            # Log document statistics
            doc_count = docs_df.count()
            mlflow.log_metric("total_documents", doc_count)
            
            # 5. Chunk documents with Spark
            chunks_df = chunk_documents_with_spark(spark, docs_df, args.chunk_size, args.chunk_overlap)
            
            # Show sample chunks
            print("\n✂️  Sample Chunks:")
            chunks_df.select("doc_id", "chunk_index", "chunk_length").show(5, truncate=False)
            
            # Log chunk statistics
            chunk_count = chunks_df.count()
            avg_chunk_length = chunks_df.agg({"chunk_length": "avg"}).collect()[0][0]
            mlflow.log_metric("total_chunks", chunk_count)
            mlflow.log_metric("avg_chunk_length", avg_chunk_length)
            
            # 6. Setup LlamaIndex LLM
            api_key = os.getenv('OPENAI_API_KEY') if args.llm_type == 'openai' else None
            llm = setup_llamaindex_llm(
                llm_type=args.llm_type,
                model_name=args.ollama_model,
                api_key=api_key,
                ollama_base_url=args.ollama_url
            )
            
            if llm is None and args.llm_type != 'mock':
                print("⚠️  Proceeding with mock responses...")
                mlflow.log_param("llm_status", "failed_fallback_used")
            else:
                mlflow.log_param("llm_status", "connected" if llm else "mock")
            
            # 7. Create RAG system
            query_engine = create_llamaindex_rag(chunks_df, llm, args.embed_model)
            
            # 8. Generate and process questions
            questions = generate_sample_questions()
            mlflow.log_param("num_questions", len(questions))
            
            print(f"\n❓ Sample Questions:")
            for i, q in enumerate(questions[:3], 1):
                print(f"   {i}. {q}")
            
            # Process questions with RAG
            results = process_questions_with_rag(query_engine, questions)
            
            # 9. Calculate and log metrics
            metrics = calculate_rag_metrics(results)
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # 10. Display results
            print(f"\n📊 RAG System Results:")
            print(f"  ✅ Questions Processed: {metrics['total_questions']}")
            print(f"  ✅ Success Rate: {metrics['success_rate']:.3f}")
            print(f"  ✅ Avg Answer Length: {metrics['avg_answer_length']:.1f} chars")
            print(f"  ✅ Source Retrieval Rate: {metrics['source_retrieval_rate']:.3f}")
            
            print(f"\n🔍 Sample Q&A:")
            for i, result in enumerate(results[:3], 1):
                print(f"\n   Q{i}: {result['question']}")
                print(f"   A{i}: {result['answer'][:150]}...")
                if result['sources']:
                    print(f"   Sources: {', '.join(result['sources'])}")
            
            # 11. Save results as artifact
            import json
            results_file = "rag_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            mlflow.log_artifact(results_file)
            os.remove(results_file)  # Clean up
            
            print(f"\n✅ MLflow Run completed: {mlflow.active_run().info.run_id}")
            print(f"🔗 View in MLflow UI: file:./mlruns")
    
    finally:
        print("\n🛑 Stopping Spark session...")
        spark.stop()
        print("✅ Spark session stopped")


if __name__ == "__main__":
    main()
