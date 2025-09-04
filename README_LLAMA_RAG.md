# LlamaIndex + Spark + MLflow RAG Integration

## Overview

This integration demonstrates a production-ready Retrieval-Augmented Generation (RAG) system that combines:

- **Apache Spark**: Distributed document processing and text chunking
- **LlamaIndex**: RAG framework with vector indexing and query processing
- **MLflow**: Experiment tracking, metrics logging, and trace capture
- **Multiple LLM backends**: Mock, Ollama (local), and OpenAI (cloud)

## Architecture

```
Documents → Spark Processing → LlamaIndex RAG → MLflow Tracking
    ↓              ↓                ↓              ↓
  TXT/MD      Chunking &       Vector Index    Experiments &
   Files      Metadata        & Embeddings      Traces
```

### Control and functional flow
<img src="images/spark_llamaindex_rag.png"/>


### Key Components

1. **Document Processing Pipeline**
   - Spark DataFrame operations for distributed text processing
   - Configurable chunking with overlap support
   - Metadata preservation (document ID, chunk index, file type)

2. **RAG System**
   - HuggingFace embeddings (local, no API calls)
   - In-memory vector store (LlamaIndex default)
   - Configurable similarity search (top-k retrieval)

3. **LLM Integration**
   - **Mock**: Keyword-based responses for testing
   - **Ollama**: Local LLM processing (privacy-first)
   - **OpenAI**: Cloud API integration

4. **MLflow Integration**
   - Automatic trace capture via `mlflow.llama_index.autolog()`
   - Parameter, metric, and artifact logging
   - Query-level observability with source attribution

## Installation

### Core Dependencies

```bash
# Install base requirements
pip install mlflow pyspark

# Install LlamaIndex components
pip install llama-index llama-index-embeddings-huggingface llama-index-llms-ollama llama-index-llms-openai
```

### Optional: Ollama Setup

For local LLM processing:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2
```

## Usage

### Basic Usage

```bash
# Mock mode (no external dependencies)
python spark/spark_llamaindex_rag.py --llm-type mock --create-sample-docs

# Ollama mode (local LLM)
python spark/spark_llamaindex_rag.py --llm-type ollama --ollama-model llama3.2 --create-sample-docs

# OpenAI mode (requires API key)
export OPENAI_API_KEY="your-api-key"
python spark/spark_llamaindex_rag.py --llm-type openai --create-sample-docs
```

### Advanced Configuration

```bash
python spark/spark_llamaindex_rag.py \
  --llm-type ollama \
  --ollama-model mistral \
  --docs-path ./my_documents \
  --chunk-size 256 \
  --chunk-overlap 25 \
  --embed-model "sentence-transformers/all-MiniLM-L6-v2" \
  --experiment-name "my-rag-experiment"
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--docs-path` | `./documents` | Directory containing TXT/MD files |
| `--llm-type` | `mock` | LLM backend: `mock`, `ollama`, `openai` |
| `--ollama-model` | `llama3.2` | Ollama model name |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |
| `--chunk-size` | `512` | Document chunk size in characters |
| `--chunk-overlap` | `50` | Overlap between chunks |
| `--embed-model` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `--experiment-name` | `spark-llamaindex-rag` | MLflow experiment name |
| `--create-sample-docs` | `False` | Generate sample documents |

## Document Processing

### Supported Formats

- **TXT files**: Plain text documents
- **MD files**: Markdown documents with preserved structure

### Chunking Strategy

The system uses sentence-based chunking with configurable overlap:

1. **Text preprocessing**: Paragraph markers inserted for structure preservation
2. **Sentence splitting**: Documents split at sentence boundaries
3. **Chunk assembly**: Sentences combined until size limit reached
4. **Overlap handling**: Configurable character overlap between chunks

### Metadata Tracking

Each chunk includes:
- `doc_id`: Source document identifier
- `chunk_id`: Unique chunk identifier
- `file_type`: File extension (.txt, .md)
- `chunk_index`: Sequential chunk number
- `chunk_length`: Character count

## RAG System Details

### Vector Storage

**Current Implementation**: In-memory vector store
- **Location**: RAM (not persisted)
- **Performance**: Fast queries, no disk I/O
- **Limitation**: Re-indexing required per run
- **Use case**: Development, testing, demonstrations

### Embedding Model

**Default**: `BAAI/bge-small-en-v1.5`
- **Type**: HuggingFace transformer model
- **Size**: ~133MB download
- **Language**: English optimized
- **Performance**: Good balance of speed and quality

### Query Processing

1. **Question embedding**: User query converted to vector
2. **Similarity search**: Top-k most relevant chunks retrieved
3. **Context assembly**: Retrieved chunks combined with metadata
4. **LLM generation**: Context + question sent to LLM
5. **Response formatting**: Answer with source attribution

## MLflow Integration

### Automatic Logging

The integration uses `mlflow.llama_index.autolog()` to capture:

- **Query traces**: Input questions and generated responses
- **Retrieval context**: Source documents and similarity scores
- **Performance metrics**: Token usage and execution timing
- **Error handling**: Failed queries with error details

### Logged Parameters

```python
{
    "docs_path": "./documents",
    "llm_type": "ollama",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "embed_model": "BAAI/bge-small-en-v1.5",
    "llamaindex_available": True,
    "sklearn_autolog_enabled": False,
    "llamaindex_autolog_enabled": True,
    "ollama_model": "llama3.2",
    "num_questions": 10
}
```

### Logged Metrics

```python
{
    "total_documents": 5,
    "total_chunks": 14,
    "avg_chunk_length": 387.2,
    "total_questions": 10,
    "successful_answers": 10,
    "success_rate": 1.0,
    "avg_answer_length": 432.4,
    "source_retrieval_rate": 1.0
}
```

### Artifacts

- **rag_results.json**: Complete Q&A results with metadata
- **Trace files**: Individual query traces with timing and context

## Performance Characteristics

### Scalability

| Component | Scaling Approach |
|-----------|------------------|
| **Document Loading** | Spark distributed processing |
| **Text Chunking** | Spark UDF parallel execution |
| **Embedding Generation** | Sequential (HuggingFace model) |
| **Vector Search** | In-memory (single-node) |
| **LLM Queries** | Sequential with tracing |

### Memory Usage

- **Spark DataFrames**: Distributed across cluster
- **Vector Index**: Full index in driver memory
- **Embedding Model**: ~133MB model weights
- **Document Chunks**: All chunks loaded for indexing

### Typical Performance

**Small Dataset (5 documents, 14 chunks)**:
- Indexing: ~2-3 seconds
- Query processing: ~800ms per question
- Memory usage: ~200MB (excluding Spark)

## Development Guide

### Project Structure

```
spark/
├── spark_llamaindex_rag.py     # Main RAG script
utils/
├── sample_documents.py         # Document generator utility
├── loader.py                   # MLflow setup utilities
└── spark_utils.py             # Spark session utilities
```

### Key Functions

#### Document Processing
```python
load_documents_with_spark(spark, docs_path)
chunk_documents_with_spark(spark, docs_df, chunk_size, chunk_overlap)
```

#### RAG System
```python
setup_llamaindex_llm(llm_type, model_name, api_key, ollama_base_url)
create_llamaindex_rag(chunks_df, llm, embed_model_name)
```

#### Query Processing
```python
process_questions_with_rag(query_engine, questions)
calculate_rag_metrics(results)
```

### Adding New LLM Backends

1. **Import the LLM class** in the try/except block
2. **Add configuration** in `setup_llamaindex_llm()`
3. **Update argument parser** with new choice
4. **Add parameter logging** in main function

Example for adding Anthropic Claude:
```python
# In imports
from llama_index.llms.anthropic import Anthropic

# In setup_llamaindex_llm()
elif llm_type == "anthropic" and api_key:
    llm = Anthropic(model="claude-3-sonnet", api_key=api_key)
    return llm
```

### Custom Document Formats

To support additional document formats:

1. **Update file discovery** in `load_documents_with_spark()`
2. **Add format-specific processing** if needed
3. **Update metadata extraction** for new file types

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Missing LlamaIndex
pip install llama-index llama-index-embeddings-huggingface

# Missing Ollama integration
pip install llama-index-llms-ollama
```

**Ollama Connection Issues**:
```bash
# Check Ollama status
ollama list

# Start Ollama service
ollama serve

# Pull required model
ollama pull llama3.2
```

**Memory Issues**:
- Reduce `chunk_size` parameter
- Use smaller embedding model
- Process fewer documents per run

**Spark Issues**:
- Check Java installation (required for PySpark)
- Increase driver memory: `--conf spark.driver.memory=4g`
- Verify file permissions on document directory

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

View MLflow traces:
```bash
mlflow ui --backend-store-uri file:./mlruns
# Open http://localhost:5000
```

## Limitations

### Current Constraints

1. **Vector Storage**: In-memory only (not persisted)
2. **Embedding Model**: Single-threaded processing
3. **LLM Queries**: Sequential processing (no batching)
4. **File Formats**: Limited to TXT and MD files
5. **Chunking**: Simple sentence-based approach

### Production Considerations

For production deployment, consider:

1. **Persistent Vector Store**: Chroma, Pinecone, or Weaviate
2. **Distributed Embeddings**: Ray or Dask for parallel processing
3. **Async LLM Calls**: Batch processing for better throughput
4. **Document Parsing**: Support for PDF, DOCX, HTML
5. **Advanced Chunking**: Semantic or hierarchical chunking

## Related Examples

- **LangChain Integration**: `spark_langchain_ollama.py`
- **Multi-mode LangChain**: `spark_langchain_multiple_mode.py`
- **Basic MLflow Tracking**: `tracking/simple_tracking_basic.py`

## References

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Ollama Documentation](https://ollama.ai/docs/)
