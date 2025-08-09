# CRag

## Problem Statement

Retrieving accurate answers from large document collections is often slow and imprecise. Traditional retrieval-augmented generation (RAG) pipelines face challenges such as:
- High latency due to repeated context retrieval for similar queries
- Inaccurate answers caused by irrelevant or outdated context
- Increased computational load and database queries for frequently asked questions

These issues make it difficult to deliver fast and reliable answers in real-world applications where users may ask similar or related questions repeatedly.

## Solution

CRag solves these problems by integrating context caching into the RAG pipeline. The key features of CRag are:
- **Context Caching:** Stores relevant context chunks for previously asked or similar queries.
- **Cache Lookup:** Checks the cache before querying the vector database, reducing redundant retrievals.
- **Efficient Answer Generation:** Uses cached context to generate answers quickly and accurately.
- **Automatic Cache Update:** Updates the cache with new context when a query is not found, ensuring freshness and relevance.

This approach significantly reduces response times, improves answer accuracy, and lowers the load on the vector database.

## Workflow

Below is the workflow for CRag. Add your workflow flow chart in the [Image] box:

<img width="4680" height="3395" alt="image" src="https://github.com/user-attachments/assets/bc31427d-00d4-4410-a9bb-8dcf4f56fe4b" />


*Add your workflow flow chart here.*

**Workflow Steps:**
1. **User Query:** The user submits a question.
2. **Cache Lookup:** The system checks if a similar query's context exists in the cache.
3. **Context Retrieval:**
    - If cached, use the stored context.
    - If not cached, retrieve context from the vector database.
4. **Answer Generation:** The RAG pipeline generates an answer using the retrieved context.
5. **Cache Update:** If new context was retrieved, store it in the cache for future queries.

## How Context Caching Works

CRag uses a cache (in-memory or persistent) to store context chunks for queries. When a new query arrives, the system:

1. **Receives Query:** User submits a question.
2. **Checks Cache:** Looks for a matching or similar query in the cache.
3. **Retrieves Context:**
    - If found, uses cached context.
    - If not found, fetches context from the vector database.
4. **Generates Answer:** Uses the RAG pipeline to generate an answer from the context.
5. **Updates Cache:** Stores new context for future queries, improving speed and accuracy for repeated questions.

## Benefits
- **Faster response times:** Answers are generated quickly for repeated or similar queries.
- **Improved answer accuracy:** Cached context ensures relevant information is used.
- **Reduced database load:** Fewer queries to the vector database, lowering computational costs.

## Cache Configuration

CRag supports enabling and disabling the context caching functionality through environment variables. This gives you control over when to use caching vs. direct processing.

### Cache Enable/Disable

The cache can be controlled using the `CACHE_ENABLED` environment variable:

**Enable Cache (Default):**
```bash
CACHE_ENABLED=true    # or 1, yes, on (case insensitive)
```

**Disable Cache:**
```bash
CACHE_ENABLED=false   # or 0, no, off (case insensitive)
```

## Parallel Processing Configuration

CRag now includes advanced async and parallel processing capabilities for faster document ingestion, based on LlamaIndex's IngestionPipeline with parallel execution.

### Performance Modes

1. **Sequential Processing**: Traditional single-threaded processing
2. **Parallel Processing**: Multi-worker parallel processing using ProcessPoolExecutor
3. **Async Processing**: Asynchronous processing with parallel workers
4. **Async Batch Processing**: Process multiple documents concurrently

### Parallel Processing Settings

Control parallel processing behavior with these environment variables:

```bash
# Enable/disable parallel processing (default: true)
PARALLEL_PROCESSING_ENABLED=true

# Number of parallel workers per document (default: 4)
MAX_PARALLEL_WORKERS=4

# Maximum concurrent documents in batch processing (default: 2)
MAX_CONCURRENT_DOCUMENTS=2

# Enable/disable async processing (default: true)
ASYNC_PROCESSING_ENABLED=true
```

### Performance Benefits

Based on LlamaIndex benchmarks, the performance hierarchy is typically:
1. **Async + Parallel**: Fastest for large workloads
2. **Async Only**: Good for I/O bound operations
3. **Parallel Only**: Good for CPU bound operations  
4. **Sequential**: Slowest but most stable

### Environment Setup

Add these configurations to your `.env` file:

```env
# Cache Configuration
CACHE_ENABLED=true

# Parallel Processing Configuration
PARALLEL_PROCESSING_ENABLED=true
MAX_PARALLEL_WORKERS=4
MAX_CONCURRENT_DOCUMENTS=2
ASYNC_PROCESSING_ENABLED=true

# Other required variables
GEMINI_API_KEY=your_gemini_api_key_here
HACKRX_TOKEN=your_hackrx_token_here
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here

# Optional directories
DATA_DIR=data/
TEMP_DIR=temp/
CHROMA_PERSIST_DIR=chroma_db/
```

### Usage Examples

**Single Document Processing:**
```python
from utils.embeddings_chunking import process_document_for_rag, process_document_for_rag_async

# Using config defaults
success = process_document_for_rag(document_url)

# Explicit parallel processing
success = process_document_for_rag(document_url, use_parallel=True, num_workers=4)

# Async processing
success = await process_document_for_rag_async(document_url, num_workers=4)
```

**Batch Document Processing:**
```python
from utils.embeddings_chunking import process_documents_batch, process_multiple_documents_async

# Batch processing with config defaults
results = process_documents_batch(document_urls)

# Explicit async batch processing
results = process_documents_batch(document_urls, use_async=True, num_workers=4)

# Advanced async batch processing
results = await process_multiple_documents_async(document_urls, num_workers=4)
```

### Testing Performance

Test the different processing modes:

```powershell
# Test parallel processing performance
python test_parallel_processing.py

# Test cache toggle functionality  
python test_cache_toggle.py
```

### How Cache Toggle Works

**When Cache is ENABLED (`CACHE_ENABLED=true`):**
- Documents are processed and stored in Gemini's context cache
- Questions are answered using cached document context
- Faster response times for repeated queries
- Cache mappings are saved/loaded for persistence across sessions
- API endpoints show cache status and allow cache management

**When Cache is DISABLED (`CACHE_ENABLED=false`):**
- Documents are processed directly without caching
- Each question triggers a fresh API call to Gemini
- No cache files are created or loaded
- Cache management functions are bypassed
- Useful for testing, debugging, or avoiding cache-related issues

### API Endpoints for Cache Management

When cache is enabled, the following endpoints are available:

- `GET /hackrx/cache/status` - Get cache status and configuration
- `DELETE /hackrx/cache/{document_hash}` - Delete specific document cache
- `DELETE /hackrx/cache/all` - Clear all caches

---
*For more details, see the code and documentation in this repository.*

## How to Run

Follow these steps to run CRag locally:

1. **Clone the repository:**
   - Use the following command to clone the project:
     ```powershell
     git clone https://github.com/shreesha345/Rag_system.git
     cd Rag_system
     ```

2. **Install dependencies:**
   - Make sure you have Python 3.12+ installed.
   - Use [uv](https://github.com/astral-sh/uv) for fast dependency management:
     ```powershell
     uv sync
     ```
   - This will install all dependencies specified in `pyproject.toml` and `uv.lock`.

3. **Start the application:**
   - Run the main script:
     ```powershell
     uv venv
     uv pip run python main.py
     ```

4. **Add your workflow image:**
   - Place your workflow flow chart image in the project directory and link it in the [Image] section above.

---
*For troubleshooting or more details, refer to the code and comments in this repository.*
