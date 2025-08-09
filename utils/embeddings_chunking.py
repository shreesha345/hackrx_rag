import sys
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Optional
import nest_asyncio
nest_asyncio.apply()

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline
from utils.vector_db import create_new_index
from utils.document_phaser import parse_document_from_url, cleanup_temp_files, parse_document_optimized
from utils.config import (
    GEMINI_API_KEY,
    BUFFER_SIZE,
    BREAKPOINT_PERCENTILE_THRESHOLD,
    DATA_DIR,
    TEMP_DIR,
    PARALLEL_PROCESSING_ENABLED,
    MAX_PARALLEL_WORKERS,
    MAX_CONCURRENT_DOCUMENTS,
    ASYNC_PROCESSING_ENABLED
)
import os
import tempfile

GoogleEmbeddings = GoogleGenAIEmbedding(
    api_key=GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"),
    model="gemini-embedding-001",
    embed_content_config=EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
    )
)

# Configuration for parallel processing
MAX_WORKERS = MAX_PARALLEL_WORKERS  # Number of parallel workers
BATCH_SIZE = 10  # Batch size for processing documents

def create_ingestion_pipeline(num_workers: Optional[int] = None) -> IngestionPipeline:
    """
    Create an optimized ingestion pipeline with parallel processing capabilities.
    
    Args:
        num_workers: Number of parallel workers. If None, uses sequential processing.
                    If > 1, enables parallel processing.
    
    Returns:
        Configured IngestionPipeline
    """
    pipeline = IngestionPipeline(
        transformations=[
            SemanticSplitterNodeParser(
                buffer_size=BUFFER_SIZE,
                breakpoint_percentile_threshold=BREAKPOINT_PERCENTILE_THRESHOLD,
                embed_model=GoogleEmbeddings,
            ),
            GoogleEmbeddings,  # This will generate embeddings
        ],
        disable_cache=True  # Disable cache for better parallel performance
    )
    return pipeline

async def process_document_for_rag_async(document_url: str, num_workers: int = MAX_WORKERS) -> bool:
    """
    Async version of RAG pipeline with parallel processing: 
    Document parsing -> Chunking -> Embeddings -> Vector Store
    
    Args:
        document_url: URL or file path to process
        num_workers: Number of parallel workers for processing
        
    Returns:
        True if successful, False otherwise.
    """
    temp_markdown_path = None
    try:
        print(f"Starting ASYNC RAG processing for: {document_url}")
        print(f"Using {num_workers} parallel workers")
        
        # Step 1: Parse document to markdown
        print("Step 1: Parsing document...")
        if os.path.isfile(document_url):
            temp_markdown_path = parse_document_optimized(document_url, use_temp_output=True)
        else:
            temp_markdown_path = parse_document_from_url(document_url, cleanup_downloaded=True)
        
        # Step 2: Load document from DATA_DIR
        print("Step 2: Loading document from DATA_DIR...")
        data_md_path = os.path.join(DATA_DIR, os.path.basename(temp_markdown_path))
        reader = SimpleDirectoryReader(
            input_files=[data_md_path],
            required_exts=[".md"],
        )
        docs = reader.load_data()
        
        # Step 3: Create and run async ingestion pipeline
        print("Step 3: Running async ingestion pipeline with parallel processing...")
        pipeline = create_ingestion_pipeline(num_workers=num_workers)
        
        # Run the pipeline asynchronously with parallel processing
        nodes = await pipeline.arun(documents=docs, num_workers=num_workers)
        
        # Step 4: Connect to existing Chroma collection or create if not exists
        print("Step 4: Connecting to local Chroma store (accumulative mode)...")
        collection = create_new_index(index_name="gemini-rag-pipeline")
        
        # Step 5: Batch store in Chroma
        print("Step 5: Storing processed nodes in vector database...")
        await store_nodes_async(collection, nodes, document_url)
        
        print(f"Successfully processed {len(nodes)} chunks to local Chroma store (accumulative)")
        print("ASYNC RAG processing completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in async RAG processing: {e}")
        return False
    finally:
        await cleanup_async(temp_markdown_path)

def process_document_for_rag_parallel(document_url: str, num_workers: int = MAX_WORKERS) -> bool:
    """
    Parallel (sync) version of RAG pipeline: Document parsing -> Chunking -> Embeddings -> Vector Store
    
    Args:
        document_url: URL or file path to process
        num_workers: Number of parallel workers for processing
        
    Returns:
        True if successful, False otherwise.
    """
    temp_markdown_path = None
    try:
        print(f"Starting PARALLEL RAG processing for: {document_url}")
        print(f"Using {num_workers} parallel workers")
        
        # Step 1: Parse document to markdown
        print("Step 1: Parsing document...")
        if os.path.isfile(document_url):
            temp_markdown_path = parse_document_optimized(document_url, use_temp_output=True)
        else:
            temp_markdown_path = parse_document_from_url(document_url, cleanup_downloaded=True)
            
        # Step 2: Load document from DATA_DIR
        print("Step 2: Loading document from DATA_DIR...")
        data_md_path = os.path.join(DATA_DIR, os.path.basename(temp_markdown_path))
        reader = SimpleDirectoryReader(
            input_files=[data_md_path],
            required_exts=[".md"],
        )
        docs = reader.load_data()
        
        # Step 3: Create and run parallel ingestion pipeline
        print("Step 3: Running parallel ingestion pipeline...")
        pipeline = create_ingestion_pipeline(num_workers=num_workers)
        
        # Run the pipeline with parallel processing
        nodes = pipeline.run(documents=docs, num_workers=num_workers)
        
        # Step 4: Connect to existing Chroma collection or create if not exists
        print("Step 4: Connecting to local Chroma store (accumulative mode)...")
        collection = create_new_index(index_name="gemini-rag-pipeline")
        
        # Step 5: Store in Chroma
        print("Step 5: Storing processed nodes in vector database...")
        store_nodes_sync(collection, nodes, document_url)
        
        print(f"Successfully processed {len(nodes)} chunks to local Chroma store (accumulative)")
        print("PARALLEL RAG processing completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in parallel RAG processing: {e}")
        return False
    finally:
        import time
        time.sleep(1)  # Delay for file handle release
        if temp_markdown_path and os.path.exists(temp_markdown_path):
            try:
                os.remove(temp_markdown_path)
            except:
                pass
        cleanup_temp_files()

async def store_nodes_async(collection, nodes, document_url: str):
    """Async function to store nodes in vector database"""
    loop = asyncio.get_event_loop()
    
    def _store_nodes():
        # To avoid ID collisions, use a unique prefix for each document
        import hashlib
        doc_id_prefix = hashlib.md5(document_url.encode()).hexdigest()[:8]
        ids = [f"{doc_id_prefix}_{i}" for i in range(len(nodes))]
        
        # Extract texts and embeddings
        texts = [node.text for node in nodes]
        embeddings = [node.embedding for node in nodes if hasattr(node, 'embedding') and node.embedding]
        metadatas = [node.metadata for node in nodes]
        
        # If embeddings are not present, generate them
        if not embeddings or len(embeddings) != len(nodes):
            embeddings = GoogleEmbeddings.get_text_embedding_batch(texts)
        
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
    
    # Run the blocking operation in thread pool
    with ThreadPoolExecutor(max_workers=1) as executor:
        await loop.run_in_executor(executor, _store_nodes)

def store_nodes_sync(collection, nodes, document_url: str):
    """Sync function to store nodes in vector database"""
    # To avoid ID collisions, use a unique prefix for each document
    import hashlib
    doc_id_prefix = hashlib.md5(document_url.encode()).hexdigest()[:8]
    ids = [f"{doc_id_prefix}_{i}" for i in range(len(nodes))]
    
    # Extract texts and embeddings
    texts = [node.text for node in nodes]
    embeddings = [node.embedding for node in nodes if hasattr(node, 'embedding') and node.embedding]
    metadatas = [node.metadata for node in nodes]
    
    # If embeddings are not present, generate them
    if not embeddings or len(embeddings) != len(nodes):
        embeddings = GoogleEmbeddings.get_text_embedding_batch(texts)
    
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts
    )

async def cleanup_async(temp_markdown_path: Optional[str]):
    """Async cleanup function"""
    await asyncio.sleep(1)  # Async delay for file handle release
    if temp_markdown_path and os.path.exists(temp_markdown_path):
        try:
            os.remove(temp_markdown_path)
        except:
            pass
    cleanup_temp_files()

async def process_multiple_documents_async(document_urls: List[str], num_workers: int = MAX_WORKERS) -> dict:
    """
    Process multiple documents asynchronously with parallel processing.
    
    Args:
        document_urls: List of URLs or file paths to process
        num_workers: Number of parallel workers per document
        
    Returns:
        Dictionary with results for each document
    """
    results = {}
    
    print(f"Starting async batch processing of {len(document_urls)} documents")
    print(f"Using {num_workers} workers per document")
    
    # Create semaphore to limit concurrent document processing
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOCUMENTS)  # Process max documents concurrently
    
    async def process_single_document(url):
        async with semaphore:
            import time
            start_time = time.time()
            try:
                success = await process_document_for_rag_async(url, num_workers)
                processing_time = time.time() - start_time
                return url, {
                    'success': success,
                    'error': '' if success else 'Processing failed',
                    'processing_time': processing_time
                }
            except Exception as e:
                processing_time = time.time() - start_time
                print(f"Error processing {url}: {e}")
                return url, {
                    'success': False,
                    'error': str(e),
                    'processing_time': processing_time
                }
    
    # Process all documents concurrently
    tasks = [process_single_document(url) for url in document_urls]
    completed_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect results
    for result in completed_results:
        if isinstance(result, tuple):
            url, result_dict = result
            results[url] = result_dict
        else:
            print(f"Unexpected result: {result}")
    
    successful = sum(1 for result_dict in results.values() if result_dict.get('success', False))
    print(f"Batch processing completed: {successful}/{len(document_urls)} documents processed successfully")
    
    return results

def process_multiple_documents_parallel(document_urls: List[str], num_workers: int = MAX_WORKERS) -> dict:
    """
    Process multiple documents with parallel processing (sync version).
    
    Args:
        document_urls: List of URLs or file paths to process
        num_workers: Number of parallel workers per document
        
    Returns:
        Dictionary with results for each document
    """
    results = {}
    
    print(f"Starting parallel batch processing of {len(document_urls)} documents")
    print(f"Using {num_workers} workers per document")
    
    # Process documents in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOCUMENTS) as executor:  # Process max documents concurrently
        
        def process_with_timing(url):
            import time
            start_time = time.time()
            try:
                success = process_document_for_rag_parallel(url, num_workers)
                processing_time = time.time() - start_time
                return {
                    'success': success,
                    'error': '' if success else 'Processing failed',
                    'processing_time': processing_time
                }
            except Exception as e:
                processing_time = time.time() - start_time
                return {
                    'success': False,
                    'error': str(e),
                    'processing_time': processing_time
                }
        
        future_to_url = {
            executor.submit(process_with_timing, url): url 
            for url in document_urls
        }
        
        for future in future_to_url:
            url = future_to_url[future]
            try:
                result_dict = future.result()
                results[url] = result_dict
            except Exception as e:
                print(f"Error processing {url}: {e}")
                results[url] = {
                    'success': False,
                    'error': str(e),
                    'processing_time': 0.0
                }
    
    successful = sum(1 for result_dict in results.values() if result_dict.get('success', False))
    print(f"Batch processing completed: {successful}/{len(document_urls)} documents processed successfully")
    
    return results

# Wrapper functions for easy async usage
async def process_document_async(document_url: str, num_workers: int = MAX_WORKERS) -> bool:
    """Convenient wrapper for async document processing"""
    return await process_document_for_rag_async(document_url, num_workers)

def process_documents_batch(document_urls: List[str], use_async: bool = None, num_workers: int = None) -> dict:
    """
    Process multiple documents in batch with parallel processing.
    
    Args:
        document_urls: List of URLs or file paths to process
        use_async: Whether to use async processing (default: from config ASYNC_PROCESSING_ENABLED)
        num_workers: Number of parallel workers per document (default: from config MAX_PARALLEL_WORKERS)
        
    Returns:
        Dictionary with results for each document
    """
    # Use config defaults if not specified
    if use_async is None:
        use_async = ASYNC_PROCESSING_ENABLED
    if num_workers is None:
        num_workers = MAX_WORKERS
        
    if use_async:
        return asyncio.run(process_multiple_documents_async(document_urls, num_workers))
    else:
        return process_multiple_documents_parallel(document_urls, num_workers)

def process_document_for_rag(document_url: str, use_parallel: bool = None, num_workers: int = None) -> bool:
    """
    Complete RAG pipeline: Document parsing -> Chunking -> Embeddings -> Vector Store
    
    Args:
        document_url: URL or file path to process
        use_parallel: Whether to use parallel processing (default: from config PARALLEL_PROCESSING_ENABLED)
        num_workers: Number of parallel workers (default: from config MAX_PARALLEL_WORKERS)
        
    Returns:
        True if successful, False otherwise.
    """
    # Use config defaults if not specified
    if use_parallel is None:
        use_parallel = PARALLEL_PROCESSING_ENABLED
    if num_workers is None:
        num_workers = MAX_WORKERS
        
    if use_parallel and num_workers > 1:
        return process_document_for_rag_parallel(document_url, num_workers)
    else:
        return process_document_for_rag_sequential(document_url)

def process_document_for_rag_sequential(document_url: str) -> bool:
    """
    Original sequential RAG pipeline: Document parsing -> Chunking -> Embeddings -> Vector Store
    Returns True if successful, False otherwise.
    """
    temp_markdown_path = None
    try:
        print(f"Starting RAG processing for: {document_url}")
        # Step 1: Parse document to markdown
        print("Step 1: Parsing document...")
        if os.path.isfile(document_url):
            temp_markdown_path = parse_document_optimized(document_url, use_temp_output=True)
        else:
            temp_markdown_path = parse_document_from_url(document_url, cleanup_downloaded=True)
        # Step 2: Load and chunk document from DATA_DIR
        print("Step 2: Loading and chunking document from DATA_DIR...")
        data_md_path = os.path.join(DATA_DIR, os.path.basename(temp_markdown_path))
        reader = SimpleDirectoryReader(
            input_files=[data_md_path],
            required_exts=[".md"],
        )
        docs = reader.load_data()
        # Step 3: Semantic chunking
        print("Step 3: Performing semantic chunking...")
        splitter = SemanticSplitterNodeParser(
            buffer_size=BUFFER_SIZE,
            breakpoint_percentile_threshold=BREAKPOINT_PERCENTILE_THRESHOLD,
            embed_model=GoogleEmbeddings,
        )
        nodes = splitter.get_nodes_from_documents(docs)
        # Step 4: Connect to existing Chroma collection or create if not exists
        print("Step 4: Connecting to local Chroma store (accumulative mode)...")
        collection = create_new_index(index_name="gemini-rag-pipeline")
        # Step 5: Batch embed node texts and store in Chroma
        # To avoid ID collisions, use a unique prefix for each document (e.g., filename or URL hash)
        import hashlib
        doc_id_prefix = hashlib.md5(document_url.encode()).hexdigest()[:8]
        ids = [f"{doc_id_prefix}_{i}" for i in range(len(nodes))]
        texts = [node.text for node in nodes]
        embeddings = GoogleEmbeddings.get_text_embedding_batch(texts)
        metadatas = [node.metadata for node in nodes]
        documents = texts
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"Successfully processed {len(nodes)} chunks to local Chroma store (accumulative)")
        print("RAG processing completed successfully!")
        return True
    except Exception as e:
        print(f"Error in RAG processing: {e}")
        return False
    finally:
        import time
        time.sleep(1)  # Delay for file handle release
        if temp_markdown_path and os.path.exists(temp_markdown_path):
            try:
                os.remove(temp_markdown_path)
            except:
                pass
        cleanup_temp_files()

# Keep original function for backward compatibility
def process_directory_for_rag(directory_path: str) -> bool:
    """Process entire directory for RAG pipeline"""
    try:
        reader = SimpleDirectoryReader(
            input_dir=directory_path,
            required_exts=[".md"],
            recursive=True
        )
        docs = reader.load_data()
        splitter = SemanticSplitterNodeParser(
            buffer_size=BUFFER_SIZE,
            breakpoint_percentile_threshold=BREAKPOINT_PERCENTILE_THRESHOLD,
            embed_model=GoogleEmbeddings,
        )
        nodes = splitter.get_nodes_from_documents(docs)
        collection = create_new_index(index_name="gemini-rag-pipeline")
        # Use unique prefix for each directory batch to avoid ID collisions
        import hashlib
        dir_id_prefix = hashlib.md5(directory_path.encode()).hexdigest()[:8]
        ids = [f"{dir_id_prefix}_{i}" for i in range(len(nodes))]
        texts = [node.text for node in nodes]
        embeddings = GoogleEmbeddings.get_text_embedding_batch(texts)
        metadatas = [node.metadata for node in nodes]
        documents = texts
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"Successfully processed {len(nodes)} chunks from directory (accumulative)")
        return True
    except Exception as e:
        print(f"Error processing directory: {e}")
        return False

if __name__ == "__main__":
    # Test the RAG processing with different methods
    test_url = "https://hackrx.blob.core.windows.net/assets/Happy%20Family%20Floater%20-%202024%20OICHLIP25046V062425%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A24%3A30Z&se=2026-08-01T17%3A24%3A00Z&sr=b&sp=r&sig=VNMTTQUjdXGYb2F4Di4P0zNvmM2rTBoEHr%2BnkUXIqpQ%3D"
    
    print("=== RAG Processing Performance Test ===")
    print(f"Test URL: {test_url[:50]}...")
    
    # # Test sequential processing
    # print("\n1. Sequential Processing:")
    import time
    # start_time = time.time()
    # success_seq = process_document_for_rag(test_url, use_parallel=False)
    # seq_time = time.time() - start_time
    # print(f"Sequential processing: {success_seq} ✅ (Time: {seq_time:.2f}s)")
    
    # # Test parallel processing
    # print("\n2. Parallel Processing (4 workers):")
    # start_time = time.time()
    # success_par = process_document_for_rag(test_url, use_parallel=True, num_workers=4)
    # par_time = time.time() - start_time
    # print(f"Parallel processing: {success_par} ✅ (Time: {par_time:.2f}s)")
    
    # Test async + parallel processing
    print("\n3. Async + Parallel Processing (4 workers):")
    start_time = time.time()
    success_async = asyncio.run(process_document_for_rag_async(test_url, num_workers=4))
    async_time = time.time() - start_time
    print(f"Async + Parallel processing: {success_async} ✅ (Time: {async_time:.2f}s)")
    
    # # Performance summary
    # print(f"\n=== Performance Summary ===")
    # print(f"Sequential:        {seq_time:.2f}s")
    # print(f"Parallel (Sync):   {par_time:.2f}s (Speedup: {seq_time/par_time:.2f}x)")
    # print(f"Async + Parallel:  {async_time:.2f}s (Speedup: {seq_time/async_time:.2f}x) ⭐ FASTEST")
    
    # # # Test batch processing
    # print(f"\n4. Async Batch + Parallel Processing Example:")
    # test_urls = [test_url]  # Add more URLs for real batch testing
    # start_time = time.time()
    # batch_results = process_documents_batch(test_urls, use_async=True, num_workers=4)
    # batch_time = time.time() - start_time
    # print(f"Batch processing results: {batch_results}")
    # print(f"Batch processing time: {batch_time:.2f}s ⭐⭐ BEST FOR MULTIPLE DOCS")
    