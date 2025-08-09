from dotenv import load_dotenv
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


load_dotenv()

TOKEN = os.getenv("HACKRX_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
DATA_DIR = os.getenv("DATA_DIR", default="data/")  # RAG pipeline files
TEMP_DIR = os.getenv("TEMP_DIR", default="temp/")  # Cache temp files

# Pinecone Configuration

# Chroma Configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", default="chroma_db/")
CHROMA_COLLECTION_NAME = "gemini-rag-pipeline"

# Gemini Configuration
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_LLM_MODEL = "gemini-2.5-flash"
TEMPERATURE = 1
# Semantic Chunking Configuration (NO sentence splitters used)
BUFFER_SIZE = 1  # Semantic splitter buffer size
BREAKPOINT_PERCENTILE_THRESHOLD = 95  # Semantic similarity threshold for chunk boundaries

# Retrieval Configuration
SIMILARITY_TOP_K = 5

# Cache Configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "false").lower() in ("true", "1", "yes", "on")

# Parallel Processing Configuration
PARALLEL_PROCESSING_ENABLED = os.getenv("PARALLEL_PROCESSING_ENABLED", "true").lower() in ("true", "1", "yes", "on")
MAX_PARALLEL_WORKERS = int(os.getenv("MAX_PARALLEL_WORKERS", "16"))
MAX_CONCURRENT_DOCUMENTS = int(os.getenv("MAX_CONCURRENT_DOCUMENTS", "2"))
ASYNC_PROCESSING_ENABLED = os.getenv("ASYNC_PROCESSING_ENABLED", "true").lower() in ("true", "1", "yes", "on")

# Synchronous Processing Timeout (when cache is disabled)
SYNC_PROCESSING_TIMEOUT = int(os.getenv("SYNC_PROCESSING_TIMEOUT", "300"))  # 5 minutes default

# Performance Configuration
ENABLE_FAST_RESPONSE = os.getenv("ENABLE_FAST_RESPONSE", "true").lower() in ("true", "1", "yes", "on")
QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "60"))  # Query timeout in seconds
CACHE_TIMEOUT = int(os.getenv("CACHE_TIMEOUT", "45"))  # Cache timeout in seconds
PROCESSING_QUEUE_TIMEOUT = int(os.getenv("PROCESSING_QUEUE_TIMEOUT", "300"))  # Queue timeout in seconds

# URL Blocker Configuration
URL_BLOCKER_ENABLED = os.getenv("URL_BLOCKER_ENABLED", "true").lower() in ("true", "1", "yes", "on")
