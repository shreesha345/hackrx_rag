"""
FastAPI application entry point
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager
import os
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from api.routes import router
from api.models import ErrorResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting HackRX Document Processing API")
    # Load cache mappings
    from utils.cache import load_cache_mapping
    load_cache_mapping()
    # Log URL blocker status
    from utils.url_blocker import BLOCKED_URLS
    logger.info(f"URL blocker activated with {len(BLOCKED_URLS)} blocked URLs")
    yield
    # Shutdown
    logger.info("Shutting down HackRX Document Processing API")
    # Save cache mappings
    from utils.cache import save_cache_mapping
    save_cache_mapping()
    # Cleanup resources
    from api.routes import _executor, cleanup_all_caches
    from utils.request_tracker import cleanup_request_tracker
    try:
        cleanup_all_caches()
        cleanup_request_tracker()
        _executor.shutdown(wait=True)
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Create FastAPI application
app = FastAPI(
    title="HackRX Document Processing API",
    description="""
    AI-powered document analysis with smart caching and RAG capabilities.
    
    Features:
    - Document processing with intelligent caching
    - RAG (Retrieval-Augmented Generation) for better answers
    - Request tracking and analytics
    - Real-time processing status
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Document Processing"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "HackRX Document Processing API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/hackrx/health",
        "endpoints": {
            "process_documents": "/api/v1/hackrx/run",
            "process_embeddings": "/api/v1/hackrx/embeddings", 
            "get_status": "/api/v1/hackrx/status/{document_id}",
            "get_requests": "/api/v1/hackrx/requests",
            "get_request_stats": "/api/v1/hackrx/requests/stats",
            "cleanup": "/api/v1/hackrx/cleanup"
        }
    }

if __name__ == "__main__":
    # Run the application with optimized settings for parallel processing
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        # reload=True,  # Remove in production
        log_level="info",
        workers=1,  # Keep at 1 for Windows, use threading for parallelism
        loop="asyncio",  # Use asyncio event loop for better concurrency
        limit_concurrency=100,  # Allow up to 100 concurrent connections
        limit_max_requests=1000  # Handle up to 1000 requests per worker
    )