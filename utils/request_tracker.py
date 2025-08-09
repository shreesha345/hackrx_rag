"""
Request tracking utilities for logging API requests
"""
import os
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import uuid

logger = logging.getLogger(__name__)

class RequestTracker:
    def __init__(self, log_directory: str = "logs", max_workers: int = 2):
        """
        Initialize request tracker
        
        Args:
            log_directory: Directory to store request logs
            max_workers: Number of background workers for processing requests
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Thread-safe queue for request data
        self.request_queue = Queue()
        
        # Background thread pool for processing requests
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="RequestTracker")
        
        # Lock for file operations
        self.file_lock = threading.Lock()
        
        # Start background processing
        self._start_background_processing()
        
        # Current log file path
        self.current_log_file = self._get_log_file_path()
        
        logger.info(f"Request tracker initialized. Logs will be stored in: {self.current_log_file}")
    
    def _get_log_file_path(self) -> Path:
        """Get current log file path based on date"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_directory / f"requests_{today}.json"
    
    def _start_background_processing(self):
        """Start background thread to process request queue"""
        def process_queue():
            while True:
                try:
                    # Get request data from queue (blocks until available)
                    request_data = self.request_queue.get()
                    if request_data is None:  # Shutdown signal
                        break
                    
                    # Process the request data
                    self._write_request_to_file(request_data)
                    self.request_queue.task_done()
                    
                except Exception as e:
                    logger.error(f"Error processing request queue: {e}")
        
        # Start background thread
        self.executor.submit(process_queue)
    
    def track_request(self, request_data: Dict[str, Any], response_data: Dict[str, Any] = None, user_info: Dict[str, Any] = None):
        """
        Track a request (non-blocking)
        
        Args:
            request_data: Request information
            response_data: Response information (optional)
            user_info: User information (optional)
        """
        try:
            # Create request record
            record = {
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "unix_timestamp": time.time(),
                "endpoint": "/api/v1/hackrx/run",
                "method": "POST",
                "request": request_data,
                "response": response_data or {},
                "user": user_info or {},
                "processing_info": {
                    "tracked_at": datetime.now().isoformat()
                }
            }
            
            # Add to queue (non-blocking)
            self.request_queue.put_nowait(record)
            
        except Exception as e:
            logger.error(f"Error tracking request: {e}")
    
    def _write_request_to_file(self, record: Dict[str, Any]):
        """Write request record to JSON file (thread-safe)"""
        try:
            current_file = self._get_log_file_path()
            
            with self.file_lock:
                # Read existing data
                existing_data = []
                if current_file.exists():
                    try:
                        with open(current_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    except (json.JSONDecodeError, FileNotFoundError):
                        existing_data = []
                
                # Ensure existing_data is a list
                if not isinstance(existing_data, list):
                    existing_data = []
                
                # Add new record
                existing_data.append(record)
                
                # Write back to file
                with open(current_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
                logger.debug(f"Request logged to {current_file}")
                
        except Exception as e:
            logger.error(f"Error writing request to file: {e}")
    
    def get_requests_by_date(self, date: str = None) -> List[Dict[str, Any]]:
        """
        Get requests for a specific date
        
        Args:
            date: Date in YYYY-MM-DD format (defaults to today)
        
        Returns:
            List of request records
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        log_file = self.log_directory / f"requests_{date}.json"
        
        if not log_file.exists():
            return []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error reading requests for date {date}: {e}")
            return []
    
    def get_request_stats(self, date: str = None) -> Dict[str, Any]:
        """
        Get request statistics for a date
        
        Args:
            date: Date in YYYY-MM-DD format (defaults to today)
        
        Returns:
            Dictionary with request statistics
        """
        requests = self.get_requests_by_date(date)
        
        if not requests:
            return {
                "date": date or datetime.now().strftime("%Y-%m-%d"),
                "total_requests": 0,
                "avg_processing_time": 0,
                "success_count": 0,
                "error_count": 0,
                "unique_documents": 0,
                "total_questions": 0
            }
        
        # Calculate statistics
        total_requests = len(requests)
        processing_times = []
        success_count = 0
        error_count = 0
        unique_documents = set()
        total_questions = 0
        
        for req in requests:
            # Processing time
            if req.get("response", {}).get("processing_time"):
                processing_times.append(req["response"]["processing_time"])
            
            # Success/Error count
            if req.get("response", {}).get("answers"):
                answers = req["response"]["answers"]
                if any("Error:" in str(ans) for ans in answers):
                    error_count += 1
                else:
                    success_count += 1
            
            # Unique documents
            if req.get("request", {}).get("documents"):
                unique_documents.add(req["request"]["documents"])
            
            # Total questions
            if req.get("request", {}).get("questions"):
                total_questions += len(req["request"]["questions"])
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "total_requests": total_requests,
            "avg_processing_time": round(avg_processing_time, 2),
            "success_count": success_count,
            "error_count": error_count,
            "unique_documents": len(unique_documents),
            "total_questions": total_questions,
            "success_rate": round((success_count / total_requests) * 100, 2) if total_requests > 0 else 0
        }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Signal shutdown to background thread
            self.request_queue.put(None)
            
            # Shutdown executor (timeout parameter was removed in newer Python versions)
            self.executor.shutdown(wait=True)
            
            logger.info("Request tracker cleanup completed")
        except Exception as e:
            logger.error(f"Error during request tracker cleanup: {e}")
            
    def get_last_requests(self, count=10, with_answers=True):
        """
        Get the last N requests with their answers for debugging
        
        Args:
            count: Number of requests to retrieve
            with_answers: Whether to include answers in the result
        
        Returns:
            List of the last N request records
        """
        try:
            current_file = self._get_log_file_path()
            
            if not current_file.exists():
                return []
                
            with self.file_lock:
                try:
                    with open(current_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if not isinstance(data, list):
                        return []
                        
                    # Sort by timestamp (newest first)
                    data.sort(key=lambda x: x.get('unix_timestamp', 0), reverse=True)
                    
                    # Take the last N requests
                    results = data[:count]
                    
                    # Filter out answers if not requested
                    if not with_answers:
                        for record in results:
                            if 'response' in record and 'answers' in record['response']:
                                del record['response']['answers']
                    
                    return results
                    
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    logger.error(f"Error reading request log file: {e}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error retrieving last requests: {e}")
            return []

# Global instance
_request_tracker = None

def get_request_tracker() -> RequestTracker:
    """Get global request tracker instance"""
    global _request_tracker
    if _request_tracker is None:
        _request_tracker = RequestTracker()
    return _request_tracker

def cleanup_request_tracker():
    """Cleanup global request tracker"""
    global _request_tracker
    if _request_tracker:
        _request_tracker.cleanup()
        _request_tracker = None