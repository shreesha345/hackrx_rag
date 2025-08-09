import sys
from utils.config import (
    TEMPERATURE,
    CACHE_ENABLED
)
import os
from dotenv import load_dotenv
import io
import json
import gc
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types
import PyPDF2
import fitz  # PyMuPDF for better PDF analysis
import tempfile
import uuid
import time
from pydantic import BaseModel, field_validator
from typing import List
# Add Pydantic models for structured output

# Unified answer format - always returns a list
class UnifiedAnswers(BaseModel):
    """Unified answer format - always returns a list"""
    answers: List[str]

    @field_validator('answers')
    def validate_answers(cls, v):
        if not v:
            raise ValueError('Must have at least one answer')
        validated_answers = []
        for answer in v:
            if not answer or str(answer).strip() == "":
                validated_answers.append("Error: Empty answer provided")
            else:
                validated_answers.append(str(answer).strip())
        return validated_answers

# Load environment variables from .env file
load_dotenv()

# Extend sys path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import GEMINI_API_KEY, DATA_DIR, TEMP_DIR
from utils.prompt import SYSTEM_PROMPT

from utils.document_phaser import (
    download_file_streaming,
    parse_document_optimized,
    parse_document_from_url,
    get_file_type_from_path,
    get_document_info,
    cleanup_temp_files,
    is_supported_file_type,
    get_file_processing_strategy
)
from utils.file_type_detector import (
    FileTypeDetector,
    should_use_rag_processing,
    is_presentation_or_spreadsheet
)
from utils.email_processor import is_email_file

try:
    from spire.doc import Document as SpireDocument, FileFormat
    SPIRE_AVAILABLE = True
except ImportError:
    SPIRE_AVAILABLE = False

client = genai.Client(api_key=GEMINI_API_KEY)
model_name = "gemini-2.5-flash"
fallback_model_name = "gemini-2.0-flash"
system_instruction = SYSTEM_PROMPT

executor = ThreadPoolExecutor(max_workers=4)
_active_caches = {}

def is_cache_enabled() -> bool:
    """Check if cache is enabled based on configuration"""
    return CACHE_ENABLED

def get_cache_status() -> dict:
    """Get current cache status and configuration"""
    return {
        "enabled": CACHE_ENABLED,
        "active_caches": len(_active_caches),
        "document_cache_mappings": len(_document_cache_map) if '_document_cache_map' in globals() else 0
    }

def create_unique_filename(base_name: str, extension: str = "") -> str:
    """Create a unique filename to avoid conflicts"""
    unique_id = str(uuid.uuid4())[:8]
    timestamp = str(int(time.time()))
    return f"{base_name}_{unique_id}_{timestamp}{extension}"

def analyze_pdf_content(pdf_path_or_stream):
    """
    Analyze PDF to determine if it's text-based or image-heavy.
    Returns dict with analysis results.
    """
    try:
        # Handle both file paths and BytesIO streams
        if isinstance(pdf_path_or_stream, (str, os.PathLike)):
            doc = fitz.open(pdf_path_or_stream)
        else:
            # For BytesIO streams
            pdf_path_or_stream.seek(0)
            doc = fitz.open(stream=pdf_path_or_stream.read(), filetype="pdf")
            pdf_path_or_stream.seek(0)  # Reset stream position
        
        total_pages = len(doc)
        text_pages = 0
        image_pages = 0
        total_text_length = 0
        total_images = 0
        
        for page_num in range(min(total_pages, 10)):  # Analyze first 10 pages for speed
            page = doc[page_num]
            
            # Extract text
            text = page.get_text().strip()
            text_length = len(text)
            total_text_length += text_length
            
            # Count images
            image_list = page.get_images()
            page_image_count = len(image_list)
            total_images += page_image_count
            
            # Classify page
            if text_length > 100:  # Threshold for meaningful text
                text_pages += 1
            if page_image_count > 0:
                image_pages += 1
        
        doc.close()
        
        # Calculate ratios
        analyzed_pages = min(total_pages, 10)
        text_ratio = text_pages / analyzed_pages if analyzed_pages > 0 else 0
        image_ratio = image_pages / analyzed_pages if analyzed_pages > 0 else 0
        avg_text_per_page = total_text_length / analyzed_pages if analyzed_pages > 0 else 0
        avg_images_per_page = total_images / analyzed_pages if analyzed_pages > 0 else 0
        
        # Determine if PDF is image-heavy
        is_image_heavy = (
            text_ratio < 0.3 or  # Less than 30% pages have meaningful text
            avg_text_per_page < 50 or  # Very little text per page
            (image_ratio > 0.7 and avg_text_per_page < 200)  # Lots of images with little text
        )
        
        return {
            'total_pages': total_pages,
            'analyzed_pages': analyzed_pages,
            'text_pages': text_pages,
            'image_pages': image_pages,
            'text_ratio': text_ratio,
            'image_ratio': image_ratio,
            'avg_text_per_page': avg_text_per_page,
            'avg_images_per_page': avg_images_per_page,
            'total_text_length': total_text_length,
            'total_images': total_images,
            'is_image_heavy': is_image_heavy,
            'recommended_processing': 'llamaparse' if is_image_heavy else 'direct_upload'
        }
        
    except Exception as e:
        print(f"Error analyzing PDF content: {e}")
        # Default to safe processing with LlamaParse
        return {
            'total_pages': 0,
            'analyzed_pages': 0,
            'text_pages': 0,
            'image_pages': 0,
            'text_ratio': 0,
            'image_ratio': 1,
            'avg_text_per_page': 0,
            'avg_images_per_page': 1,
            'total_text_length': 0,
            'total_images': 1,
            'is_image_heavy': True,
            'recommended_processing': 'llamaparse',
            'error': str(e)
        }

import hashlib
import pickle
from pathlib import Path
import threading

# --- Document URL tracking and cache management ---
_document_cache_map = {}  # Maps document URLs (hash) to cache names
_cache_creation_lock = threading.Lock()  # Ensure thread-safe cache creation

def get_document_hash(document_url: str) -> str:
    """Generate a consistent hash for document URL"""
    normalized_url = document_url.strip().lower()
    return hashlib.md5(normalized_url.encode('utf-8')).hexdigest()

def save_cache_mapping():
    """Save cache mapping to disk for persistence"""
    if not is_cache_enabled():
        print("Cache is disabled - skipping cache mapping save")
        return
        
    cache_file = Path("cache_mapping.pkl")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(_document_cache_map, f)
    except Exception as e:
        print(f"Error saving cache mapping: {e}")

def load_cache_mapping():
    """Load cache mapping from disk"""
    global _document_cache_map
    
    if not is_cache_enabled():
        print("Cache is disabled - skipping cache mapping load")
        _document_cache_map = {}
        return
        
    cache_file = Path("cache_mapping.pkl")
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                _document_cache_map = pickle.load(f)
            print(f"Loaded {len(_document_cache_map)} cached documents")
        except Exception as e:
            print(f"Error loading cache mapping: {e}")
            _document_cache_map = {}
    else:
        _document_cache_map = {}

def get_or_create_cache_with_tracking(document_url: str) -> str:
    """
    Get existing cache or create new one for document with proper tracking.
    Returns cache name and ensures no duplicate caches for same document.
    Returns None if file should be processed with RAG instead of caching.
    """
    if not is_cache_enabled():
        print("Cache is disabled - proceeding without caching")
        return None
    
    # Check if this file type should be forced to RAG processing
    if should_force_rag_processing(document_url, cache_enabled=True):
        file_category = FileTypeDetector.get_file_category(document_url)
        print(f"File type '{file_category}' detected - forcing RAG processing instead of caching")
        return None
        
    doc_hash = get_document_hash(document_url)
    with _cache_creation_lock:
        # Check if we already have a cache for this document
        if doc_hash in _document_cache_map:
            cache_name = _document_cache_map[doc_hash]
            print(f"Reusing existing cache for document: {document_url[:50]}...")
            print(f"Cache name: {cache_name}")
            # Verify cache still exists and is accessible
            try:
                # Test if cache is accessible with a simple query
                client.models.generate_content(
                    model=model_name,
                    contents="test",
                    config=types.GenerateContentConfig(
                        cached_content=cache_name,
                        temperature=TEMPERATURE
                    )
                )
                return cache_name
            except Exception as e:
                print(f"Cached content no longer accessible: {e}")
                # Remove invalid cache from mapping
                del _document_cache_map[doc_hash]
                save_cache_mapping()
        # Create new cache
        print(f"Creating new cache for document: {document_url[:50]}...")
        cache_name = create_cache_optimized(document_url)
        _document_cache_map[doc_hash] = cache_name
        save_cache_mapping()
        print(f"New cache created and stored: {cache_name}")
        return cache_name

def process_questions_without_cache(document_url: str, questions: list) -> dict:
    """
    Process questions without using cache - use RAG system for presentations/spreadsheets,
    direct API calls for others.
    """
    if isinstance(questions, str):
        questions = [questions]
    
    # Check if this is a presentation or spreadsheet that should use RAG
    file_category = FileTypeDetector.get_file_category(document_url)
    if file_category in ['presentation', 'spreadsheet']:
        print(f"Processing {file_category} file through RAG pipeline...")
        return process_questions_with_rag(document_url, questions)
    
    all_answers = []
    
    try:
        print(f"Processing {len(questions)} questions without cache for document: {document_url[:50]}...")
        print("Cache is disabled - using direct Gemini API calls")
        
        # Process each question individually without cache
        for i, question in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)}: {question[:50]}...")
            
            try:
                # Create a direct prompt that includes document context
                prompt = f"""Based on the document at: {document_url}

Please answer the following question: {question}

If you need to analyze the document content, please do so and provide a comprehensive answer."""

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=TEMPERATURE,
                        system_instruction=system_instruction
                    )
                )
                
                answer = response.text.strip() if response.text else "No answer provided"
                all_answers.append(answer)
                print(f"Question {i} completed successfully")
                
            except Exception as e:
                error_msg = f"Error processing question {i}: {str(e)}"
                print(error_msg)
                all_answers.append(error_msg)
        
        return {"answers": all_answers}
        
    except Exception as e:
        error_msg = f"Error in non-cached processing: {str(e)}"
        print(error_msg)
        return {"answers": [error_msg] * len(questions)}

def process_questions_with_rag(document_url: str, questions: list) -> dict:
    """
    Process questions using the RAG system after indexing the document.
    Used for presentations, spreadsheets, and when cache is disabled.
    """
    if isinstance(questions, str):
        questions = [questions]
    
    try:
        print(f"Processing document for RAG indexing: {document_url[:50]}...")
        
        # Download and parse the document
        downloaded_file_path = None
        markdown_path = None
        
        try:
            if os.path.isfile(document_url):
                file_path = document_url
            else:
                # Download the file
                extension = FileTypeDetector.get_file_extension(document_url)
                downloaded_file_path = download_file_to_unique_path(document_url, extension or '.pdf')
                file_path = downloaded_file_path
            
            # Parse the document to markdown
            markdown_path = parse_document_optimized(file_path, use_temp_output=True)
            
            print(f"Document parsed to markdown: {markdown_path}")
            
            # Import and process through RAG
            from utils.embeddings_chunking import process_directory
            from utils.query import ask_question
            
            # Process the markdown file for embeddings
            temp_dir = os.path.dirname(markdown_path)
            process_directory(temp_dir)
            
            # Now use the RAG system to answer questions
            print(f"Querying RAG system with {len(questions)} questions...")
            result = ask_question(questions)
            
            # Parse the result
            if isinstance(result, str):
                try:
                    parsed_result = json.loads(result)
                    if "answers" in parsed_result:
                        return {"answers": parsed_result["answers"]}
                    elif "answer" in parsed_result:
                        return {"answers": [parsed_result["answer"]]}
                    else:
                        return {"answers": [result]}
                except json.JSONDecodeError:
                    return {"answers": [result]}
            else:
                return {"answers": [str(result)]}
                
        finally:
            # Clean up temporary files
            if downloaded_file_path and os.path.exists(downloaded_file_path):
                try:
                    os.remove(downloaded_file_path)
                    print(f"Cleaned up downloaded file: {downloaded_file_path}")
                except:
                    pass
            
            if markdown_path and os.path.exists(markdown_path):
                try:
                    os.remove(markdown_path)
                    print(f"Cleaned up markdown file: {markdown_path}")
                except:
                    pass
                    
    except Exception as e:
        error_msg = f"Error in RAG processing: {str(e)}"
        print(error_msg)
        return {"answers": [error_msg] * len(questions)}

def get_or_create_cache(document_url: str) -> str:
    """
    Get existing cache or create new one for document. Returns cache name.
    This now uses the tracking system to avoid duplicate caches.
    """
    return get_or_create_cache_with_tracking(document_url)

def is_pdf_url(url):
    try:
        parsed_url = urlparse(url)
        if parsed_url.path.lower().endswith('.pdf'):
            return True
        response = requests.head(url, timeout=10, allow_redirects=True)
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' in content_type:
            return True
        content_disposition = response.headers.get('content-disposition', '').lower()
        if 'pdf' in content_disposition:
            return True
        return False
    except Exception:
        return url.lower().endswith('.pdf') or '.pdf' in url.lower()

def stream_pdf_from_url(url):
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' not in content_type:
            raise ValueError(f"URL does not serve PDF content. Content-Type: {content_type}")
        pdf_data = io.BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                pdf_data.write(chunk)
        pdf_data.seek(0)
        return pdf_data

def download_file_to_unique_path(url, extension='.pdf'):
    """Download file to a unique path to avoid conflicts"""
    unique_filename = create_unique_filename("cache_download", extension)
    temp_path = os.path.join(TEMP_DIR, unique_filename)
    
    # Ensure directory exists
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    try:
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return temp_path
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise e

def process_pdf_intelligently(pdf_path_or_stream, is_url_source=False, original_url=None):
    """
    Intelligently process PDF based on its content type.
    Returns a file object ready for Gemini upload and the mime type.
    """
    temp_files_to_cleanup = []
    
    try:
        # Analyze PDF content
        print("Analyzing PDF content...")
        analysis = analyze_pdf_content(pdf_path_or_stream)
        
        print(f"PDF Analysis Results:")
        print(f"  - Total pages: {analysis['total_pages']}")
        print(f"  - Text ratio: {analysis['text_ratio']:.2f}")
        print(f"  - Image ratio: {analysis['image_ratio']:.2f}")
        print(f"  - Avg text per page: {analysis['avg_text_per_page']:.0f}")
        print(f"  - Is image-heavy: {analysis['is_image_heavy']}")
        print(f"  - Recommended processing: {analysis['recommended_processing']}")
        
        if analysis['is_image_heavy']:
            print("PDF appears to be image-heavy. Using LlamaParse for text extraction...")
            
            # Save PDF to unique temporary file for LlamaParse
            if isinstance(pdf_path_or_stream, (str, os.PathLike)):
                temp_pdf_path = pdf_path_or_stream
                cleanup_temp_pdf = False
            else:
                # Save BytesIO to unique temporary file
                unique_pdf_name = create_unique_filename("temp_pdf", ".pdf")
                temp_pdf_path = os.path.join(TEMP_DIR, unique_pdf_name)
                temp_files_to_cleanup.append(temp_pdf_path)
                
                os.makedirs(TEMP_DIR, exist_ok=True)
                with open(temp_pdf_path, 'wb') as temp_pdf:
                    pdf_path_or_stream.seek(0)
                    temp_pdf.write(pdf_path_or_stream.read())
                cleanup_temp_pdf = True
            
            try:
                # Use LlamaParse to extract text
                if is_url_source and original_url:
                    # Download to unique file for LlamaParse
                    unique_download_path = download_file_to_unique_path(original_url, '.pdf')
                    temp_files_to_cleanup.append(unique_download_path)
                    markdown_path = parse_document_optimized(unique_download_path, use_temp_output=True)
                else:
                    markdown_path = parse_document_optimized(temp_pdf_path, use_temp_output=True)
                
                # Read the markdown content
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                # Create BytesIO with markdown content
                text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                mime_type = 'text/markdown'
                
                print(f"Successfully extracted text using LlamaParse. Content length: {len(markdown_content)} chars")
                return text_bytes, mime_type, temp_files_to_cleanup
                
            except Exception as e:
                print(f"LlamaParse processing failed: {e}")
                # Cleanup temporary PDF if created
                if cleanup_temp_pdf and os.path.exists(temp_pdf_path):
                    try:
                        os.remove(temp_pdf_path)
                    except:
                        pass
                raise e
        else:
            print("PDF appears to be text-based. Using direct upload...")
            
            if isinstance(pdf_path_or_stream, (str, os.PathLike)):
                # Return file handle for direct upload
                return open(pdf_path_or_stream, 'rb'), 'application/pdf', temp_files_to_cleanup
            else:
                # Return BytesIO stream
                pdf_path_or_stream.seek(0)
                return pdf_path_or_stream, 'application/pdf', temp_files_to_cleanup
                
    except Exception as e:
        print(f"Error in intelligent PDF processing: {e}")
        print("Falling back to LlamaParse processing...")
        
        # Fallback to LlamaParse
        try:
            if isinstance(pdf_path_or_stream, (str, os.PathLike)):
                temp_pdf_path = pdf_path_or_stream
            else:
                # Save BytesIO to unique temporary file
                unique_pdf_name = create_unique_filename("fallback_pdf", ".pdf")
                temp_pdf_path = os.path.join(TEMP_DIR, unique_pdf_name)
                temp_files_to_cleanup.append(temp_pdf_path)
                
                os.makedirs(TEMP_DIR, exist_ok=True)
                with open(temp_pdf_path, 'wb') as temp_pdf:
                    pdf_path_or_stream.seek(0)
                    temp_pdf.write(pdf_path_or_stream.read())
            
            if is_url_source and original_url:
                # Download to unique file for LlamaParse
                unique_download_path = download_file_to_unique_path(original_url, '.pdf')
                temp_files_to_cleanup.append(unique_download_path)
                markdown_path = parse_document_optimized(unique_download_path, use_temp_output=True)
            else:
                markdown_path = parse_document_optimized(temp_pdf_path, use_temp_output=True)
            
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
            return text_bytes, 'text/markdown', temp_files_to_cleanup
            
        except Exception as fallback_error:
            print(f"Fallback processing also failed: {fallback_error}")
            raise

def convert_word_to_markdown_fast(file_path: str) -> str:
    if not SPIRE_AVAILABLE:
        return parse_document_optimized(file_path, use_temp_output=False)
    try:
        unique_md_name = create_unique_filename("temp_markdown", ".md")
        output_path = os.path.join(TEMP_DIR, unique_md_name)
        document = SpireDocument()
        document.LoadFromFile(file_path)
        document.SaveToFile(output_path, FileFormat.Markdown)
        document.Dispose()
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if os.path.exists(output_path):
            os.remove(output_path)
        return content
    except Exception:
        return parse_document_optimized(file_path, use_temp_output=False)

def is_word_document(file_path: str) -> bool:
    """Check if file is a Word document."""
    ext = get_file_type_from_path(file_path).lower()
    return ext in ['.doc', '.docx', '.docm', '.dot', '.dotm']

def is_presentation_document(file_path: str) -> bool:
    """Check if file is a presentation document."""
    ext = get_file_type_from_path(file_path).lower()
    return ext in ['.ppt', '.pptx', '.pptm', '.pot', '.potx', '.potm', '.key']

def is_spreadsheet_document(file_path: str) -> bool:
    """Check if file is a spreadsheet document."""
    ext = get_file_type_from_path(file_path).lower()
    return ext in ['.xlsx', '.xls', '.xlsm', '.xlsb', '.xlw', '.csv', '.tsv', 
                   '.dif', '.sylk', '.slk', '.prn', '.numbers', '.et', '.ods', 
                   '.fods', '.uos1', '.uos2', '.dbf', '.wk1', '.wk2', '.wk3', 
                   '.wk4', '.wks', '.123', '.wq1', '.wq2', '.wb1', '.wb2', 
                   '.wb3', '.qpw', '.xlr', '.eth']

def should_force_rag_processing(file_path: str, cache_enabled: bool = True) -> bool:
    """
    Determine if file should be forced to RAG processing instead of caching.
    Presentations and spreadsheets are always RAG'd.
    """
    if not cache_enabled:
        return True
    
    return should_use_rag_processing(file_path, cache_enabled)

def create_cache_optimized(document_url: str) -> str:
    downloaded_file_path = None
    markdown_path = None
    document = None
    file_handle = None
    temp_files_to_cleanup = []
    
    try:
        if os.path.isdir(document_url):
            import queue
            from threading import Thread
            files = [os.path.join(document_url, f) for f in os.listdir(document_url)
                     if os.path.isfile(os.path.join(document_url, f))]
            file_queue = queue.Queue()
            results = [None] * len(files)

            # Enqueue all files with their original index
            for idx, file_path in enumerate(files):
                file_queue.put((idx, file_path))

            def worker():
                while True:
                    try:
                        idx, file_path = file_queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        cache_name = get_or_create_cache(file_path)
                        results[idx] = (os.path.basename(file_path), cache_name)
                    except Exception:
                        results[idx] = (os.path.basename(file_path), None)
                    finally:
                        file_queue.task_done()

            num_workers = min(4, len(files))
            threads = [Thread(target=worker) for _ in range(num_workers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Build ordered cache_map
            cache_map = {}
            for res in results:
                if res and res[0] and res[1]:
                    cache_map[res[0]] = res[1]
            return cache_map

        if os.path.isfile(document_url):
            file_path = document_url
            file_info = get_document_info(file_path)
            file_category = FileTypeDetector.get_file_category(file_path)
            
            # Check if file should be forced to RAG processing
            if should_force_rag_processing(file_path, cache_enabled=True):
                print(f"File type '{file_category}' requires RAG processing - using LlamaParse")
                markdown_content = parse_document_optimized(file_path, use_temp_output=False)
                text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                document = client.files.upload(
                    file=text_bytes,
                    config=dict(mime_type='text/markdown')
                )
            elif file_info['is_pdf']:
                # Use intelligent PDF processing
                file_handle, mime_type, temp_cleanup = process_pdf_intelligently(file_path)
                temp_files_to_cleanup.extend(temp_cleanup)
                document = client.files.upload(
                    file=file_handle,
                    config=dict(mime_type=mime_type)
                )
            elif is_word_document(file_path):
                markdown_content = convert_word_to_markdown_fast(file_path)
                text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                document = client.files.upload(
                    file=text_bytes,
                    config=dict(mime_type='text/markdown')
                )
            elif is_presentation_document(file_path):
                print(f"Processing presentation document: {file_path}")
                markdown_content = parse_document_optimized(file_path, use_temp_output=False)
                text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                document = client.files.upload(
                    file=text_bytes,
                    config=dict(mime_type='text/markdown')
                )
            elif is_spreadsheet_document(file_path):
                print(f"Processing spreadsheet document: {file_path}")
                markdown_content = parse_document_optimized(file_path, use_temp_output=False)
                text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                document = client.files.upload(
                    file=text_bytes,
                    config=dict(mime_type='text/markdown')
                )
            elif is_email_file(file_path):
                markdown_content = parse_document_optimized(file_path, use_temp_output=False)
                text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                document = client.files.upload(
                    file=text_bytes,
                    config=dict(mime_type='text/markdown')
                )
            else:
                # Default to LlamaParse for other supported formats
                markdown_content = parse_document_optimized(file_path, use_temp_output=False)
                text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                document = client.files.upload(
                    file=text_bytes,
                    config=dict(mime_type='text/markdown')
                )
        else:
            # Handle URL-based documents
            file_category = FileTypeDetector.get_file_category(document_url)
            
            # Check if URL points to a file type that should be forced to RAG processing
            if should_force_rag_processing(document_url, cache_enabled=True):
                print(f"URL file type '{file_category}' requires RAG processing - downloading and parsing")
                downloaded_file_path = download_file_to_unique_path(document_url)
                temp_files_to_cleanup.append(downloaded_file_path)
                markdown_path = parse_document_optimized(downloaded_file_path, use_temp_output=True)
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                document = client.files.upload(
                    file=text_bytes,
                    config=dict(mime_type='text/markdown')
                )
            elif is_pdf_url(document_url):
                try:
                    pdf_stream = stream_pdf_from_url(document_url)
                    # Use intelligent PDF processing for URL-sourced PDFs
                    file_handle, mime_type, temp_cleanup = process_pdf_intelligently(
                        pdf_stream, 
                        is_url_source=True, 
                        original_url=document_url
                    )
                    temp_files_to_cleanup.extend(temp_cleanup)
                    document = client.files.upload(
                        file=file_handle,
                        config=dict(mime_type=mime_type)
                    )
                except Exception as e:
                    print(f"PDF processing failed, falling back to LlamaParse: {e}")
                    # Download to unique path for LlamaParse fallback
                    unique_download_path = download_file_to_unique_path(document_url, '.pdf')
                    temp_files_to_cleanup.append(unique_download_path)
                    markdown_path = parse_document_optimized(unique_download_path, use_temp_output=True)
                    with open(markdown_path, 'r', encoding='utf-8') as f:
                        markdown_content = f.read()
                    text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                    document = client.files.upload(
                        file=text_bytes,
                        config=dict(mime_type='text/markdown')
                    )
            else:
                # Download to unique path to avoid conflicts
                extension = FileTypeDetector.get_file_extension(document_url)
                downloaded_file_path = download_file_to_unique_path(document_url, extension or '.pdf')
                temp_files_to_cleanup.append(downloaded_file_path)
                
                if is_word_document(downloaded_file_path):
                    markdown_content = convert_word_to_markdown_fast(downloaded_file_path)
                    text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                    document = client.files.upload(
                        file=text_bytes,
                        config=dict(mime_type='text/markdown')
                    )
                elif is_presentation_document(downloaded_file_path):
                    print(f"Processing downloaded presentation: {downloaded_file_path}")
                    markdown_content = parse_document_optimized(downloaded_file_path, use_temp_output=False)
                    text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                    document = client.files.upload(
                        file=text_bytes,
                        config=dict(mime_type='text/markdown')
                    )
                elif is_spreadsheet_document(downloaded_file_path):
                    print(f"Processing downloaded spreadsheet: {downloaded_file_path}")
                    markdown_content = parse_document_optimized(downloaded_file_path, use_temp_output=False)
                    text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                    document = client.files.upload(
                        file=text_bytes,
                        config=dict(mime_type='text/markdown')
                    )
                elif is_email_file(downloaded_file_path):
                    markdown_content = parse_document_optimized(downloaded_file_path, use_temp_output=False)
                    text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                    document = client.files.upload(
                        file=text_bytes,
                        config=dict(mime_type='text/markdown')
                    )
                else:
                    # Use LlamaParse for other supported formats
                    markdown_path = parse_document_optimized(downloaded_file_path, use_temp_output=True)
                    with open(markdown_path, 'r', encoding='utf-8') as f:
                        markdown_content = f.read()
                    text_bytes = io.BytesIO(markdown_content.encode('utf-8'))
                    document = client.files.upload(
                        file=text_bytes,
                        config=dict(mime_type='text/markdown')
                    )

        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                system_instruction=system_instruction,
                contents=[document],
            )
        )
        return cache.name
        
    except Exception as e:
        print(f"Error creating cache: {e}")
        raise e
    finally:
        # Cleanup resources
        if file_handle and hasattr(file_handle, 'close'):
            try:
                file_handle.close()
            except:
                pass
        
        # Clean up all temporary files
        for temp_file in temp_files_to_cleanup:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    print(f"Warning: Could not clean up {temp_file}: {e}")
        
        if downloaded_file_path and os.path.exists(downloaded_file_path):
            try: 
                os.remove(downloaded_file_path)
            except: 
                pass
        if markdown_path and os.path.exists(markdown_path):
            try: 
                os.remove(markdown_path)
            except: 
                pass
        if document:
            del document
        gc.collect()

def create_batch_prompt(questions: list) -> str:
    """Create a unified prompt that always expects answers as a list"""
    if not questions:
        questions = [""]

    batch_prompt = f"""
    {SYSTEM_PROMPT}

    You will use the ReAct (Reasoning and Acting) framework to answer questions systematically.

    IMPORTANT INSTRUCTION: Do NOT copy exact text from any provided documents. Instead, use the documents as reference material to understand the topic and formulate your own original answers based on that understanding.

    For each question, follow this pattern:
    - THOUGHT: Analyze the question and understand what is being asked
    - ACTION: Review relevant document content to gather information and context
    - OBSERVATION: Synthesize the information from documents to form your understanding
    - FINAL ANSWER: Create an original answer based on your understanding of the reference material

    IMPORTANT: Always respond with a JSON object containing an 'answers' array.
    For a single question, use: {{"answers": ["your answer"]}}
    For multiple questions, use: {{"answers": ["answer 1", "answer 2", "..."]}}

    You must provide exactly {len(questions)} answer(s) in the answers array.

    Questions:
    """

    for i, question in enumerate(questions, 1):
        batch_prompt += f"""
    {i}. {question}

    THOUGHT: [Analyze this question - what is being asked?]
    ACTION: [Review the provided documents for relevant information and context]
    OBSERVATION: [What key concepts and information can you extract? How does this help answer the question?]
    FINAL ANSWER: [Create an original answer based on your understanding, not copying from documents]

    """

    batch_prompt += f"""
    After working through all questions using the ReAct framework above, provide ONLY the final JSON response below:

    Please respond in this exact JSON format with exactly {len(questions)} answer(s):
    {{
        "answers": [
            {',\n        '.join([f'"answer to question {i+1}"' for i in range(len(questions))])}
        ]
    }}

    CRITICAL: 
    - The answers array must contain exactly {len(questions)} string(s)
    - Your response should ONLY contain the JSON object - no reasoning, explanations, or additional text
    - Use the ReAct framework internally to think through each question, but output only the final answers
    - DO NOT copy exact text from documents - create original answers based on your understanding of the reference material
    - Use documents as context and reference, not as a source to copy from
    """
    return batch_prompt


def ask_batch_query_optimized(cache_name: str, questions: list) -> dict:
    """Unified processing - always returns answers as a list"""
    import time
    start_time = time.time()
    
    def _parse_response(response_text, questions):
        # Clean up markdown code blocks if present
        if response_text.startswith('```'):
            start_marker = response_text.find('```')
            end_marker = response_text.rfind('```')
            if start_marker != end_marker and start_marker != -1 and end_marker != -1:
                content = response_text[start_marker:end_marker]
                if content.startswith('```json'):
                    content = content[7:]
                else:
                    content = content[3:]
                response_text = content.strip()
        try:
            json_response = json.loads(response_text)
            print(f"DEBUG: Parsed JSON response: {json_response}")
            # Handle different response formats and normalize to list
            answers = []
            if isinstance(json_response, dict):
                if 'answers' in json_response:
                    answers = json_response['answers']
                elif 'answer' in json_response:
                    answers = [json_response['answer']]
                else:
                    answers = [str(json_response)]
            elif isinstance(json_response, list):
                answers = json_response
            else:
                answers = [str(json_response)]
            # Ensure we have the right number of answers
            while len(answers) < len(questions):
                answers.append("Error: No answer provided for this question")
            if len(answers) > len(questions):
                answers = answers[:len(questions)]
            return {"answers": answers}
        except Exception as e:
            print(f"JSON parsing failed: {e}")
            return {"answers": [f"Error: JSON parsing failed. {str(e)}"] * len(questions)}

    print(f"Starting batch query for {len(questions)} questions with cache: {cache_name[:20]}...")
    batch_prompt = create_batch_prompt(questions)
    
    # First attempt
    try:
        query_start = time.time()
        response = client.models.generate_content(
            model=model_name,
            contents=batch_prompt,
            config=types.GenerateContentConfig(
                cached_content=cache_name,
                response_schema=UnifiedAnswers,
                response_mime_type="application/json",
                temperature=TEMPERATURE
            )
        )
        query_time = time.time() - query_start
        print(f"Query completed in {query_time:.2f}s")
        
        response_text = response.text.strip()
        parsed = _parse_response(response_text, questions)
        # Check for 503 or model overload in response or error
        if any(
            (isinstance(ans, str) and ("503" in ans or "overload" in ans.lower() or "unavailable" in ans.lower()))
            for ans in parsed["answers"]
        ):
            print("Primary model failed or overloaded, retrying with fallback model...")
            # Use the same model for fallback to avoid INVALID_ARGUMENT error
            response_fb = client.models.generate_content(
                model=model_name,
                contents=batch_prompt,
                config=types.GenerateContentConfig(
                    cached_content=cache_name,
                    response_schema=UnifiedAnswers,
                    response_mime_type="application/json",
                    temperature=TEMPERATURE
                )
            )
            response_text_fb = response_fb.text.strip()
            parsed_fb = _parse_response(response_text_fb, questions)
            if any(
                (isinstance(ans, str) and ("503" in ans or "overload" in ans.lower() or "unavailable" in ans.lower()))
                for ans in parsed_fb["answers"]
            ):
                parsed_fb["answers"] = ["No answer available (all models failed)"] * len(questions)
            return parsed_fb
        return parsed
    except Exception as e:
        print(f"Batch query failed: {e}")
        if any(word in str(e).lower() for word in ["503", "unavailable", "overload"]):
            try:
                # Use the same model for fallback to avoid INVALID_ARGUMENT error
                response_fb = client.models.generate_content(
                    model=model_name,
                    contents=batch_prompt,
                    config=types.GenerateContentConfig(
                        cached_content=cache_name,
                        response_schema=UnifiedAnswers,
                        response_mime_type="application/json",
                        temperature=TEMPERATURE
                    )
                )
                response_text_fb = response_fb.text.strip()
                parsed_fb = _parse_response(response_text_fb, questions)
                if any(
                    (isinstance(ans, str) and ("503" in ans or "overload" in ans.lower() or "unavailable" in ans.lower()))
                    for ans in parsed_fb["answers"]
                ):
                    parsed_fb["answers"] = ["No answer available (all models failed)"] * len(questions)
                return parsed_fb
            except Exception as e2:
                print(f"Fallback model also failed: {e2}")
                return {"answers": ["No answer available (all models failed)"] * len(questions)}
        return {"answers": [f"Error: Unable to process the request. {str(e)}"] * len(questions)}

def process_questions_with_cache_batch(document_url: str, questions: list, max_batch_size: int = 5) -> dict:
    """
    Process questions in batches and return results.
    Cache is preserved for reuse until RAG processing is completed.
    If cache is disabled or file should be RAG'd, falls back to direct processing without cache.
    """
    if not questions:
        return {"answers": []}

    if isinstance(questions, str):
        questions = [questions]

    # Check if cache is enabled
    if not is_cache_enabled():
        print("Cache is disabled - using direct processing method")
        return process_questions_without_cache(document_url, questions)
    
    # Check if this file type should be forced to RAG processing
    if should_force_rag_processing(document_url, cache_enabled=True):
        file_category = FileTypeDetector.get_file_category(document_url)
        print(f"File type '{file_category}' requires RAG processing - skipping cache")
        return process_questions_without_cache(document_url, questions)

    cache_name = None
    all_answers = []

    try:
        print(f"Processing {len(questions)} questions for document: {document_url[:50]}...")
        # Use the tracking system to get or create cache
        cache_name = get_or_create_cache_with_tracking(document_url)
        
        # If cache creation failed (file should be RAG'd), fall back to direct processing
        if cache_name is None:
            print("Cache creation skipped - falling back to RAG processing")
            return process_questions_without_cache(document_url, questions)
            
        print(f"Using cache: {cache_name}")

        total_questions = len(questions)
        batch_start_time = time.time()
        
        # Process smaller batches for better responsiveness
        effective_batch_size = min(max_batch_size, 3)  # Limit to 3 questions per batch for faster processing
        
        for i in range(0, total_questions, effective_batch_size):
            batch = questions[i:i + effective_batch_size]
            batch_num = (i // effective_batch_size) + 1
            batch_time = time.time()
            print(f"Processing batch {batch_num} with {len(batch)} questions...")

            try:
                batch_response = ask_batch_query_optimized(cache_name, batch)
                batch_answers = batch_response.get('answers', [])
                all_answers.extend(batch_answers)
                
                batch_duration = time.time() - batch_time
                print(f"Batch {batch_num} completed successfully in {batch_duration:.2f}s")
                
                # Quick garbage collection to free memory
                gc.collect()

            except Exception as e:
                batch_duration = time.time() - batch_time
                print(f"Batch {batch_num} processing failed after {batch_duration:.2f}s: {str(e)}")
                error_msg = f"Batch processing failed: {str(e)}"
                all_answers.extend([error_msg] * len(batch))

        total_batch_time = time.time() - batch_start_time
        print(f"All batches completed in {total_batch_time:.2f} seconds")

        # Ensure we have exactly the right number of answers
        if len(all_answers) != len(questions):
            print(f"Warning: Answer count mismatch. Expected {len(questions)}, got {len(all_answers)}")
            if len(all_answers) < len(questions):
                all_answers.extend(["Error: Missing answer"] * (len(questions) - len(all_answers)))
            else:
                all_answers = all_answers[:len(questions)]

        return {"answers": all_answers}

    except Exception as e:
        print(f"Error in process_questions_with_cache_batch: {e}")
        return {"answers": [f"Error: {str(e)}"] * len(questions)}

# --- Cache cleanup for document ---
def cleanup_cache_for_document(document_url: str):
    """Delete cache for a specific document and remove from mapping"""
    if not is_cache_enabled():
        print("Cache is disabled - no cache cleanup needed")
        return
        
    doc_hash = get_document_hash(document_url)
    if doc_hash in _document_cache_map:
        cache_name = _document_cache_map[doc_hash]
        try:
            delete_cache(cache_name)
        except Exception as e:
            print(f"Error deleting cache {cache_name}: {e}")
        del _document_cache_map[doc_hash]
        save_cache_mapping()

# --- Dummy delete_cache for demonstration (implement as needed) ---
def delete_cache(cache_name: str):
    """Delete the cache by name (implement actual logic as needed)"""
    print(f"Deleting cache: {cache_name}")
    # Implement actual cache deletion logic here

def process_questions_with_cache(document_url: str, questions: list) -> dict:
    return process_questions_with_cache_batch(document_url, questions)

def process_questions_with_cache_json(document_url: str, questions: list) -> str:
    results = process_questions_with_cache_batch(document_url, questions)
    return json.dumps(results, indent=2, ensure_ascii=False)


def cleanup_all_caches():
    """Delete all active caches and clean up temp files."""
    if not is_cache_enabled():
        print("Cache is disabled - cleaning up temp files only")
        cleanup_temp_files()
        gc.collect()
        return
        
    for doc_id, cache_name in list(_active_caches.items()):
        try:
            delete_cache(cache_name)
        except Exception:
            pass
    _active_caches.clear()
    cleanup_temp_files()
    gc.collect()

if __name__ == "__main__":
    # Example usage (uncomment and adapt as needed):
    data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
        "questions": [
            "If my car is stolen, what case will it be in law?",
            "If I am arrested without a warrant, is that legal?",
            "If someone denies me a job because of my caste, is that allowed?",
            "If the government takes my land for a project, can I stop it?",
            "If my child is forced to work in a factory, is that legal?",
            "If I am stopped from speaking at a protest, is that against my rights?",
            "If a religious place stops me from entering because I'm a woman, is that constitutional?",
            "If I change my religion, can the government stop me?",
            "If the police torture someone in custody, what right is being violated?",
            "If I'm denied admission to a public university because I'm from a backward community, can I do something?"
        ]
    }
    results = process_questions_with_cache_batch(data["documents"], data["questions"], max_batch_size=10)
    print("Batch Processing Results:")
    print(json.dumps(results, indent=2))
