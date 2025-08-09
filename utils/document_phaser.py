"""
Document Parser Module
Handles document downloading, parsing, and conversion operations.
"""

import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
import re
import tempfile
import requests
from pathlib import Path
from urllib.parse import urlparse
from llama_cloud_services import LlamaParse
from .config import LLAMA_CLOUD_API_KEY, DATA_DIR, TEMP_DIR
from .email_processor import is_email_file, process_email_file
import gc
import shutil

def get_file_type_from_path(file_path):
    """Determine file type from file path."""
    file_extension = Path(file_path).suffix.lower()
    return file_extension

def is_supported_file_type(file_path_or_url):
    """Check if file type is supported for processing."""
    from .file_type_detector import FileTypeDetector
    return FileTypeDetector.is_supported_file_type(file_path_or_url)

def get_file_processing_strategy(file_path_or_url, cache_enabled=True):
    """Get processing strategy for file based on type."""
    from .file_type_detector import FileTypeDetector
    return FileTypeDetector.get_processing_strategy(file_path_or_url, cache_enabled)

def is_pdf_url(url):
    """Determine if a URL is likely a PDF without downloading the entire file."""
    try:
        # First check URL path
        parsed_url = urlparse(url)
        if parsed_url.path.lower().endswith('.pdf'):
            return True
        
        # Make a HEAD request to check content-type
        response = requests.head(url, timeout=10, allow_redirects=True)
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' in content_type:
            return True
        
        # Check content-disposition header
        content_disposition = response.headers.get('content-disposition', '').lower()
        if 'pdf' in content_disposition:
            return True
        
        return False
    except Exception as e:
        print(f"Warning: Could not determine if URL is PDF: {e}")
        # If HEAD request fails, fall back to URL analysis
        return url.lower().endswith('.pdf') or '.pdf' in url.lower()

def is_supported_url(url):
    """Check if URL points to a supported file type."""
    return is_supported_file_type(url)

def download_file_streaming(url, filename=None):
    """Download a file from URL with streaming and memory optimization."""
    # Use temporary file to avoid keeping entire file in memory
    temp_file = None
    final_path = None

    try:
        # Determine filename and create temp file
        if filename is None:
            if 'docs.google.com/document/d/' in url:
                match = re.search(r'document/d/([\w-]+)', url)
                if match:
                    file_id = match.group(1)
                    url = f'https://docs.google.com/document/d/{file_id}/export?format=docx'
                    filename = 'data.docx'
            else:
                parsed_url = urlparse(url)
                path = parsed_url.path.lower()
                
                # Enhanced file type detection
                from .file_type_detector import FileTypeDetector
                extension = FileTypeDetector.get_file_extension(url)
                
                if extension:
                    filename = f'data{extension}'
                elif path.endswith('.pdf'):
                    filename = 'data.pdf'
                elif path.endswith(('.docx', '.doc')):
                    filename = 'data.docx'
                elif path.endswith(('.pptx', '.ppt')):
                    filename = 'data.pptx'
                elif path.endswith(('.xlsx', '.xls')):
                    filename = 'data.xlsx'
                elif path.endswith(('.eml', '.msg')):
                    filename = 'data.eml'
                else:
                    # Try to detect from content-type header
                    try:
                        head_response = requests.head(url, timeout=10, allow_redirects=True)
                        content_type = head_response.headers.get('content-type', '').lower()
                        
                        if 'pdf' in content_type:
                            filename = 'data.pdf'
                        elif 'wordprocessingml' in content_type or 'msword' in content_type:
                            filename = 'data.docx'
                        elif 'presentationml' in content_type or 'mspowerpoint' in content_type:
                            filename = 'data.pptx'
                        elif 'spreadsheetml' in content_type or 'msexcel' in content_type:
                            filename = 'data.xlsx'
                        else:
                            filename = 'data.pdf'  # Default fallback
                    except:
                        filename = 'data.pdf'  # Default fallback

        # Create temporary file
        suffix = Path(filename).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=DATA_DIR)

        # Stream download directly to temp file
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            
            # Check content type and adjust filename if needed
            content_type = response.headers.get('content-type', '').lower()
            if 'officedocument.wordprocessingml' in content_type or 'application/vnd.openxmlformats' in content_type:
                if not filename.endswith(('.docx', '.doc')):
                    filename = 'data.docx'
            elif 'officedocument.presentationml' in content_type:
                if not filename.endswith(('.pptx', '.ppt')):
                    filename = 'data.pptx'
            elif 'officedocument.spreadsheetml' in content_type:
                if not filename.endswith(('.xlsx', '.xls')):
                    filename = 'data.xlsx'
            elif 'application/pdf' in content_type:
                if not filename.endswith('.pdf'):
                    filename = 'data.pdf'

            # Stream write in chunks
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)

        temp_file.close()

        # Move to final location
        final_path = os.path.join(DATA_DIR, filename)
        os.makedirs(DATA_DIR, exist_ok=True)

        # If final path exists, remove it first
        if os.path.exists(final_path):
            os.remove(final_path)

        os.rename(temp_file.name, final_path)
        print(f"File downloaded successfully: {final_path}")
        return final_path

    except Exception as e:
        # Clean up on error
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except:
                pass
        raise e

def parse_document_optimized(file_path, use_temp_output=True):
    """Optimized document parsing with memory management."""
    
    # Check if it's an email file
    if is_email_file(file_path):
        print(f"Processing email file: {file_path}")
        return process_email_file(file_path, TEMP_DIR)
    
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY or os.getenv("LLAMA_CLOUD_API_KEY"),
        num_workers=2,  # Reduced workers to save memory
        verbose=False,  # Reduce console output for speed
        # fast_mode=True  # Use fast mode for quicker parsing
    )

    try:
        # Parse document
        result = parser.parse(file_path)
        markdown_documents = result.get_markdown_documents(split_by_page=False)

        if use_temp_output:
            # Save markdown to TEMP_DIR
            temp_md = tempfile.NamedTemporaryFile(mode='w', suffix='.md',
                                                delete=False, dir=TEMP_DIR, encoding='utf-8')
            try:
                for doc in markdown_documents:
                    temp_md.write(getattr(doc, 'text', str(doc)) + '\n\n')
                temp_md.close()
                # Copy to DATA_DIR for RAG
                os.makedirs(DATA_DIR, exist_ok=True)
                data_md_path = os.path.join(DATA_DIR, os.path.basename(temp_md.name))
                shutil.copy(temp_md.name, data_md_path)
                print(f"Markdown saved to TEMP_DIR: {temp_md.name} and DATA_DIR: {data_md_path}")
                return temp_md.name
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_md.name):
                    os.remove(temp_md.name)
                raise e
        else:
            # Return content directly in memory (for small files)
            content = ""
            for doc in markdown_documents:
                content += getattr(doc, 'text', str(doc)) + '\n\n'
            # Save to both TEMP_DIR and DATA_DIR
            temp_md_path = os.path.join(TEMP_DIR, "parsed_markdown.md")
            data_md_path = os.path.join(DATA_DIR, "parsed_markdown.md")
            os.makedirs(TEMP_DIR, exist_ok=True)
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(temp_md_path, 'w', encoding='utf-8') as f:
                f.write(content)
            shutil.copy(temp_md_path, data_md_path)
            print(f"Markdown saved to TEMP_DIR: {temp_md_path} and DATA_DIR: {data_md_path}")
            return temp_md_path

    finally:
        # Force garbage collection after parsing
        del result, markdown_documents
        gc.collect()

def parse_document_from_url(url, markdown_filename="parsed_markdown.md", cleanup_downloaded=False):
    """Download a document from URL, parse it with LlamaParse, and save markdown."""
    downloaded_file_path = None
    try:
        # Check if it's already a local file path
        if os.path.isfile(url):
            file_path = url
        else:
            # Download the file
            downloaded_file_path = download_file_streaming(url)
            file_path = downloaded_file_path

        # Use optimized parsing (saves to both TEMP_DIR and DATA_DIR)
        result_path = parse_document_optimized(file_path, use_temp_output=True)

        print(f"Markdown saved to TEMP_DIR: {result_path} and DATA_DIR.")
        return result_path  # Returns TEMP_DIR path

    finally:
        # Clean up downloaded file if requested and it was downloaded (not a local file)
        if cleanup_downloaded and downloaded_file_path and os.path.exists(downloaded_file_path):
            try:
                os.remove(downloaded_file_path)
                print(f"Cleaned up downloaded file: {downloaded_file_path}")
            except OSError as e:
                print(f"Warning: Could not delete downloaded file {downloaded_file_path}: {e}")

def get_document_info(file_path):
    """Get basic information about a document file."""
    if not os.path.exists(file_path):
        return None

    file_extension = get_file_type_from_path(file_path)
    file_size = os.path.getsize(file_path)

    return {
        'path': file_path,
        'extension': file_extension,
        'size_bytes': file_size,
        'size_mb': round(file_size / (1024 * 1024), 2),
        'is_pdf': file_extension == '.pdf',
        'is_large_file': file_size > 50 * 1024 * 1024  # 50MB threshold
    }

def cleanup_temp_files(directory=None):
    """Clean up temporary files in the specified directory or DATA_DIR."""
    if directory is None:
        directory = DATA_DIR

    if not os.path.exists(directory):
        return

    temp_patterns = ['temp_', 'tmp_', 'cache_', 'parsed_', 'data.']
    # Do not delete any markdown files
    cleaned_count = 0

    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                # Skip any markdown files
                if filename.endswith('.md'):
                    continue
                # Check if file matches temp patterns
                if any(filename.startswith(pattern) for pattern in temp_patterns):
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                        print(f"Cleaned up: {filename}")
                    except OSError as e:
                        print(f"Warning: Could not delete {filename}: {e}")

        if cleaned_count > 0:
            print(f"Cleaned up {cleaned_count} temporary files")
        else:
            print("No temporary files found to clean up")

    except Exception as e:
        print(f"Error during cleanup: {e}")

# Backward compatibility aliases
download_file = download_file_streaming

if __name__ == "__main__":
    # Example usage
    url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    markdown_file = parse_document_from_url(url, cleanup_downloaded=False)
    print(f"Markdown file created at: {markdown_file}")

def is_url_allowed(url: str) -> bool:
    """
    Check if a URL is allowed for processing
    
    Args:
        url: URL to check
        
    Returns:
        True if URL is allowed, False if it should be blocked
    """
    from .url_blocker import validate_url
    is_valid, _ = validate_url(url)
    return is_valid
