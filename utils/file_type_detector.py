"""
File Type Detection and Classification Module
Handles detection and classification of various document types for RAG processing.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Set
from urllib.parse import urlparse


class FileTypeDetector:
    """
    Comprehensive file type detection and classification system.
    Categorizes files into different processing strategies based on type.
    """
    
    # Document types - cacheable
    DOCUMENT_EXTENSIONS = {
        '.pdf', '.doc', '.docx', '.docm', '.dot', '.dotm', '.rtf', '.xml', '.epub',
        '.pages', '.abw', '.cwk', '.lwp', '.mw', '.mcw', '.wpd', '.wps', '.zabw',
        '.uof', '.uot', '.uop', '.sxi', '.sti', '.sdw', '.sgl', '.stw', '.sdd',
        '.sxw', '.sxg', '.602'
    }
    
    # Presentation types - should be RAG'd, not cached
    PRESENTATION_EXTENSIONS = {
        '.ppt', '.pptx', '.pptm', '.pot', '.potx', '.potm', '.key'
    }
    
    # Spreadsheet types - should be RAG'd, not cached
    SPREADSHEET_EXTENSIONS = {
        '.xlsx', '.xls', '.xlsm', '.xlsb', '.xlw', '.csv', '.tsv', '.dif', '.sylk',
        '.slk', '.prn', '.numbers', '.et', '.ods', '.fods', '.uos1', '.uos2',
        '.dbf', '.wk1', '.wk2', '.wk3', '.wk4', '.wks', '.123', '.wq1', '.wq2',
        '.wb1', '.wb2', '.wb3', '.qpw', '.xlr', '.eth'
    }
    
    # Image types - cacheable
    IMAGE_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.tiff', '.webp', '.web'
    }
    
    # HTML types - cacheable
    HTML_EXTENSIONS = {
        '.htm', '.html'
    }
    
    # Email types - cacheable
    EMAIL_EXTENSIONS = {
        '.eml', '.msg'
    }
    
    # Archive and compressed files - UNSUPPORTED
    ARCHIVE_EXTENSIONS = {
        '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.z', '.tgz', '.tbz2',
        '.txz', '.lz', '.lzma', '.cab', '.iso', '.dmg', '.pkg', '.deb', '.rpm',
        '.msi', '.exe', '.app', '.apk', '.jar', '.war', '.ear'
    }
    
    # Binary and system files - UNSUPPORTED  
    BINARY_EXTENSIONS = {
        '.bin', '.dat', '.tmp', '.log', '.bak', '.old', '.orig', '.swp', '.swo',
        '.cache', '.lock', '.pid', '.so', '.dll', '.dylib', '.lib', '.o', '.obj',
        '.pyc', '.pyo', '.class'
    }
    
    # Media files - UNSUPPORTED (too large or not text-based)
    MEDIA_EXTENSIONS = {
        '.mp3', '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v',
        '.m4a', '.wav', '.flac', '.ogg', '.aac', '.wma', '.3gp', '.f4v'
    }
    
    # All explicitly unsupported extensions
    UNSUPPORTED_EXTENSIONS = ARCHIVE_EXTENSIONS | BINARY_EXTENSIONS | MEDIA_EXTENSIONS
    
    # All supported extensions
    ALL_SUPPORTED_EXTENSIONS = (
        DOCUMENT_EXTENSIONS | PRESENTATION_EXTENSIONS | SPREADSHEET_EXTENSIONS |
        IMAGE_EXTENSIONS | HTML_EXTENSIONS | EMAIL_EXTENSIONS
    )
    
    # Extensions that should NEVER be cached (always RAG'd)
    FORCE_RAG_EXTENSIONS = PRESENTATION_EXTENSIONS | SPREADSHEET_EXTENSIONS
    
    @classmethod
    def get_file_extension(cls, file_path_or_url: str) -> str:
        """Extract file extension from file path or URL."""
        if file_path_or_url.startswith(('http://', 'https://')):
            parsed_url = urlparse(file_path_or_url)
            path = parsed_url.path
        else:
            path = file_path_or_url
        
        return Path(path).suffix.lower()
    
    @classmethod
    def is_supported_file_type(cls, file_path_or_url: str) -> bool:
        """Check if the file type is supported for processing."""
        extension = cls.get_file_extension(file_path_or_url)
        
        # Explicitly reject unsupported file types
        if extension in cls.UNSUPPORTED_EXTENSIONS:
            return False
            
        # Accept supported file types
        return extension in cls.ALL_SUPPORTED_EXTENSIONS
    
    @classmethod
    def get_file_category(cls, file_path_or_url: str) -> str:
        """
        Determine the category of file for processing strategy.
        Returns: 'document', 'presentation', 'spreadsheet', 'image', 'html', 'email', 'unsupported'
        """
        extension = cls.get_file_extension(file_path_or_url)
        
        # Check for explicitly unsupported file types first
        if extension in cls.UNSUPPORTED_EXTENSIONS:
            if extension in cls.ARCHIVE_EXTENSIONS:
                return 'archive'
            elif extension in cls.BINARY_EXTENSIONS:
                return 'binary'
            elif extension in cls.MEDIA_EXTENSIONS:
                return 'media'
            else:
                return 'unsupported'
        
        # Check supported file types
        if extension in cls.DOCUMENT_EXTENSIONS:
            return 'document'
        elif extension in cls.PRESENTATION_EXTENSIONS:
            return 'presentation'
        elif extension in cls.SPREADSHEET_EXTENSIONS:
            return 'spreadsheet'
        elif extension in cls.IMAGE_EXTENSIONS:
            return 'image'
        elif extension in cls.HTML_EXTENSIONS:
            return 'html'
        elif extension in cls.EMAIL_EXTENSIONS:
            return 'email'
        else:
            return 'unsupported'
    
    @classmethod
    def should_force_rag(cls, file_path_or_url: str) -> bool:
        """
        Check if file type should be forced to RAG processing (not cached).
        Presentations and spreadsheets should always be RAG'd.
        """
        extension = cls.get_file_extension(file_path_or_url)
        return extension in cls.FORCE_RAG_EXTENSIONS
    
    @classmethod
    def can_be_cached(cls, file_path_or_url: str, cache_enabled: bool = True) -> bool:
        """
        Check if file can be cached based on type and cache settings.
        """
        if not cache_enabled:
            return False
        
        # Force RAG for certain file types
        if cls.should_force_rag(file_path_or_url):
            return False
        
        # Check if file type is supported and cacheable
        extension = cls.get_file_extension(file_path_or_url)
        cacheable_extensions = (
            cls.DOCUMENT_EXTENSIONS | cls.IMAGE_EXTENSIONS | 
            cls.HTML_EXTENSIONS | cls.EMAIL_EXTENSIONS
        )
        return extension in cacheable_extensions
    
    @classmethod
    def get_processing_strategy(cls, file_path_or_url: str, cache_enabled: bool = True) -> Dict[str, any]:
        """
        Get comprehensive processing strategy for a file.
        Returns dictionary with processing recommendations.
        """
        extension = cls.get_file_extension(file_path_or_url)
        category = cls.get_file_category(file_path_or_url)
        should_cache = cls.can_be_cached(file_path_or_url, cache_enabled)
        force_rag = cls.should_force_rag(file_path_or_url)
        
        strategy = {
            'extension': extension,
            'category': category,
            'is_supported': cls.is_supported_file_type(file_path_or_url),
            'should_cache': should_cache,
            'force_rag': force_rag,
            'processing_method': 'rag' if force_rag or not cache_enabled else ('cache' if should_cache else 'rag'),
            'requires_llamaparse': category in ['document', 'presentation', 'spreadsheet'],
            'can_direct_upload': category in ['image', 'html'],
        }
        
        return strategy
    
    @classmethod
    def get_supported_extensions_by_category(cls) -> Dict[str, Set[str]]:
        """Get all extensions grouped by category (including unsupported ones for reference)."""
        return {
            'documents': cls.DOCUMENT_EXTENSIONS,
            'presentations': cls.PRESENTATION_EXTENSIONS,
            'spreadsheets': cls.SPREADSHEET_EXTENSIONS,
            'images': cls.IMAGE_EXTENSIONS,
            'html': cls.HTML_EXTENSIONS,
            'email': cls.EMAIL_EXTENSIONS,
            'archives': cls.ARCHIVE_EXTENSIONS,
            'binary_files': cls.BINARY_EXTENSIONS,
            'media_files': cls.MEDIA_EXTENSIONS
        }
    
    @classmethod
    def get_file_info(cls, file_path_or_url: str, cache_enabled: bool = True) -> Dict[str, any]:
        """
        Get comprehensive file information and processing recommendations.
        """
        strategy = cls.get_processing_strategy(file_path_or_url, cache_enabled)
        
        info = {
            'file_path_or_url': file_path_or_url,
            'extension': strategy['extension'],
            'category': strategy['category'],
            'is_supported': strategy['is_supported'],
            'processing_strategy': strategy,
            'recommendations': {
                'use_cache': strategy['should_cache'],
                'force_rag': strategy['force_rag'],
                'use_llamaparse': strategy['requires_llamaparse'],
                'processing_method': strategy['processing_method']
            }
        }
        
        return info


# Utility functions for backward compatibility
def get_file_type_from_path(file_path: str) -> str:
    """Get file extension from path (backward compatibility)."""
    return FileTypeDetector.get_file_extension(file_path)

def is_supported_document_type(file_path_or_url: str) -> bool:
    """Check if document type is supported (backward compatibility)."""
    return FileTypeDetector.is_supported_file_type(file_path_or_url)

def should_use_rag_processing(file_path_or_url: str, cache_enabled: bool = True) -> bool:
    """Determine if file should use RAG processing instead of caching."""
    strategy = FileTypeDetector.get_processing_strategy(file_path_or_url, cache_enabled)
    return strategy['processing_method'] == 'rag'

def is_presentation_or_spreadsheet(file_path_or_url: str) -> bool:
    """Check if file is a presentation or spreadsheet type."""
    category = FileTypeDetector.get_file_category(file_path_or_url)
    return category in ['presentation', 'spreadsheet']


if __name__ == "__main__":
    # Test the file type detector
    test_files = [
        "document.pdf",
        "presentation.pptx",
        "spreadsheet.xlsx", 
        "image.jpg",
        "webpage.html",
        "email.eml",
        "archive.zip",
        "binary.bin",
        "media.mp4",
        "unsupported.txt"
    ]
    
    print("File Type Detection Test Results:")
    print("=" * 50)
    
    for test_file in test_files:
        info = FileTypeDetector.get_file_info(test_file, cache_enabled=True)
        print(f"\nFile: {test_file}")
        print(f"  Category: {info['category']}")
        print(f"  Supported: {info['is_supported']}")
        if info['is_supported']:
            print(f"  Processing: {info['recommendations']['processing_method']}")
            print(f"  Force RAG: {info['recommendations']['force_rag']}")
            print(f"  Can Cache: {info['recommendations']['use_cache']}")
        else:
            print(f"  Reason: File type not supported for processing")
