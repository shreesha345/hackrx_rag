"""
URL Blocker Module
Handles URL validation and blocking of specific URLs
"""

from typing import List, Tuple, Union
import logging
from .config import URL_BLOCKER_ENABLED

logger = logging.getLogger(__name__)

# List of blocked URLs
BLOCKED_URLS = [
    "https://register.hackrx.in/utils/get-secret-token",
    "https://register.hackrx.in/utils/get-secret-token?hackTeam=5693",
    # Add more variations of the URL to ensure comprehensive blocking
    "register.hackrx.in/utils/get-secret-token"
]

# Block messages to return
BLOCK_MESSAGES = {
    "default": "Sorry, I am a document assistant and I am not allowed to provide you with authentication tokens or access sensitive information. This URL has been blocked for security reasons."
}

def add_blocked_url(url: str) -> dict:
    """
    Add a URL to the blocked list
    
    Args:
        url: URL to block
        
    Returns:
        Dict with updated URL blocker status
    """
    global BLOCKED_URLS
    
    # Check if URL is already blocked
    if url in BLOCKED_URLS:
        logger.info(f"URL {url} is already blocked")
        return get_url_blocker_status()
    
    # Add URL to blocked list
    BLOCKED_URLS.append(url)
    logger.info(f"Added URL to block list: {url}")
    
    return get_url_blocker_status()

def remove_blocked_url(url: str) -> dict:
    """
    Remove a URL from the blocked list
    
    Args:
        url: URL to unblock
        
    Returns:
        Dict with updated URL blocker status
    """
    global BLOCKED_URLS
    
    # Check if URL is in the blocked list
    if url not in BLOCKED_URLS:
        logger.info(f"URL {url} is not in the blocked list")
        return get_url_blocker_status()
    
    # Remove URL from blocked list
    BLOCKED_URLS.remove(url)
    logger.info(f"Removed URL from block list: {url}")
    
    return get_url_blocker_status()

def is_blocked_url(url: str) -> Tuple[bool, str]:
    """
    Check if a URL is in the blocked list
    
    Args:
        url: URL to check
        
    Returns:
        Tuple of (is_blocked, message)
    """
    # Normalize URL for comparison
    url_lower = url.lower()
    
    # Check for exact matches
    if url in BLOCKED_URLS:
        return True, BLOCK_MESSAGES["default"]
    
    # Check for partial matches (for URLs with parameters)
    for blocked_url in BLOCKED_URLS:
        if url_lower.startswith(blocked_url.lower()):
            return True, BLOCK_MESSAGES["default"]
        
        # Check if URL contains any of the blocked URLs (for URLs that might be embedded)
        if blocked_url.lower() in url_lower:
            return True, BLOCK_MESSAGES["default"]
            
    # Check for "hackTeam=5693" parameter regardless of domain
    if "hackteam=5693" in url_lower:
        return True, BLOCK_MESSAGES["default"]
    
    # Check for "get-secret-token" path regardless of domain
    if "get-secret-token" in url_lower:
        return True, BLOCK_MESSAGES["default"]
            
    return False, ""

def validate_url(url: str) -> Tuple[bool, Union[str, None]]:
    """
    Validate a URL and check if it should be blocked
    
    Args:
        url: URL to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if URL is valid and not blocked
        - error_message: Error message if URL is invalid or blocked, None otherwise
    """
    # Check if URL blocking is enabled
    if not URL_BLOCKER_ENABLED:
        logger.info(f"URL blocker is disabled, allowing URL: {url}")
        return True, None
    
    # Check if URL is blocked
    is_blocked, block_message = is_blocked_url(url)
    if is_blocked:
        logger.warning(f"Blocked URL: {url}")
        return False, block_message
    
    # URL is valid and not blocked
    return True, None

def get_url_blocker_status() -> dict:
    """
    Get the current status of the URL blocker
    
    Returns:
        Dict with URL blocker status information
    """
    global URL_BLOCKER_ENABLED
    
    return {
        "enabled": URL_BLOCKER_ENABLED,
        "blocked_urls_count": len(BLOCKED_URLS),
        "blocked_urls": BLOCKED_URLS if URL_BLOCKER_ENABLED else []
    }

def toggle_url_blocker(enable: bool = None) -> dict:
    """
    Toggle the URL blocker on or off
    
    Args:
        enable: Set to True to enable, False to disable, None to toggle
        
    Returns:
        Dict with updated URL blocker status
    """
    global URL_BLOCKER_ENABLED
    
    # If enable is None, toggle the current state
    if enable is None:
        URL_BLOCKER_ENABLED = not URL_BLOCKER_ENABLED
    else:
        URL_BLOCKER_ENABLED = enable
    
    logger.info(f"URL blocker {'enabled' if URL_BLOCKER_ENABLED else 'disabled'}")
    
    # Return the updated status
    return get_url_blocker_status()
