import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
import json

URL_STORE_PATH = os.path.join(os.path.dirname(__file__), 'stored_url.json')

def store_or_check_document_url(url: str) -> bool:
    """
    If the document URL is already stored, return True.
    If not, store the new URL and return False.
    The URL is stored in a JSON file for extensibility.
    """
    url = url.strip()
    data = {}
    if os.path.exists(URL_STORE_PATH):
        with open(URL_STORE_PATH, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        stored_url = data.get('url', '')
        if stored_url == url:
            return True
    # Store the new URL in JSON
    with open(URL_STORE_PATH, 'w') as f:
        json.dump({'url': url}, f)
    return False


if __name__ == "__main__":
    # Example usage
    url = "https://example.com/document.pdf"
    if store_or_check_document_url(url):
        print("URL already stored.")
    else:
        print("New URL stored.")