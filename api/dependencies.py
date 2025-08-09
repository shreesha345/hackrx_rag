from dotenv import load_dotenv
load_dotenv()
"""
FastAPI dependencies for authentication and common functionality
"""
from fastapi import HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Import TOKEN from config
from utils.config import TOKEN

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the bearer token against the TOKEN from config"""
    if credentials.credentials != (TOKEN or os.getenv("TOKEN")):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return credentials.credentials

def get_current_user(token: str = Depends(verify_token)):
    """Get current authenticated user (placeholder for future expansion)"""
    return {"authenticated": True, "token": token}
