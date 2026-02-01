from fastapi import APIRouter, Depends, Cookie, HTTPException, Header
from fastapi.responses import JSONResponse
import uuid
from typing import Optional
from datetime import datetime
from jose import jwt
from jose.exceptions import JWTError
from config import settings

router = APIRouter(prefix="/video", tags=["Video Processing"])

# Simple secret key for JWT validation (use settings.SECRET_KEY in production)
SECRET_KEY = getattr(settings, 'SECRET_KEY', 'your-secret-key-here')
ALGORITHM = "HS256"

async def get_current_user(token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    """
    Validate JWT token from either cookie or Authorization header
    """
    token_from_header = None
    
    # Check Authorization header first
    if authorization and authorization.startswith("Bearer "):
        token_from_header = authorization[7:]
    
    # Use token from header, fallback to cookie
    token_to_validate = token_from_header or token
    
    if not token_to_validate:
        return None
    
    try:
        # Decode JWT token
        payload = jwt.decode(token_to_validate, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("id")
        username: str = payload.get("sub")
        role: str = payload.get("role", "user")
        exp: int = payload.get("exp")
        
        # Check if token is expired
        if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
            return None
            
        return {"user_id": user_id, "username": username, "role": role}
    except JWTError:
        return None

@router.post("/start-task")
async def start_task(current_user = Depends(get_current_user)):
    """
    Start a new video processing task.
    Requires authentication via JWT token (cookie or Authorization header).
    """
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Log user info if authenticated
    if current_user:
        print(f"Task started by user: {current_user.get('username')} (ID: {current_user.get('user_id')})")
    else:
        print(f"Task started by anonymous user")
    
    return JSONResponse({
        "task_id": task_id,
        "message": "Task started successfully"
    })
