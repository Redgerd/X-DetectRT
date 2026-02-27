from fastapi import APIRouter, Depends, Cookie, HTTPException, Header, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uuid
import base64
from typing import Optional
from datetime import datetime
from jose import jwt
from jose.exceptions import JWTError
from config import settings

# Import the Celery task for GenD inference
from core.celery.detection_tasks import run_gend_inference

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


@router.post("/test-gend-inference")
async def test_gend_inference(
    image: UploadFile = File(..., description="Image file for deepfake detection"),
    task_id: str = Form(..., description="Task ID for tracking"),
    frame_index: int = Form(0, description="Frame index in the video"),
    timestamp: str = Form("", description="Frame timestamp"),
    current_user = Depends(get_current_user)
):
    """
    Test endpoint for GenD deepfake detection on a single image.
    Takes an image file and task_id, returns detection results.
    
    Args:
        image: Image file (JPEG, PNG, etc.)
        task_id: Task ID for tracking the request
        frame_index: Optional frame index
        timestamp: Optional timestamp
    
    Returns:
        JSON with detection results (real_prob, fake_prob, is_anomaly, confidence)
    """
    # Validate task_id
    if not task_id:
        task_id = str(uuid.uuid4())
    
    # Read and encode the image
    try:
        image_content = await image.read()
        image_base64 = base64.b64encode(image_content).decode("utf-8")
        
        # Determine MIME type
        content_type = image.content_type
        if content_type:
            image_base64 = f"data:{content_type};base64,{image_base64}"
        
        # Dispatch the Celery task
        task = run_gend_inference.delay(
            task_id=task_id,
            frame_data=image_base64,
            frame_index=frame_index,
            timestamp=timestamp
        )
        
        return JSONResponse({
            "message": "GenD inference task dispatched",
            "task_id": task_id,
            "celery_task_id": task.id,
            "frame_index": frame_index,
            "timestamp": timestamp
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
