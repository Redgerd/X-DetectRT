from fastapi import APIRouter, Depends, Cookie, HTTPException, Header, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
import uuid
import base64
from typing import Optional
from datetime import datetime
from jose import jwt
from jose.exceptions import JWTError
from config import settings, XAI_CONFIG
from sqlalchemy.orm import joinedload

# Import the Celery task for GenD inference
from core.celery.detection_tasks import run_gend_inference

# Import models and database
from models import VideoAnalysisTask, DetectionResult, Users, ProcessedFrame
from core.database import SessionLocal

import json 
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
    explain: bool = Query(False, description="Run XAI pipeline and return explanations"),
    current_user = Depends(get_current_user)
):
    """
    Test endpoint for GenD deepfake detection on a single image.
    
    Args:
        image      : Image file (JPEG, PNG, etc.)
        task_id    : Task ID for tracking
        frame_index: Optional frame index
        timestamp  : Optional timestamp
        explain    : When True, run all XAI techniques and return results
                     alongside the main prediction.

    Returns:
        JSON with detection results and (when explain=True) a list of XAI
        technique results: { technique, scores, figure_base64 }
    """
    import logging
    import numpy as np
    import cv2
    from PIL import Image as PILImage

    logger = logging.getLogger(__name__)

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
            image_base64_with_prefix = f"data:{content_type};base64,{image_base64}"
        else:
            image_base64_with_prefix = image_base64

        if explain:
            # ----------------------------------------------------------------
            # Synchronous path: run inference + XAI inline (for direct API use)
            # ----------------------------------------------------------------
            import torch
            import torch.nn.functional as F
            from services.detection.model import load_gend_model, _GEND_DEVICE
            from xai.pipeline import run_xai_pipeline

            # Decode image to PIL
            nparr = np.frombuffer(image_content, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(frame_rgb)

            model = load_gend_model()
            device = _GEND_DEVICE or "cpu"

            # Run main inference
            tensor = model.feature_extractor.preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=-1)
            real_prob = probs[0, 0].item()
            fake_prob = probs[0, 1].item()
            is_anomaly = fake_prob > 0.5

            # Run XAI pipeline
            try:
                xai_results = run_xai_pipeline(
                    model=model,
                    pil_image=pil_image,
                    device=device,
                    frame_probs=[fake_prob],   # single-frame time series for SHAP
                    config=XAI_CONFIG,
                )
            except Exception as xai_err:
                logger.error(f"[XAI] Pipeline error: {xai_err}", exc_info=True)
                xai_results = [{"technique": "pipeline", "error": str(xai_err), "figure_base64": None}]

            return JSONResponse({
                "task_id": task_id,
                "frame_index": frame_index,
                "timestamp": timestamp,
                "is_anomaly": is_anomaly,
                "real_prob": round(real_prob, 4),
                "fake_prob": round(fake_prob, 4),
                "confidence": round(fake_prob * 100 if is_anomaly else real_prob * 100, 2),
                "anomaly_type": "GenD Deepfake" if is_anomaly else None,
                "xai_results": xai_results,
            })
        else:
            # ----------------------------------------------------------------
            # Async path: dispatch Celery task (original behaviour)
            # ----------------------------------------------------------------
            task = run_gend_inference.delay(
                task_id=task_id,
                frame_data=image_base64_with_prefix,
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


@router.get("/anomalies/count")
async def get_anomalies_count(current_user = Depends(get_current_user)):
    """
    Get the total count of anomalies detected in videos.
    Admins see all anomalies, regular users see only their own.
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    db = SessionLocal()
    try:
        query = db.query(DetectionResult).join(ProcessedFrame, DetectionResult.frame_id == ProcessedFrame.id).join(VideoAnalysisTask, ProcessedFrame.task_id == VideoAnalysisTask.task_id).filter(DetectionResult.is_anomaly == True)
        if current_user['role'] != 'admin':
            query = query.filter(VideoAnalysisTask.user_id == current_user['user_id'])
        count = query.count()
        return {"total_anomalies": count}
    finally:
        db.close()


@router.get("/test/anomalies/count")
async def get_anomalies_count_test():
    """
    Test route: Get the total count of anomalies detected in all videos (no auth required).
    """
    db = SessionLocal()
    try:
        count = db.query(DetectionResult).filter(DetectionResult.is_anomaly == True).count()
        return {"total_anomalies": count}
    finally:
        db.close()


@router.get("/test/detections")
async def get_detections_test(task_id: str = Query(..., description="Task ID to get detections for")):
    """
    Test route: Get all detection results for a specific video task (no auth required).
    """
    db = SessionLocal()
    try:
        # Join DetectionResult with ProcessedFrame to get frame info
        detections = db.query(DetectionResult, ProcessedFrame).join(ProcessedFrame, DetectionResult.frame_id == ProcessedFrame.id).filter(ProcessedFrame.task_id == task_id).all()

        result = []
        for detection, frame in detections:
            result.append({
                "id": detection.id,
                "frame_id": detection.frame_id,
                "is_anomaly": detection.is_anomaly,
                "confidence": detection.confidence,
                "real_prob": detection.real_prob,
                "fake_prob": detection.fake_prob,
                "anomaly_type": detection.anomaly_type,
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
            })
        return {"detections": result}
    finally:
        db.close()


@router.get("/test/xai")
async def get_xai_test(task_id: str = Query(..., description="Task ID to get XAI results for")):
    """
    Test route: Get all XAI results for a specific video task (no auth required).
    """
    db = SessionLocal()
    try:
        from models import XAIResult
        # Join XAIResult with ProcessedFrame to get frame info
        xai_results = db.query(XAIResult, ProcessedFrame).join(ProcessedFrame, XAIResult.frame_id == ProcessedFrame.id).filter(ProcessedFrame.task_id == task_id).all()

        result = []
        for xai, frame in xai_results:
            result.append({
                "id": xai.id,
                "frame_id": xai.frame_id,
                "gradcam_b64": xai.gradcam_b64,
                "ela_b64": xai.ela_b64,
                "fft_data": json.loads(xai.fft_data) if xai.fft_data else None,
                "lime_data": json.loads(xai.lime_data) if xai.lime_data else None,
                "error": xai.error,
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
            })
        return {"xai_results": result}
    finally:
        db.close()


@router.get("/test/all")
async def get_all_videos_test():
    """
    Test route: Get all video analysis data including the user who uploaded them (no auth required).
    """
    db = SessionLocal()
    try:
        tasks = db.query(VideoAnalysisTask).options(joinedload(VideoAnalysisTask.user)).all()

        result = []
        for task in tasks:
            # Check if this task has any anomalies
            has_anomalies = db.query(DetectionResult).join(ProcessedFrame).filter(
                ProcessedFrame.task_id == task.task_id,
                DetectionResult.is_anomaly == True
            ).count() > 0

            result.append({
                "task_id": task.task_id,
                "video_path": task.video_path,
                "status": task.status.value,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "faces_detected_frames": task.faces_detected_frames,
                "frames_skipped": task.frames_skipped,
                "has_anomalies": has_anomalies,
                "user": {
                    "id": task.user.id,
                    "username": task.user.username,
                    "email": task.user.email,
                    "role": task.user.role.value
                }
            })
        return {"videos": result}
    finally:
        db.close()


@router.get("/all")
async def get_all_videos(current_user = Depends(get_current_user)):
    """
    Get all video analysis data including the user who uploaded them.
    Admins see all videos, regular users see only their own.
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    db = SessionLocal()
    try:
        query = db.query(VideoAnalysisTask).options(joinedload(VideoAnalysisTask.user))
        if current_user['role'] != 'admin':
            query = query.filter(VideoAnalysisTask.user_id == current_user['user_id'])

        tasks = query.all()

        result = []
        for task in tasks:

            # Check if this task has any anomalies
            has_anomalies = db.query(DetectionResult).join(ProcessedFrame).filter(
                ProcessedFrame.task_id == task.task_id,
                DetectionResult.is_anomaly == True
            ).count() > 0

            result.append({
                "task_id": task.task_id,
                "video_path": task.video_path,
                "status": task.status.value,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "faces_detected_frames": task.faces_detected_frames,
                "frames_skipped": task.frames_skipped,
                "has_anomalies": has_anomalies,
                "user": {
                    "id": task.user.id,
                    "username": task.user.username,
                    "email": task.user.email,
                    "role": task.user.role.value
                }
            })
        return {"videos": result}
    finally:
        db.close()

