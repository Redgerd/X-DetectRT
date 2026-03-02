# backend/api/audio/routes.py
"""
READ – Audio Deepfake Detection API Routes

Endpoints:
    POST /audio/analyze           — Upload audio file, dispatch Celery pipeline
    GET  /audio/result/{task_id}  — Poll for detection + XAI results
"""

import json
import uuid
import base64
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Header, Cookie
from fastapi.responses import JSONResponse
from jose import jwt
from jose.exceptions import JWTError

from config import settings
from api.audio.schemas import AudioAnalysisResponse, AudioResultResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["Audio Detection"])

# Allowed audio MIME types
ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/flac", "audio/x-flac",
    "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/x-ogg",
    "audio/webm",
    "application/octet-stream",   # some clients send this for wav/flac
}

MAX_AUDIO_SIZE_BYTES = 50 * 1024 * 1024   # 50 MB


# ---------------------------------------------------------------------------
# Auth dependency (mirrors api/video/routes.py)
# ---------------------------------------------------------------------------

SECRET_KEY = getattr(settings, "SECRET_KEY", "your-secret-key-here")
ALGORITHM  = "HS256"


async def get_current_user(
    token: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
):
    token_from_header = None
    if authorization and authorization.startswith("Bearer "):
        token_from_header = authorization[7:]

    token_to_validate = token_from_header or token
    if not token_to_validate:
        return None

    try:
        payload = jwt.decode(token_to_validate, SECRET_KEY, algorithms=[ALGORITHM])
        exp: int = payload.get("exp")
        if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
            return None
        return {
            "user_id":  payload.get("id"),
            "username": payload.get("sub"),
            "role":     payload.get("role", "user"),
        }
    except JWTError:
        return None


# ---------------------------------------------------------------------------
# POST /audio/analyze
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=AudioAnalysisResponse, status_code=202)
async def analyze_audio(
    audio: UploadFile = File(..., description="Audio file (wav, flac, mp3, ogg …)"),
    task_id: str = Form(
        default="",
        description="Optional custom task ID; a UUID will be generated if omitted",
    ),
    current_user=Depends(get_current_user),
):
    """
    Submit an audio file for deepfake detection.

    The file is base64-encoded and forwarded to the Celery audio pipeline which:
      1. Preprocesses (resample to 16 kHz, pad/trim to 64 600 samples)
      2. Extracts WavLM features (Module A)
      3. Classifies with DeepFakeDetector (Module B)
      4. Generates Integrated Gradients + SHAP heatmaps (Module C, async)

    Results are available via GET /audio/result/{task_id} or via the
    WebSocket at /ws/audio/{task_id}.
    """
    # Generate task ID if not provided
    if not task_id:
        task_id = str(uuid.uuid4())

    user_info = f"user={current_user['username']}" if current_user else "anonymous"
    logger.info(f"[AudioAPI] /analyze called: task_id={task_id}, {user_info}")

    # Read and validate file
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_AUDIO_SIZE_BYTES // (1024*1024)} MB).",
        )

    # Encode to base64 for Celery transport
    content_type = audio.content_type or "audio/wav"
    audio_b64 = f"data:{content_type};base64," + base64.b64encode(audio_bytes).decode("utf-8")

    # Dispatch Celery task
    try:
        from core.celery.audio_detection_tasks import run_audio_pipeline
        celery_task = run_audio_pipeline.delay(task_id, audio_b64)
        logger.info(
            f"[AudioAPI] Celery task dispatched: celery_id={celery_task.id}, task_id={task_id}"
        )
    except Exception as e:
        logger.error(f"[AudioAPI] Failed to dispatch Celery task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start audio analysis: {e}")

    return AudioAnalysisResponse(
        task_id=task_id,
        celery_task_id=celery_task.id,
        message="Audio analysis started. Poll GET /audio/result/{task_id} or connect to WS /ws/audio/{task_id}.",
    )


# ---------------------------------------------------------------------------
# GET /audio/result/{task_id}
# ---------------------------------------------------------------------------

@router.get("/result/{task_id}", response_model=AudioResultResponse)
async def get_audio_result(
    task_id: str,
    current_user=Depends(get_current_user),
):
    """
    Poll for the detection verdict and XAI heatmaps for a given task.

    Status values:
      - ``pending``            — Detection task hasn't completed yet.
      - ``detection_complete`` — Verdict is ready; XAI heatmaps not yet available.
      - ``complete``           — Both verdict and heatmaps are ready.
    """
    import redis as _redis
    try:
        r = _redis.from_url(settings.REDIS_URL, decode_responses=True)
        detection_raw = r.get(f"audio_detection_result:{task_id}")
        xai_raw       = r.get(f"audio_xai_result:{task_id}")
        r.close()
    except Exception as e:
        logger.error(f"[AudioAPI] Redis error for task_id={task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Result store temporarily unavailable.")

    if not detection_raw:
        return AudioResultResponse(task_id=task_id, status="pending")

    try:
        detection = json.loads(detection_raw)
    except Exception:
        raise HTTPException(status_code=500, detail="Corrupted detection result in store.")

    response = AudioResultResponse(
        task_id=task_id,
        status="detection_complete",
        verdict=detection.get("verdict"),
        is_fake=detection.get("is_fake"),
        confidence=detection.get("confidence"),
        fake_prob=detection.get("fake_prob"),
        real_prob=detection.get("real_prob"),
        waveform_b64=detection.get("waveform_b64"),
    )

    if xai_raw:
        try:
            xai = json.loads(xai_raw)
            response.ig_heatmap_b64   = xai.get("ig_heatmap_b64")
            response.shap_heatmap_b64 = xai.get("shap_heatmap_b64")
            response.status = "complete"
        except Exception:
            pass   # XAI data corrupted — return detection result only

    return response
