# backend/main.py

# imports
import asyncio
import logging
from fastapi import FastAPI,  WebSocket
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
from config import settings

# Internal Imports
from core.database import test_db_connection
from core.celery.celery_app import celery_app
from core.celery.frame_selection import extract_faces_with_optical_flow
from services.detection.model import load_gend_model
import os, uuid

# FastAPI App Setup
app = FastAPI(
    title="X-DETECT API",
    description="API for deepfake project.",
    version="1.0.0",
    openapi_tags=[{"name": "Auth", "description": "Authentication related endpoints"}],
    openapi_security=[{
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }]
)

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup and Shutdown Events
# Startup and Shutdown Events
@app.on_event("startup")
async def startup_db_check():
    # ------------------------------
    # Database Check
    # ------------------------------
    # if test_db_connection():
    #     logger.info("✅ Database connected successfully.")
    # else:
    #     logger.error("❌ Database connection failed on startup.")

    # ------------------------------
    # Redis Initialization
    # ------------------------------
    global redis_client
    try:
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        logger.info("✅ Redis client initialized.")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}", exc_info=True)

    # ------------------------------
    # Load GenD Model
    # ------------------------------
    try:
        logger.info("🚀 Loading GenD model at startup...")
        load_gend_model()
        logger.info("✅ GenD model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load GenD model: {e}", exc_info=True)
        raise e  # optional: crash app if model fails


@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()
        logger.info("Redis client closed.")
    logger.info("App shutdown complete.")
    
# Routes
from api.auth.routes import router as auth_router
from api.video.websocket import router as video_ws_router
from api.video.routes import router as video_router

app.include_router(video_ws_router)
app.include_router(video_router)
app.include_router(auth_router)


@app.get("/health")
async def health():
    logging.info("Health check running...")

    health_status = {"status": "OK"}

    # Redis Check
    try:
        health_redis_client = redis.from_url(settings.REDIS_URL)
        redis_ping_status = await health_redis_client.ping()
        await health_redis_client.close()
        health_status["redis_check"] = redis_ping_status
        logger.info(f"Redis ping: {redis_ping_status}")
    except Exception as e:
        health_status["redis_check"] = f"Failed: {e}"
        logger.error(f"Redis health check failed: {e}", exc_info=True)

    # Database Check
    db_status = test_db_connection()
    health_status["sqlalchemy_check"] = db_status
    logger.info(f"Database check: {db_status}")

    # Celery Worker Check
    try:
        ping_response = celery_app.control.ping(timeout=1)
        health_status["celery_worker_ping_response"] = ping_response or "No response"
        logger.info(f"Celery worker ping: {ping_response}")
    except Exception as e:
        health_status["celery_worker_ping_response"] = f"Failed: {e}"
        logger.error(f"Celery ping failed: {e}", exc_info=True)

    return health_status


# ------------------------------
# Test
# ------------------------------
# backend/core/api.py
import base64
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
import cv2
from services.detection.model import run_gend_inference as gend_model_inference
from core.celery.detection_tasks import base64_to_image

@app.post("/gend-inference/")
async def gend_inference_api(task_id: str, file: UploadFile = File(...)):
    """
    API endpoint to run GenD inference on a single uploaded image.
    Inputs:
        - task_id: str
        - file: image file
    Returns:
        - JSON with inference result
    """
    # Read image bytes
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Convert PIL to base64 (optional, just for returning the image)
    img_np = np.array(img)
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    # Call the actual GenD model inference
    result = gend_model_inference(task_id, img)

    # Prepare response
    real_prob = result.get("real_prob", 0.5)
    fake_prob = result.get("fake_prob", 0.5)
    is_anomaly = fake_prob > 0.5
    confidence = fake_prob * 100 if is_anomaly else real_prob * 100

    detection_result = {
        "frame_index": 0,
        "is_anomaly": is_anomaly,
        "confidence": round(confidence, 2),
        "real_prob": round(real_prob, 4),
        "fake_prob": round(fake_prob, 4),
        "anomaly_type": "GenD Deepfake" if is_anomaly else None,
        "original_frame_data": img_b64,
        "task_id": task_id
    }

    return detection_result