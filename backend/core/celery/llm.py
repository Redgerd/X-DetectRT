# backend/core/celery/spatialDetection.py
import cv2
import numpy as np
import base64
import json
import logging
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from typing import List, Dict, Any, Optional
import redis

from config import settings
from .celery_app import celery_app

logger = logging.getLogger(__name__)

# Redis client for publishing detection results
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    encoded_data = base64_str.split(',')[1] if ',' in base64_str else base64_str
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def image_to_base64(img: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buffer).decode("utf-8")

@celery_app.task(name="llm.run_llm")
def run_llm(task_id: str, frame_results: Dict[str, Any]) -> Dict[str, Any]:
    result = {

    }
    return result