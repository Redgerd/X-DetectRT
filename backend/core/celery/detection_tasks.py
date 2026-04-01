# backend\core\celery\detection_tasks.py

import os
import logging
import base64
import json
import numpy as np
from PIL import Image
from celery import shared_task
from celery.signals import worker_process_init

# Redis
import redis
from config import settings

# OpenCV import
import cv2

# Import the GenD inference function
from services.detection.model import run_gend_inference as gend_model_inference

# Import the XAI task 
from core.celery.explainable_ai import run_explainable_ai


# Redis client
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

# Logger
logger = logging.getLogger(__name__)

# ============================================================
# Helper Functions
# ============================================================

def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    encoded_data = base64_str.split(',')[1] if ',' in base64_str else base64_str
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def image_to_base64(img: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    import cv2
    _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buffer).decode("utf-8")

# ============================================================
# Celery Tasks
# ============================================================

@shared_task(name="backend.core.celery.detection_tasks.run_gend_inference", bind=True, max_retries=3)
def run_gend_inference(self, task_id: str, frame_data: str, frame_index: int = 0, timestamp: str = "") -> dict:
    """
    Celery task for running GenD inference on a single frame.
    """
    logger.info(f"[GenD Inference] Starting inference for task_id: {task_id}, frame_index: {frame_index}")
    try:
        # Convert base64 to OpenCV image
        frame = base64_to_image(frame_data)
        if frame is None:
            raise ValueError("Failed to decode frame data")
        
        # Convert to PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # --- Run the actual model inference ---
        result = gend_model_inference(task_id, pil_image)  # call the real inference function
        real_prob = result.get("real_prob", 0.5)
        fake_prob = result.get("fake_prob", 0.5)
        is_anomaly = fake_prob > 0.5
        confidence = fake_prob * 100 if is_anomaly else real_prob * 100
        
        detection_result = {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "is_anomaly": is_anomaly,
            "confidence": round(confidence, 2),
            "real_prob": round(real_prob, 4),
            "fake_prob": round(fake_prob, 4),
            "anomaly_type": "GenD Deepfake" if is_anomaly else None,
            "original_frame_data": frame_data,
            "task_id": task_id,
            "frame_data": frame_data
        }

        # Publish to Redis
        try:
            redis_client.publish(
                f"task_detection:{task_id}",
                json.dumps({
                    "type": "detection_ready",
                    **detection_result
                })
            )
        except Exception as redis_err:
            logger.warning(f"[GenD Inference] Redis publish error: {redis_err}")

        logger.info(f"[GenD Inference] Completed for frame {frame_index}: fake_prob={fake_prob:.4f}, is_anomaly={is_anomaly}")
        
        # ── Dispatch XAI task only if anomaly detected ─────────────────────────
        if is_anomaly:
            try:
                frame_results = {
                    "results": [detection_result]
                }
                run_explainable_ai.delay(task_id, {"results": [detection_result]})
                logger.info(f"[XAI] XAI task dispatched for task_id={task_id}, frame_index={frame_index}")
            except Exception as xai_err:
                logger.warning(f"[GenD Inference] Failed to dispatch XAI task: {xai_err}")
        else:
            logger.info(f"[XAI] Skipping XAI task for frame {frame_index} - no anomaly detected")
        
        return detection_result

    except Exception as e:
        logger.error(f"[GenD Inference] Error processing frame {frame_index}: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=5)