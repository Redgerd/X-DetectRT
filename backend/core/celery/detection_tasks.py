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
        result = run_gend_inference(task_id, pil_image)  # call the real inference function
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
            "task_id": task_id
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
        return detection_result

    except Exception as e:
        logger.error(f"[GenD Inference] Error processing frame {frame_index}: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=5)


@shared_task(name="backend.core.celery.detection_tasks.run_gend_pipeline", bind=True)
def run_gend_pipeline(self, task_id: str, frame_results: dict) -> dict:
    """
    Wrapper task that runs GenD detection on all frames.
    """
    logger.info(f"[GenD Pipeline] Starting GenD detection pipeline for task_id: {task_id}")
    
    preview_frames = frame_results.get("preview_frames", [])
    if not preview_frames or all(f == "" for f in preview_frames):
        logger.warning(f"[GenD Pipeline] No preview frames found in frame_results")
        return {"error": "No frames to analyze", "task_id": task_id}
    
    results = []
    anomaly_count = 0
    total_frames = 0

    for i, frame_b64 in enumerate(preview_frames):
        if not frame_b64:
            continue
        total_frames += 1
        try:
            frame = base64_to_image(frame_b64)
            if frame is None:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            result = gend_model_inference(task_id, pil_image)
            real_prob = result.get("real_prob", 0.5)
            fake_prob = result.get("fake_prob", 0.5)
            is_anomaly = fake_prob > 0.5
            confidence = fake_prob * 100 if is_anomaly else real_prob * 100
            
            if is_anomaly:
                anomaly_count += 1

            frame_result = {
                "frame_index": i,
                "timestamp": f"00:00:{i // 30:02d}:{(i % 30) * 2:02d}",
                "is_anomaly": is_anomaly,
                "confidence": round(confidence, 2),
                "real_prob": round(real_prob, 4),
                "fake_prob": round(fake_prob, 4),
                "anomaly_type": "GenD Deepfake" if is_anomaly else None,
                "frame_data": frame_b64,
                "task_id": task_id
            }
            results.append(frame_result)

            # Publish to Redis
            try:
                redis_client.publish(
                    f"task_detection:{task_id}",
                    json.dumps({
                        "type": "detection_ready",
                        **frame_result
                    })
                )
            except Exception as redis_err:
                logger.warning(f"[GenD Pipeline] Redis publish error: {redis_err}")

        except Exception as e:
            logger.error(f"[GenD Pipeline] Error processing frame {i}: {e}")
            continue

    final_result = {
        "message": "GenD detection analysis complete",
        "task_id": task_id,
        "total_frames": total_frames,
        "anomaly_count": anomaly_count,
        "results": results,
        "anomaly_percentage": round((anomaly_count / total_frames * 100) if total_frames > 0 else 0, 2)
    }

    try:
        redis_client.set(f"detection_result:{task_id}", json.dumps(final_result))
    except Exception as e:
        logger.warning(f"[GenD Pipeline] Failed to store final result: {e}")

    # Trigger XAI asynchronously
    try:
        from core.celery.explainable_ai import run_explainable_ai
        run_explainable_ai.delay(task_id, final_result)
        logger.info(f"[GenD Pipeline] XAI task dispatched for task_id={task_id}")
    except Exception as xai_err:
        logger.warning(f"[GenD Pipeline] Failed to dispatch XAI task: {xai_err}")

    logger.info(f"[GenD Pipeline] Completed: {anomaly_count}/{total_frames} anomalies detected")
    return final_result