import os
import logging
import base64
import json
import numpy as np
from PIL import Image
from celery import shared_task
from celery.signals import worker_process_init
from tensorflow.keras.models import load_model

# Import the GenD inference function from detection service
from services.detection.model import run_gend_inference as gend_inference

# Import Redis for publishing results
import redis
from config import settings

# Global variable for model
xception_model = None

# Path to model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "ml_models", "XceptionNet.keras"))

# Redis client
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

# Logger
logger = logging.getLogger(__name__)

# Used to test if the model is loaded properly
def confirm_model_loaded():
    global xception_model
    return xception_model is not None

# Placeholder for detection task
@shared_task(name="backend.core.celery.detection_tasks.perform_detection")
def perform_detection(input_data):
    global xception_model
    if xception_model is None:
        raise RuntimeError("Model not loaded")
    
    # Add actual prediction logic here using input_data
    # prediction = xception_model.predict(...)
    return {"status": "success", "message": "Model is loaded and ready"}


# ============================================================
# GenD Inference Celery Task
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


@shared_task(name="backend.core.celery.detection_tasks.run_gend_inference", bind=True, max_retries=3)
def run_gend_inference(self, task_id: str, frame_data: str, frame_index: int = 0, timestamp: str = "") -> dict:
    """
    Celery task for running GenD inference on a single frame.
    
    Args:
        task_id: The task identifier
        frame_data: Base64 encoded frame image
        frame_index: Frame index in the video
        timestamp: Frame timestamp
    
    Returns:
        Dict containing detection results
    """
    import cv2
    logger.info(f"[GenD Inference] Starting inference for task_id: {task_id}, frame_index: {frame_index}")
    
    try:
        # Convert base64 to image
        frame = base64_to_image(frame_data)
        if frame is None:
            raise ValueError("Failed to decode frame data")
        
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Run GenD inference
        result = gend_inference(task_id, pil_image)
        
        real_prob = result.get("real_prob", 0.5)
        fake_prob = result.get("fake_prob", 0.5)
        
        # Determine if it's an anomaly based on fake probability
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
        
        # Publish detection result to Redis for real-time updates
        try:
            redis_client.publish(
                f"task_detection:{task_id}",
                json.dumps({
                    "type": "detection_ready",
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "frame_data": frame_data,
                    "is_anomaly": detection_result["is_anomaly"],
                    "confidence": detection_result["confidence"],
                    "anomaly_type": detection_result["anomaly_type"],
                    "real_prob": detection_result["real_prob"],
                    "fake_prob": detection_result["fake_prob"],
                    "task_id": task_id
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
    Wrapper task that chains after frame_selection and runs GenD detection.
    This is the replacement for run_chained_detection in spatialDetection.
    
    Args:
        task_id: The task identifier
        frame_results: The result dict from extract_faces_with_optical_flow
    
    Returns:
        Dict containing detection results
    """
    import cv2
    logger.info(f"[GenD Pipeline] Starting GenD detection pipeline for task_id: {task_id}")
    
    # Extract preview frames from frame_selection result
    preview_frames = frame_results.get("preview_frames", [])
    
    if not preview_frames or all(f == "" for f in preview_frames):
        logger.warning(f"[GenD Pipeline] No preview frames found in frame_results")
        return {"error": "No frames to analyze", "task_id": task_id}
    
    results = []
    anomaly_count = 0
    total_frames = 0
    
    # Process each preview frame
    for i, frame_b64 in enumerate(preview_frames):
        if not frame_b64:
            continue
            
        total_frames += 1
        
        try:
            # Run inference synchronously (we could also use .delay() for async)
            frame = base64_to_image(frame_b64)
            if frame is None:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            result = gend_inference(task_id, pil_image)
            
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
            
            # Publish detection result to Redis
            try:
                redis_client.publish(
                    f"task_detection:{task_id}",
                    json.dumps({
                        "type": "detection_ready",
                        "frame_index": i,
                        "timestamp": frame_result["timestamp"],
                        "frame_data": frame_b64,
                        "is_anomaly": is_anomaly,
                        "confidence": confidence,
                        "anomaly_type": frame_result["anomaly_type"],
                        "real_prob": real_prob,
                        "fake_prob": fake_prob,
                        "task_id": task_id
                    })
                )
            except Exception as redis_err:
                logger.warning(f"[GenD Pipeline] Redis publish error: {redis_err}")
                
        except Exception as e:
            logger.error(f"[GenD Pipeline] Error processing frame {i}: {e}")
            continue
    
    # Final result
    final_result = {
        "message": "GenD detection analysis complete",
        "task_id": task_id,
        "total_frames": total_frames,
        "anomaly_count": anomaly_count,
        "results": results,
        "anomaly_percentage": round((anomaly_count / total_frames * 100) if total_frames > 0 else 0, 2)
    }
    
    # Store final result in Redis
    try:
        redis_client.set(f"detection_result:{task_id}", json.dumps(final_result))
    except Exception as e:
        logger.warning(f"[GenD Pipeline] Failed to store final result: {e}")
    
    logger.info(f"[GenD Pipeline] Completed: {anomaly_count}/{total_frames} anomalies detected")
    
    return final_result

