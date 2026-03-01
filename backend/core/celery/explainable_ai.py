# backend/core/celery/explainable_ai.py
"""
Celery task for Explainable AI (XAI) generation.

Implements:
    - Grad-CAM++ : Visual heatmaps (which facial regions look suspicious)
    - LIME        : Local superpixel attribution (why the model classified it fake)

Reference: DYP.pdf – Step 3 "Integrating Dual XAI Techniques"
"""
import base64
import json
import logging
import numpy as np
import cv2
from typing import Dict, Any
from PIL import Image

from celery.exceptions import SoftTimeLimitExceeded
import redis

from config import settings
from .celery_app import celery_app

# Import GenD model loader and XAI helpers
from services.detection.model import load_gend_model, _GEND_DEVICE
from services.detection.xai_methods import generate_gradcam, generate_lime

logger = logging.getLogger(__name__)

# Redis client for publishing results
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base64_to_pil(base64_str: str) -> Image.Image:
    """Decode a base64 image string to a PIL Image."""
    encoded = base64_str.split(",")[1] if "," in base64_str else base64_str
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# Celery Task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="explainable_ai.run_explainable_ai",
    bind=True,
    max_retries=2,
    soft_time_limit=120,
    time_limit=150,
)
def run_explainable_ai(self, task_id: str, frame_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate Grad-CAM++ and LIME explanations for frames flagged as anomalies.

    Args:
        task_id: Pipeline task identifier.
        frame_results: Output from the GenD detection pipeline.
                       Expected to contain a ``results`` list, each item having:
                         - ``frame_data``: base64 JPEG string of the frame
                         - ``is_anomaly``: bool
                         - ``frame_index``: int
                         - ``timestamp``: str
                         - ``fake_prob``: float
                         - ``real_prob``: float

    Returns:
        Dict with ``task_id``, ``xai_results`` list, and a ``message`` summary.
    """
    logger.info(f"[XAI] Starting XAI generation for task_id={task_id}")

    try:
        # -----------------------------------------------------------------
        # Load GenD model (already cached globally after first load)
        # -----------------------------------------------------------------
        model = load_gend_model()
        device = _GEND_DEVICE or "cpu"

        # Enable gradients only where needed (Grad-CAM++ requires them)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(True)

        detection_results = frame_results.get("results", [])

        if not detection_results:
            logger.warning(f"[XAI] No detection results found in frame_results for task_id={task_id}")
            return {
                "task_id": task_id,
                "message": "No frames to explain – detection results were empty.",
                "xai_results": [],
            }

        xai_results = []

        for frame_result in detection_results:
            frame_index = frame_result.get("frame_index", -1)
            frame_b64 = frame_result.get("frame_data", "")
            is_anomaly = frame_result.get("is_anomaly", False)
            fake_prob = frame_result.get("fake_prob", 0.0)
            real_prob = frame_result.get("real_prob", 0.0)
            timestamp = frame_result.get("timestamp", "")

            if not frame_b64:
                logger.warning(f"[XAI] Skipping frame {frame_index}: no frame data.")
                continue

            try:
                # Decode frame
                pil_image = _base64_to_pil(frame_b64)

                # Preprocess for model (returns a (C, H, W) tensor)
                import torch
                tensor = model.feature_extractor.preprocess(pil_image).unsqueeze(0).to(device)

                # ----------------------------------------------------------
                # 1. Grad-CAM++ heatmap
                # ----------------------------------------------------------
                gradcam_b64 = generate_gradcam(model, tensor, pil_image)

                # ----------------------------------------------------------
                # 2. LIME local explanation
                # ----------------------------------------------------------
                lime_b64 = generate_lime(model, pil_image, device)

                xai_entry = {
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "is_anomaly": is_anomaly,
                    "fake_prob": fake_prob,
                    "real_prob": real_prob,
                    "gradcam_b64": gradcam_b64,
                    "lime_b64": lime_b64,
                }
                xai_results.append(xai_entry)

                # ----------------------------------------------------------
                # Publish individual XAI result to Redis in real time
                # ----------------------------------------------------------
                try:
                    redis_client.publish(
                        f"task_xai:{task_id}",
                        json.dumps({
                            "type": "xai_ready",
                            "task_id": task_id,
                            "frame_index": frame_index,
                            "timestamp": timestamp,
                            "is_anomaly": is_anomaly,
                            "fake_prob": fake_prob,
                            "real_prob": real_prob,
                            "gradcam_b64": gradcam_b64,
                            "lime_b64": lime_b64,
                        })
                    )
                except Exception as redis_err:
                    logger.warning(f"[XAI] Redis publish error for frame {frame_index}: {redis_err}")

                logger.info(f"[XAI] Generated explanations for frame {frame_index} (anomaly={is_anomaly})")

            except SoftTimeLimitExceeded:
                logger.error(f"[XAI] Soft time limit exceeded at frame {frame_index}")
                break
            except Exception as frame_err:
                logger.error(f"[XAI] Failed to generate XAI for frame {frame_index}: {frame_err}", exc_info=True)
                continue

        # -----------------------------------------------------------------
        # Build and store final aggregated result
        # -----------------------------------------------------------------
        final_result = {
            "task_id": task_id,
            "message": "XAI generation complete",
            "total_frames_explained": len(xai_results),
            "xai_results": xai_results,
        }

        try:
            redis_client.set(f"xai_result:{task_id}", json.dumps(final_result))
        except Exception as e:
            logger.warning(f"[XAI] Failed to store final XAI result in Redis: {e}")

        logger.info(f"[XAI] Completed: {len(xai_results)} frames explained for task_id={task_id}")
        return final_result

    except SoftTimeLimitExceeded:
        logger.error(f"[XAI] Task timed out for task_id={task_id}")
        raise
    except Exception as e:
        logger.error(f"[XAI] Unexpected error for task_id={task_id}: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=10)