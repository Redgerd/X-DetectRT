# backend/core/celery/explainable_ai.py
"""
Celery task for Explainable AI (XAI) generation.

Runs all 7 XAI techniques per anomalous frame via run_xai():
    - SHAP TimeShap         (temporal frame contributions)
    - LIME Superpixels      (local superpixel attribution)
    - Integrated Gradients  (anatomical zone attribution)
    - SAM Attribution       (auto-segmented face parts)
    - Counterfactual        (minimum perturbation to flip FAKE→REAL)
    - TCAV                  (concept activation vector sensitivity)
    - Prototype Analysis    (cosine similarity to known deepfakes)
"""

import base64
import json
import logging
import numpy as np
import cv2
from typing import Dict, Any
from PIL import Image
from celery import shared_task

from celery.exceptions import SoftTimeLimitExceeded
import redis

from config import settings
from .celery_app import celery_app

from services.explaination.pipeline import run_xai

logger = logging.getLogger(__name__)

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _base64_to_pil(base64_str: str) -> Image.Image:
    """Decode a base64 image string to a PIL Image."""
    encoded = base64_str.split(",")[1] if "," in base64_str else base64_str
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _pil_to_base64(pil_image: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL Image to a base64 string."""
    import io
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Celery Task
# ─────────────────────────────────────────────────────────────────────────────

@shared_task(
    name="backend.core.celery.explainable_ai.run_explainable_ai",
    bind=True,
    max_retries=2,
    soft_time_limit=120,
    time_limit=150,
)
def run_explainable_ai(self, task_id: str, frame_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all 7 XAI techniques on each anomalous frame.

    Args:
        task_id      : Pipeline task identifier.
        frame_results: Output from the GenD detection pipeline.
                       Expected to contain a ``results`` list, each item having:
                         - ``frame_data``  : base64 JPEG string of the frame
                         - ``is_anomaly``  : bool
                         - ``frame_index`` : int
                         - ``timestamp``   : str
                         - ``fake_prob``   : float
                         - ``real_prob``   : float

    Returns:
        Dict with ``task_id``, ``xai_results`` list, and a ``message`` summary.
        Each entry in ``xai_results`` contains per-technique dicts, each with:
            - ``technique``      : str
            - ``scores``         : dict | None
            - ``figure_base64``  : str  | None  (base64 PNG chart)
            - ``error``          : str  | None
            - ``elapsed_seconds``: float
    """
    logger.info(f"[XAI] Starting XAI generation for task_id={task_id}")

    try:
        detection_results = frame_results.get("results", [])

        if not detection_results:
            logger.warning(f"[XAI] No detection results for task_id={task_id}")
            return {
                "task_id": task_id,
                "message": "No frames to explain – detection results were empty.",
                "xai_results": [],
            }

        # Build a parallel list of fake_probs across all frames for SHAP TimeShap
        # (gives temporal context even when explaining a single frame)
        all_frame_probs = [r.get("fake_prob", 0.0) for r in detection_results]

        xai_results = []

        for frame_result in detection_results:
            frame_index = frame_result.get("frame_index", -1)
            frame_b64   = frame_result.get("frame_data", "")
            is_anomaly  = frame_result.get("is_anomaly", False)
            fake_prob   = frame_result.get("fake_prob", 0.0)
            real_prob   = frame_result.get("real_prob", 0.0)
            timestamp   = frame_result.get("timestamp", "")

            if not frame_b64:
                logger.warning(f"[XAI] Skipping frame {frame_index}: no frame data.")
                continue

            try:
                # ── Run all 7 XAI techniques ──────────────────────────────
                technique_results = run_xai(
                    media_type="image",
                    b64_image=frame_b64,
                    frame_probs=all_frame_probs,   # temporal context for SHAP
                    frame_tensors=None,            # tensors not stored; probs suffice
                )

                xai_entry = {
                    "frame_index": frame_index,
                    "timestamp":   timestamp,
                    "is_anomaly":  is_anomaly,
                    "fake_prob":   fake_prob,
                    "real_prob":   real_prob,
                    "techniques":  technique_results,  # List[dict] — one per technique
                }
                xai_results.append(xai_entry)

                # ── Publish to Redis in real time ─────────────────────────
                try:
                    redis_client.publish(
                        f"task_xai:{task_id}",
                        json.dumps({
                            "type":        "xai_ready",
                            "task_id":     task_id,
                            "frame_index": frame_index,
                            "timestamp":   timestamp,
                            "is_anomaly":  is_anomaly,
                            "fake_prob":   fake_prob,
                            "real_prob":   real_prob,
                            "techniques":  technique_results,
                        })
                    )
                except Exception as redis_err:
                    logger.warning(
                        f"[XAI] Redis publish error for frame {frame_index}: {redis_err}"
                    )

                logger.info(
                    f"[XAI] Frame {frame_index} — "
                    f"{len(technique_results)} techniques completed "
                    f"(anomaly={is_anomaly})"
                )

            except SoftTimeLimitExceeded:
                logger.error(f"[XAI] Soft time limit exceeded at frame {frame_index}")
                break
            except Exception as frame_err:
                logger.error(
                    f"[XAI] Failed for frame {frame_index}: {frame_err}", exc_info=True
                )
                continue

        # ── Store aggregated result in Redis ──────────────────────────────
        final_result = {
            "task_id":               task_id,
            "message":               "XAI generation complete",
            "total_frames_explained": len(xai_results),
            "xai_results":           xai_results,
        }

        try:
            redis_client.set(f"xai_result:{task_id}", json.dumps(final_result))
        except Exception as e:
            logger.warning(f"[XAI] Failed to store final XAI result in Redis: {e}")

        logger.info(
            f"[XAI] Completed: {len(xai_results)} frames explained for task_id={task_id}"
        )
        return final_result

    except SoftTimeLimitExceeded:
        logger.error(f"[XAI] Task timed out for task_id={task_id}")
        raise
    except Exception as e:
        logger.error(f"[XAI] Unexpected error for task_id={task_id}: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=10)