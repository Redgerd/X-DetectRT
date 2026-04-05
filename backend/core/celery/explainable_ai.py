# backend/core/celery/explainable_ai.py

"""
Celery task for Explainable AI (XAI) generation.

Implements:
    - Grad-CAM++ : Visual heatmaps (which facial regions look suspicious)
    - ELA         : Error Level Analysis (JPEG compression artefact map)
    - 2D FFT      : Fast Fourier Transform frequency-domain anomaly map

All three results are batched and published together per frame.

Reference: DYP.pdf – Step 3 "Integrating Dual XAI Techniques"
"""
import base64
import json
import logging
import numpy as np
import cv2
from typing import Dict, Any, List, Optional
from PIL import Image

from celery.exceptions import SoftTimeLimitExceeded
import redis

from celery import shared_task

from config import settings
from core.celery.celery_app import celery_app

# Import GenD model loader
from services.detection.model import load_gend_model, _GEND_DEVICE

# XAI services
from services.explaination.explaination import generate_gradcam, generate_ela, generate_fft, generate_lime


from models import XAIResult, ProcessedFrame
from core.database import SessionLocal

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
@shared_task(
    name="backend.core.celery.explainable_ai.run_explainable_ai",
    bind=True,
    max_retries=2,
    soft_time_limit=120,
    time_limit=150,
)
def run_explainable_ai(self, task_id: str, frame_results: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate Grad-CAM++, ELA, and 2D FFT explanations for each frame.

    All three visualisations are computed per frame and published together
    in a single Redis message so consumers receive a complete XAI bundle.

    Args:
        task_id: Pipeline task identifier.
        frame_results: Output from the GenD detection pipeline.
                       Expected to contain a ``results`` list, each item having:
                         - ``frame_data``       : base64 JPEG string of the frame
                         - ``is_anomaly``       : bool
                         - ``frame_index``      : int
                         - ``timestamp``        : str
                         - ``fake_prob``        : float
                         - ``real_prob``        : float
                         - ``confidence``       : float (optional)
                         - ``anomaly_type``     : str (optional)

    Returns:
        Dict with ``task_id``, ``xai_results`` list, and a ``message`` summary.
        Each entry in ``xai_results`` contains:
            - gradcam_b64  : Grad-CAM++ heatmap overlay (base64 JPEG)
            - ela_b64      : ELA heatmap overlay (base64 JPEG)
            - fft_data      : 2D FFT power spectrum as structured graph data (dict)
    """
    logger.info(f"[XAI] Starting XAI generation for task_id={task_id}")

    try:
        # -----------------------------------------------------------------
        # Load GenD model (cached globally after first load)
        # -----------------------------------------------------------------
        model = load_gend_model()
        device = _GEND_DEVICE or "cpu"

        # Grad-CAM requires gradients
        model.eval()
        for param in model.parameters():
            param.requires_grad_(True)

        # Normalise input shape
        if "results" in frame_results:
            detection_results = frame_results.get("results", [])
        elif "frame_data" in frame_results:
            detection_results = [frame_results]
        else:
            detection_results = []

        if not detection_results:
            logger.warning(f"[XAI] No detection results found for task_id={task_id}")
            return {
                "task_id": task_id,
                "message": "No frames to explain – detection results were empty.",
                "xai_results": [],
            }

        xai_results: List[Dict[str, Any]] = []

        # -----------------------------------------------------------------
        # Per-frame XAI: Grad-CAM++ → ELA → 2D FFT
        # -----------------------------------------------------------------
        for frame_result in detection_results:
            frame_index  = frame_result.get("frame_index", -1)
            frame_b64    = frame_result.get("frame_data") or frame_result.get("original_frame_data", "")
            is_anomaly   = frame_result.get("is_anomaly", False)
            fake_prob    = frame_result.get("fake_prob", 0.0)
            real_prob    = frame_result.get("real_prob", 0.0)
            timestamp    = frame_result.get("timestamp", "")
            confidence   = frame_result.get("confidence", max(fake_prob, real_prob) * 100)
            anomaly_type = frame_result.get("anomaly_type")

            if not frame_b64:
                logger.warning(f"[XAI] Skipping frame {frame_index}: no frame data.")
                continue

            try:
                import torch

                pil_image = _base64_to_pil(frame_b64)
                tensor = model.feature_extractor.preprocess(pil_image).unsqueeze(0).to(device)

                # ── 1. explainable_pipeline ────────────────────────────────────────
                logger.info(f"[XAI] Generating Grad-CAM++ for frame {frame_index}")
                gradcam_b64 = generate_gradcam(model, tensor, pil_image)

                # ── 2. Error Level Analysis ───────────────────────────────
                logger.info(f"[XAI] Generating ELA for frame {frame_index}")
                ela_b64 = generate_ela(pil_image)

                # ── 3. 2D FFT ─────────────────────────────────────────────
                logger.info(f"[XAI] Generating 2D FFT for frame {frame_index}")
                fft_data = generate_fft(pil_image)

                # ── 4. LIME superpixel attribution ────────────────────────
                logger.info(f"[XAI] Generating LIME for frame {frame_index}")
                lime_data = generate_lime(model, pil_image, device=device)

                # ── 5. LLM narrative explanation ──────────────────────────
                logger.info(f"[XAI] Dispatching LLM analysis for frame {frame_index}")
                from core.celery.llm import run_llm
                llm_result = run_llm(
                    task_id=task_id,
                    frame_data={
                        "frame_index": frame_index,
                        "frame_data":  frame_b64,   # original frame
                        "gradcam_b64": gradcam_b64,
                        "ela_b64":     ela_b64,
                        "fake_prob":   fake_prob,
                        "real_prob":   real_prob,
                    },
                )
                llm_analysis = llm_result.get("analysis", "") if isinstance(llm_result, dict) else ""

                # ── Bundle all four and store ──────────────────────────────
                xai_entry = {
                    "frame_index":  frame_index,
                    "timestamp":    timestamp,
                    "is_anomaly":   is_anomaly,
                    "fake_prob":    fake_prob,
                    "real_prob":    real_prob,
                    "confidence":   confidence,
                    "anomaly_type": anomaly_type,
                    # XAI outputs
                    "gradcam_b64":  gradcam_b64,
                    "ela_b64":      ela_b64,
                    "fft_data":     fft_data,
                    "lime_data":    lime_data,
                    # LLM narrative explanation
                    "llm_analysis":  llm_analysis,
                }
                xai_results.append(xai_entry)

                # Publish the complete bundle to Redis in real time
                try:
                    redis_client.publish(
                        f"task_xai:{task_id}",
                        json.dumps({
                            "type":         "xai_ready",
                            "task_id":      task_id,
                            "frame_index":  frame_index,
                            "timestamp":    timestamp,
                            "is_anomaly":   is_anomaly,
                            "fake_prob":    fake_prob,
                            "real_prob":    real_prob,
                            "confidence":   confidence,
                            "anomaly_type": anomaly_type,
                            # All XAI outputs sent together
                            "gradcam_b64":  gradcam_b64,
                            "ela_b64":      ela_b64,
                            "fft_data":     fft_data,
                            "lime_data":    lime_data,
                            "llm_analysis": llm_analysis,
                        })
                    )
                except Exception as redis_err:
                    logger.warning(f"[XAI] Redis publish error for frame {frame_index}: {redis_err}")
                
                if user_id:
                    # Insert XAI result into database
                    db = SessionLocal()
                    try:
                        frame = db.query(ProcessedFrame).filter_by(task_id=task_id, frame_index=frame_index).first()
                        if frame:
                            xai_db = XAIResult(
                                frame_id=frame.id,
                                gradcam_b64=gradcam_b64,
                                ela_b64=ela_b64,
                                fft_data=json.dumps(fft_data) if fft_data else None,
                                lime_data=json.dumps(lime_data) if lime_data else None
                            )
                            db.add(xai_db)
                            db.commit()
                    except Exception as e:
                        logger.warning(f"[XAI] Failed to insert XAI for frame {frame_index}: {e}")
                        db.rollback()
                    finally:
                        db.close()

                logger.info(
                                    f"[XAI] Frame {frame_index} complete: "
                                    f"Grad-CAM + ELA + FFT + LIME generated (anomaly={is_anomaly})"
                                )

            except SoftTimeLimitExceeded:
                logger.error(f"[XAI] Soft time limit exceeded at frame {frame_index}")
                break
            except Exception as frame_err:
                logger.error(
                    f"[XAI] Failed to generate XAI for frame {frame_index}: {frame_err}",
                    exc_info=True,
                )
                xai_results.append({
                    "frame_index":  frame_index,
                    "timestamp":    timestamp,
                    "is_anomaly":   is_anomaly,
                    "fake_prob":    fake_prob,
                    "real_prob":    real_prob,
                    "confidence":   confidence,
                    "anomaly_type": anomaly_type,
                    "gradcam_b64":  None,
                    "ela_b64":      None,
                    "fft_data":     None,
                    "lime_data":    None,
                    "error":        str(frame_err),
                    "message":      "XAI generation failed for this frame",
                })

                if user_id:
                    # Insert failed XAI result into database
                    db = SessionLocal()
                    try:
                        frame = db.query(ProcessedFrame).filter_by(task_id=task_id, frame_index=frame_index).first()
                        if frame:
                            xai_db = XAIResult(
                                frame_id=frame.id,
                                error=str(frame_err)
                            )
                            db.add(xai_db)
                            db.commit()
                    except Exception as e:
                        logger.warning(f"[XAI] Failed to insert failed XAI for frame {frame_index}: {e}")
                        db.rollback()
                    finally:
                        db.close()

                continue

        # -----------------------------------------------------------------
        # Aggregate and persist final result
        # -----------------------------------------------------------------
        total_explained = sum(
            1 for r in xai_results
            if r.get("gradcam_b64") and r.get("ela_b64")
            and r.get("fft_data") is not None and r.get("lime_data") is not None
        )
        total_failed = sum(1 for r in xai_results if "error" in r)

        final_result = {
            "task_id":               task_id,
            "message": (
                f"XAI generation complete: {total_explained} frames fully explained "
                f"(Grad-CAM + ELA + FFT + LIME), {total_failed} failed"
            ),
            "total_frames":           len(detection_results),
            "total_frames_explained": total_explained,
            "total_frames_failed":    total_failed,
            "xai_results":            xai_results,
        }

        try:
            redis_client.setex(f"xai_result:{task_id}", 3600, json.dumps(final_result))
        except Exception as e:
            logger.warning(f"[XAI] Failed to store final XAI result in Redis: {e}")

        logger.info(
            f"[XAI] Completed: {total_explained}/{len(detection_results)} frames "
            f"fully explained for task_id={task_id}"
        )

        return final_result

    except SoftTimeLimitExceeded:
        logger.error(f"[XAI] Task timed out for task_id={task_id}")
        timeout_result = {
            "task_id": task_id,
            "message": "XAI generation timed out",
            "error":   "Soft time limit exceeded",
            "xai_results": [],
        }
        try:
            redis_client.setex(f"xai_result:{task_id}", 3600, json.dumps(timeout_result))
        except Exception:
            pass
        raise

    except Exception as e:
        logger.error(f"[XAI] Unexpected error for task_id={task_id}: {e}", exc_info=True)
        error_result = {
            "task_id": task_id,
            "message": "XAI generation failed",
            "error":   str(e),
            "xai_results": [],
        }
        try:
            redis_client.setex(f"xai_result:{task_id}", 3600, json.dumps(error_result))
        except Exception:
            pass
        raise self.retry(exc=e, countdown=10)