# backend/core/celery/audio_xai.py
"""
READ – Celery Task: Audio XAI Generation

Runs both Integrated Gradients and SHAP explanations on the audio that was
already classified by run_audio_pipeline, then publishes the heatmap PNGs
to Redis for real-time delivery to the frontend.
"""

import base64
import json
import logging

import redis
from celery.exceptions import SoftTimeLimitExceeded

from config import settings
from .celery_app import celery_app

logger = logging.getLogger(__name__)

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


def _publish(channel: str, payload: dict) -> None:
    try:
        redis_client.publish(channel, json.dumps(payload))
    except Exception as e:
        logger.warning(f"[AudioXAI] Redis publish error on {channel}: {e}")


def _store(key: str, payload: dict) -> None:
    try:
        redis_client.set(key, json.dumps(payload))
    except Exception as e:
        logger.warning(f"[AudioXAI] Redis store error for {key}: {e}")


@celery_app.task(
    name="audio_xai.run_audio_xai",
    bind=True,
    max_retries=1,
    soft_time_limit=180,
    time_limit=210,
)
def run_audio_xai(
    self,
    task_id: str,
    audio_b64: str,
    fake_prob: float,
    real_prob: float,
) -> dict:
    """
    Generate Integrated Gradients and SHAP temporal heatmaps for an
    audio clip that was already classified by run_audio_pipeline.

    Args:
        task_id:   Pipeline identifier.
        audio_b64: Base64-encoded raw audio file (same as sent to detection task).
        fake_prob: Fake probability from the detection step (for context).
        real_prob: Real probability from the detection step (for context).

    Returns:
        Dict containing base64 PNG strings for both heatmaps.
    """
    logger.info(f"[AudioXAI] Starting XAI generation for task_id={task_id}")

    try:
        from services.audio.audio_utils import load_and_preprocess_audio
        from services.audio.model import load_audio_models, _AUDIO_DEVICE
        from services.audio.xai_audio import generate_audio_xai

        # Decode audio
        encoded = audio_b64.split(",", 1)[1] if "," in audio_b64 else audio_b64
        audio_bytes = base64.b64decode(encoded)
        audio_tensor = load_and_preprocess_audio(audio_bytes)

        # Ensure models are loaded (idempotent call)
        wavlm, _, detector = load_audio_models()
        device = _AUDIO_DEVICE or "cpu"

        # Enable gradients for the detector (required by IG)
        detector.train(False)
        for param in detector.parameters():
            param.requires_grad_(True)

        # Run both XAI methods
        xai_results = generate_audio_xai(audio_tensor, detector, wavlm, device)

        ig_b64   = xai_results.get("ig_heatmap_b64")
        shap_b64 = xai_results.get("shap_heatmap_b64")

        payload = {
            "type":           "audio_xai_ready",
            "task_id":        task_id,
            "fake_prob":      round(fake_prob, 4),
            "real_prob":      round(real_prob, 4),
            "ig_heatmap_b64":   ig_b64,
            "shap_heatmap_b64": shap_b64,
            "message":        "Audio XAI generation complete",
        }

        # Publish real-time update
        _publish(f"task_audio_xai:{task_id}", payload)

        # Persist for REST polling
        _store(f"audio_xai_result:{task_id}", payload)

        # Notify detection channel that XAI is done too
        _publish(
            f"task_audio_detection:{task_id}",
            {"type": "audio_xai_ready", "task_id": task_id},
        )

        logger.info(f"[AudioXAI] Completed for task_id={task_id} "
                    f"(IG={'OK' if ig_b64 else 'FAIL'}, "
                    f"SHAP={'OK' if shap_b64 else 'FAIL'})")
        return payload

    except SoftTimeLimitExceeded:
        logger.error(f"[AudioXAI] Soft time limit exceeded for task_id={task_id}")
        raise

    except Exception as exc:
        logger.error(f"[AudioXAI] Error for task_id={task_id}: {exc}", exc_info=True)
        raise self.retry(exc=exc, countdown=15)
