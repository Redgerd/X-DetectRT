# backend/core/celery/audio_detection_tasks.py
"""
READ – Celery Tasks: Audio Deepfake Detection Pipeline

Task flow:
    run_audio_pipeline
        ├─ Preprocess audio  (audio_utils)
        ├─ WavLM + Classifier inference  (services.audio.model)
        ├─ Publish verdict to Redis  (task_audio_detection:{task_id})
        ├─ Store result     (audio_detection_result:{task_id})
        └─ Dispatch run_audio_xai.delay(...)
"""

import base64
import json
import logging

import redis
from celery.exceptions import SoftTimeLimitExceeded

from config import settings
from .celery_app import celery_app

logger = logging.getLogger(__name__)

# Redis client (synchronous, used inside Celery workers)
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_audio_b64(audio_b64: str) -> bytes:
    """Strip optional data-URI prefix and base64-decode to raw bytes."""
    if "," in audio_b64:
        audio_b64 = audio_b64.split(",", 1)[1]
    return base64.b64decode(audio_b64)


def _publish(channel: str, payload: dict) -> None:
    """Safely publish a JSON payload to a Redis channel."""
    try:
        redis_client.publish(channel, json.dumps(payload))
    except Exception as e:
        logger.warning(f"[AudioTasks] Redis publish error on {channel}: {e}")


def _store(key: str, payload: dict) -> None:
    """Safely store a JSON payload in Redis."""
    try:
        redis_client.set(key, json.dumps(payload))
    except Exception as e:
        logger.warning(f"[AudioTasks] Redis store error for {key}: {e}")


# ---------------------------------------------------------------------------
# Main detection task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="audio_detection.run_audio_pipeline",
    bind=True,
    max_retries=2,
    soft_time_limit=180,
    time_limit=210,
)
def run_audio_pipeline(self, task_id: str, audio_b64: str) -> dict:
    """
    End-to-end audio deepfake detection pipeline.

    Args:
        task_id:   Unique pipeline identifier.
        audio_b64: Base64-encoded audio file (any format librosa supports).
                   May optionally include a data-URI prefix.

    Returns:
        dict with verdict, probabilities, and status metadata.
    """
    logger.info(f"[AudioPipeline] Starting for task_id={task_id}")

    try:
        # ------------------------------------------------------------------
        # 1. Decode & preprocess
        # ------------------------------------------------------------------
        from services.audio.audio_utils import load_and_preprocess_audio, waveform_to_base64_png
        from services.audio.model import run_audio_inference

        audio_bytes = _decode_audio_b64(audio_b64)
        audio_tensor = load_and_preprocess_audio(audio_bytes)
        logger.info(f"[AudioPipeline] Audio preprocessed: {audio_tensor.shape}")

        # Waveform PNG for display in frontend (generated once here)
        waveform_b64 = waveform_to_base64_png(audio_tensor)

        # ------------------------------------------------------------------
        # 2. WavLM + Classifier inference
        # ------------------------------------------------------------------
        result = run_audio_inference(task_id, audio_tensor)

        fake_prob = result["fake_prob"]
        real_prob = result["real_prob"]
        is_fake   = fake_prob > 0.5
        confidence = fake_prob * 100 if is_fake else real_prob * 100

        verdict = "FAKE" if is_fake else "REAL"
        logger.info(
            f"[AudioPipeline] Verdict={verdict} (fake_prob={fake_prob:.4f}) "
            f"for task_id={task_id}"
        )

        # ------------------------------------------------------------------
        # 3. Build detection payload
        # ------------------------------------------------------------------
        detection_payload = {
            "type":        "audio_detection_ready",
            "task_id":     task_id,
            "verdict":     verdict,
            "is_fake":     is_fake,
            "confidence":  round(confidence, 2),
            "fake_prob":   round(fake_prob, 4),
            "real_prob":   round(real_prob, 4),
            "waveform_b64": waveform_b64,
            "message":     "Audio deepfake detection complete",
        }

        # ------------------------------------------------------------------
        # 4. Publish real-time update to Redis
        # ------------------------------------------------------------------
        _publish(f"task_audio_detection:{task_id}", detection_payload)

        # ------------------------------------------------------------------
        # 5. Persist result for REST polling
        # ------------------------------------------------------------------
        _store(f"audio_detection_result:{task_id}", detection_payload)

        # ------------------------------------------------------------------
        # 6. Dispatch XAI task asynchronously
        # ------------------------------------------------------------------
        try:
            from core.celery.audio_xai import run_audio_xai
            run_audio_xai.delay(task_id, audio_b64, fake_prob, real_prob)
            logger.info(f"[AudioPipeline] XAI task dispatched for task_id={task_id}")
        except Exception as xai_err:
            logger.warning(f"[AudioPipeline] Failed to dispatch XAI task: {xai_err}")

        logger.info(f"[AudioPipeline] Completed for task_id={task_id}")
        return detection_payload

    except SoftTimeLimitExceeded:
        logger.error(f"[AudioPipeline] Soft time limit exceeded for task_id={task_id}")
        raise

    except Exception as exc:
        logger.error(f"[AudioPipeline] Error for task_id={task_id}: {exc}", exc_info=True)
        raise self.retry(exc=exc, countdown=10)
