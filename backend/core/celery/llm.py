import cv2
import numpy as np
import base64
import json
import logging
import time  # Added for performance tracking
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from typing import List, Dict, Any, Optional
import redis

from config import settings
from .celery_app import celery_app

logger = logging.getLogger(__name__)

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

# ... (base64_to_image and image_to_base64 remain the same) ...

def _build_llm_prompt(
    normal_frame_b64: str,
    gradcam_b64: Optional[str],
    ela_b64: Optional[str],
    fake_prob: float,
    real_prob: float,
) -> list:
    """Build optimized message payload for unified forensic analysis."""
    
    # 1. System Prompt: Updated for "Image as a Whole" logic
    system_content = (
        "You are a Forensic Deepfake Analyst. Your task is to analyze the provided "
        "imagery and describe the primary manipulation 'hot spots' by focusing on the "
        "image as a single, unified entity. Do not list separate findings for different "
        "tools; instead, provide a cohesive summary of where forensic signatures overlap."
    )

    # 2. User Content: Refined for spatial orientation and unified reporting
    analysis_context = (
        f"### Forensic Metadata\n"
        f"- **Model Detection Confidence (Fake):** {fake_prob:.2%}\n\n"
        "### Task\n"
        "Provide a single-paragraph summary identifying the specific spatial coordinates "
        "(e.g., central facial features, along the jawline, or near the hairline) where "
        "visual anomalies and high-intensity activations are most concentrated. "
        "Describe how these areas suggest localized digital inconsistencies or structural "
        "artifacts in the subject's geometry. Ensure the response treats all forensic "
        "signatures as a unified indicator of manipulation without explicitly naming "
        "the individual analysis techniques used (Grad-CAM or ELA). Avoid introductory filler."
    )

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": analysis_context},
                {"type": "text", "text": "Source Frame:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{normal_frame_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gradcam_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ela_b64}"}}
            ]
        }
    ]
    return messages


def _call_llm_api(prompt: list) -> str:
    """Call Groq API with the prompt."""
    import os
    from openai import OpenAI

    # 1. Configuration - Use Groq's official OpenAI-compatible endpoint
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    
    # Update your .env to use GROQ_API_KEY
    api_key = os.getenv("GROQ_API_KEY")

    # Llama 4 Scout is currently the recommended multimodal/vision model on Groq
    model = os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    if not api_key:
        logger.error("[LLM] Authentication Error: No GROQ_API_KEY found.")
        return "LLM analysis unavailable: No API key configured"

    logger.info(f"[LLM] Dispatching request to Groq Model: {model}")
    
    start_time = time.time()
    try:
        # 2. Initialize the client pointing to Groq
        client = OpenAI(
            api_key=api_key,
            base_url=GROQ_BASE_URL
        )
        
        # 3. Create completion
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            max_tokens=500,
            temperature=0.3,
        )
        
        duration = time.time() - start_time
        logger.info(f"[LLM] Groq response received in {duration:.2f}s.")
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"[LLM] Groq API call failed after {time.time() - start_time:.2f}s: {str(e)}")
        return f"LLM analysis failed: {str(e)}"


@celery_app.task(name="backend.core.celery.llm.run_llm", bind=True, max_retries=3)
def run_llm(self, task_id: str, frame_data: Dict[str, Any]) -> Dict[str, Any]:
    frame_idx = frame_data.get("frame_index", "N/A")
    logger.info(f"[LLM] Starting Task - TaskID: {task_id} | Frame: {frame_idx}")

    try:
        normal_frame_b64 = frame_data.get("frame_data", "")
        gradcam_b64      = frame_data.get("gradcam_b64")
        ela_b64          = frame_data.get("ela_b64")
        fake_prob        = frame_data.get("fake_prob", 0.0)
        real_prob        = frame_data.get("real_prob", 0.0)

        if not normal_frame_b64:
            logger.warning(f"[LLM] Aborted: No image data for task {task_id} frame {frame_idx}")
            return {"task_id": task_id, "analysis": "No frame data provided"}

        messages = _build_llm_prompt(
            normal_frame_b64=normal_frame_b64,
            gradcam_b64=gradcam_b64,
            ela_b64=ela_b64,
            fake_prob=fake_prob,
            real_prob=real_prob,
        )

        analysis = _call_llm_api(messages)

        result = {
            "task_id":      task_id,
            "frame_index":  frame_idx,
            "analysis":     analysis,
            "status":       "success",
        }

        # ── Publish to Redis ───────────────────────────────────────────────
        try:
            redis_client.publish(
                f"task_xai:{task_id}",
                json.dumps({
                    "type":         "llm_ready",
                    "task_id":      task_id,
                    "frame_index":  frame_idx,
                    "analysis":     analysis,
                }),
            )
            logger.info(f"[LLM] Published to Redis channel task_xai:{task_id} | Frame: {frame_idx}")
        except Exception as redis_err:
            logger.warning(f"[LLM] Redis publish failed for task {task_id} frame {frame_idx}: {redis_err}")

        logger.info(
            f"[LLM] Task Success - TaskID: {task_id} | Frame: {frame_idx} | "
            f"Analysis: {len(analysis)} chars"
        )
        return result

    except SoftTimeLimitExceeded:
        logger.error(f"[LLM] Soft timeout for TaskID: {task_id} | Frame: {frame_idx}")
        _publish_llm_error(task_id, frame_idx, "LLM analysis timed out")
        return {"task_id": task_id, "frame_index": frame_idx, "analysis": "LLM analysis timed out"}

    except Exception as e:
        logger.exception(f"[LLM] Critical Failure - TaskID: {task_id} | Frame: {frame_idx}")
        _publish_llm_error(task_id, frame_idx, str(e))
        return {"task_id": task_id, "frame_index": frame_idx, "analysis": f"LLM analysis failed: {str(e)}"}


def _publish_llm_error(task_id: str, frame_idx: Any, error: str) -> None:
    """Helper: publish an error event so consumers aren't left waiting."""
    try:
        redis_client.publish(
            f"task_xai:{task_id}",
            json.dumps({
                "type":        "llm_error",
                "task_id":     task_id,
                "frame_index": frame_idx,
                "error":       error,
            }),
        )
        logger.info(f"[LLM] Published error event to Redis for task {task_id} frame {frame_idx}")
    except Exception as e:
        logger.warning(f"[LLM] Failed to publish error event: {e}")