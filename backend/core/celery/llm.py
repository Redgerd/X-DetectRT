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
    """Build message payload for LLM with the three images."""
    logger.debug(f"[LLM] Building prompt. Props: fake={fake_prob:.2f}, gradcam={bool(gradcam_b64)}, ela={bool(ela_b64)}")
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert deepfake analyst..."
        }
    ]
    
    user_content = []
    user_content.append({
        "type": "text",
        "text": f"Frame classification: Fake probability = {fake_prob:.2%}, Real probability = {real_prob:.2%}. "
                f"Analyze these images and explain why the model made its decision."
    })
    
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{normal_frame_b64}"}
    })
    
    if gradcam_b64:
        user_content.append({"type": "text", "text": "Grad-CAM heatmap:"})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gradcam_b64}"}})
    
    if ela_b64:
        user_content.append({"type": "text", "text": "Error Level Analysis (ELA):"})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ela_b64}"}})
    
    messages.append({"role": "user", "content": user_content})
    return messages


def _call_llm_api(prompt: list) -> str:
    """Call LLM API (OpenAI/Groq) with the prompt."""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("[LLM] Authentication Error: No API key found in environment variables.")
        return "LLM analysis unavailable: No API key configured"
    
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = "https://api.groq.com/openai/v1" if os.getenv("GROQ_API_KEY") else None

    logger.info(f"[LLM] Dispatching request to {model} (Base URL: {base_url or 'OpenAI Default'})")
    
    start_time = time.time()
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            max_tokens=500,
            temperature=0.3,
        )
        
        duration = time.time() - start_time
        logger.info(f"[LLM] API response received in {duration:.2f}s. Tokens: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"[LLM] API call failed after {time.time() - start_time:.2f}s: {str(e)}")
        return f"LLM analysis failed: {str(e)}"


@celery_app.task(name="llm.run_llm", bind=True, max_retries=3)
def run_llm(self, task_id: str, frame_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run LLM analysis on XAI results."""
    
    # Extract metadata for logging context
    frame_idx = frame_data.get("frame_index", "N/A")
    logger.info(f"[LLM] Starting Task - TaskID: {task_id} | Frame: {frame_idx}")
    
    try:
        normal_frame_b64 = frame_data.get("frame_data", "")
        gradcam_b64 = frame_data.get("gradcam_b64")
        ela_b64 = frame_data.get("ela_b64")
        fake_prob = frame_data.get("fake_prob", 0.0)
        real_prob = frame_data.get("real_prob", 0.0)
        
        if not normal_frame_b64:
            logger.warning(f"[LLM] Aborted: No image data for task {task_id}")
            return {"task_id": task_id, "analysis": "No frame data provided"}
        
        # Build prompt
        messages = _build_llm_prompt(
            normal_frame_b64=normal_frame_b64,
            gradcam_b64=gradcam_b64,
            ela_b64=ela_b64,
            fake_prob=fake_prob,
            real_prob=real_prob,
        )
        
        # Call API
        analysis = _call_llm_api(messages)
        
        logger.info(f"[LLM] Task Success - TaskID: {task_id} | Analysis Length: {len(analysis)} chars")
        return {
            "task_id": task_id,
            "analysis": analysis,
            "status": "success"
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"[LLM] Soft timeout exceeded for TaskID: {task_id}. Model was too slow.")
        return {"task_id": task_id, "analysis": "LLM analysis timed out"}
        
    except Exception as e:
        # Log stack trace for unexpected errors
        logger.exception(f"[LLM] Critical Failure for TaskID: {task_id}")
        return {"task_id": task_id, "analysis": f"LLM analysis failed: {str(e)}"}