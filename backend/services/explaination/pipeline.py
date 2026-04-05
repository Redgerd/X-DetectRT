# backend/services/explaination/pipeline.py
"""
Unified XAI Pipeline Orchestrator for X-DetectRT.

Routes XAI generation requests to the correct method depending on media type:
  - "audio"  → services.audio.xai_audio.generate_audio_xai
  - "image" / "video" → (future: image/video XAI methods)

Usage example (from a route or Celery task):

    from services.explaination.pipeline import run_xai

    result = run_xai(
        media_type="audio",
        audio_tensor=tensor,
        detector_model=detector,
        wavlm_model=wavlm,
        device="cpu",
    )
    # result -> {"ig_heatmap_b64": "...", "shap_heatmap_b64": "..."}
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def run_xai(media_type: str, **kwargs) -> dict:
    """
    Dispatch XAI generation for the given *media_type*.

    Args:
        media_type: One of "audio", "image", "video".
        **kwargs:   Media-type–specific arguments forwarded to the underlying
                    XAI function (see below).

    Keyword args for media_type="audio":
        audio_tensor   (torch.FloatTensor) — pre-processed 1-D audio tensor
        detector_model (nn.Module)          — loaded DeepFakeDetector
        wavlm_model    (nn.Module)          — loaded WavLM Base+ (frozen)
        device         (str)                — torch device string

    Keyword args for media_type="image":
        b64_image      (str)                — base64 encoded image
        device         (str)                — torch device string ("cpu" or "cuda")

    Returns:
        dict — keys depend on media type:
            audio  → {"ig_heatmap_b64": str|None, "shap_heatmap_b64": str|None}
            image  → {"gradcam_b64": str|None, "ela_b64": str|None}
            others → {} (not yet implemented)
    """
    logger.info(f"[XAIPipeline] run_xai called for media_type={media_type!r}")

    if media_type == "audio":
        return _run_audio_xai(**kwargs)

    if media_type == "image":
        return _run_image_xai(**kwargs)

    # Placeholder for future image/video XAI
    logger.warning(
        f"[XAIPipeline] XAI not yet implemented for media_type={media_type!r}. "
        "Returning empty result."
    )
    return {}


# ---------------------------------------------------------------------------
# Internal dispatcher — audio
# ---------------------------------------------------------------------------

def _run_audio_xai(
    audio_tensor: Any,
    detector_model: Any,
    wavlm_model: Any,
    device: str = "cpu",
) -> dict:
    """
    Run Integrated Gradients + SHAP temporal heatmaps for an audio clip.

    Returns:
        {"ig_heatmap_b64": str|None, "shap_heatmap_b64": str|None}
    """
    try:
        from services.audio.xai_audio import generate_audio_xai
        results = generate_audio_xai(audio_tensor, detector_model, wavlm_model, device)
        logger.info(
            f"[XAIPipeline] Audio XAI complete — "
            f"IG={'OK' if results.get('ig_heatmap_b64') else 'FAIL'}, "
            f"SHAP={'OK' if results.get('shap_heatmap_b64') else 'FAIL'}"
        )
        return results
    except Exception as e:
        logger.error(f"[XAIPipeline] Audio XAI failed: {e}", exc_info=True)
        return {"ig_heatmap_b64": None, "shap_heatmap_b64": None}


def _run_image_xai(
    b64_image: str,
    device: str = "cpu",
) -> dict:
    """
    Run Grad-CAM++ and Error Level Analysis (ELA) for an image.

    Returns:
        {"gradcam_b64": str|None, "ela_b64": str|None}
    """
    try:
        from services.explaination.explaination import (
            generate_gradcam,
            generate_ela,
            get_xai_model,
        )
        from services.explaination.explaination import _pil_to_base64
        from PIL import Image
        import torch
        import base64
        import numpy as np
        import cv2

        # Decode base64 to PIL Image
        encoded_data = b64_image.split(',')[1] if ',' in b64_image else b64_image
        img_bytes = base64.b64decode(encoded_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(rgb)

        # Load model for Grad-CAM
        model = get_xai_model()
        model.to(device)
        model.eval()

        # Preprocess for Grad-CAM
        preprocess = model.feature_extractor.preprocess
        image_tensor = preprocess(original_pil).unsqueeze(0).to(device)

        # Generate Grad-CAM
        gradcam_b64 = generate_gradcam(model, image_tensor, original_pil)

        # Generate ELA
        ela_b64 = generate_ela(original_pil)

        logger.info(
            f"[XAIPipeline] Image XAI complete — "
            f"Grad-CAM={'OK' if gradcam_b64 else 'FAIL'}, "
            f"ELA={'OK' if ela_b64 else 'FAIL'}"
        )
        return {"gradcam_b64": gradcam_b64, "ela_b64": ela_b64}

    except Exception as e:
        logger.error(f"[XAIPipeline] Image XAI failed: {e}", exc_info=True)
        return {"gradcam_b64": None, "ela_b64": None}