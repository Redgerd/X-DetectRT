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

    Returns:
        dict — keys depend on media type:
            audio  → {"ig_heatmap_b64": str|None, "shap_heatmap_b64": str|None}
            others → {} (not yet implemented)
    """
    logger.info(f"[XAIPipeline] run_xai called for media_type={media_type!r}")

    if media_type == "audio":
        return _run_audio_xai(**kwargs)

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