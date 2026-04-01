# backend/services/explaination/explaination.py
"""
XAI service layer — fully independent of services/detection/model.py.

Maintains its own GenD singleton so the XAI service can be initialised,
restarted, or tested without coupling to the detection service.

Usage:
    from services.explaination.explaination import load_xai_model, get_xai_model, get_xai_device, get_xai_config, run_xai
    
    # Model loads automatically on first use, or can be pre-loaded:
    load_xai_model()  # optional, loads model now
    
    # Then use XAI features
    results = run_xai(b64_image, frame_probs)
"""

import logging
import os
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────
_xai_model  = None
_xai_device: Optional[str]  = None
_xai_config: Optional[dict] = None

# GenD checkpoint path — override via XAI_MODEL_PATH env var
_DEFAULT_MODEL_PATH = os.environ.get("XAI_MODEL_PATH", "/app/GenD_PE_L")

# xai_config.yaml path — override via XAI_CONFIG_PATH env var
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),   # same dir as explaination.py
    "xai_config.yaml",
)


# ─────────────────────────────────────────────────────────────────────────────
# Load XAI model (exactly like load_gend_model in detection)
# ─────────────────────────────────────────────────────────────────────────────
def load_xai_model(device=None):
    """
    Load GenD model for XAI once and store it globally.
    Subsequent calls reuse the same model.
    
    Args:
        device: Optional device string ('cuda' or 'cpu')
    
    Returns:
        The loaded GenD model
    """
    global _xai_model, _xai_device, _xai_config
    
    # Already loaded
    if _xai_model is not None:
        return _xai_model
    
    # ── Resolve device ────────────────────────────────────────────────────
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _xai_device = device
    
    # ── Load GenD model ───────────────────────────────────────────────────
    model_path = os.environ.get("XAI_MODEL_PATH", _DEFAULT_MODEL_PATH)
    logger.info(f"[XAI] Loading GenD from '{model_path}' on device '{device}' …")
    
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"[XAI] GenD checkpoint not found at '{model_path}'. "
            "Set the XAI_MODEL_PATH environment variable to override."
        )
    
    from services.detection.modeling_gend import GenD
    
    model = GenD.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()
    _xai_model = model
    logger.info(f"[XAI] GenD model loaded successfully on {device}.")
    
    # ── Load xai_config.yaml ──────────────────────────────────────────────
    config_path = os.environ.get("XAI_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
    config_path = os.path.normpath(config_path)
    logger.info(f"[XAI] Loading config from '{config_path}' …")
    
    if not os.path.exists(config_path):
        raise RuntimeError(
            f"[XAI] xai_config.yaml not found at '{config_path}'. "
            "Set the XAI_CONFIG_PATH environment variable to override."
        )
    
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    
    # xai_config.yaml has a top-level 'xai:' key
    _xai_config = raw.get("xai", raw)
    
    enabled = _xai_config.get("techniques", [])
    logger.info(f"[XAI] Config loaded — {len(enabled)} techniques enabled: {enabled}")
    
    return _xai_model


# ─────────────────────────────────────────────────────────────────────────────
# Getters (auto-load if needed)
# ─────────────────────────────────────────────────────────────────────────────
def get_xai_model():
    """Return the XAI-dedicated GenD model instance (auto-loads if needed)."""
    return load_xai_model()


def get_xai_device() -> str:
    """Return the device the XAI model was loaded onto (auto-loads if needed)."""
    load_xai_model()  # Ensure model is loaded
    return _xai_device


def get_xai_config() -> dict:
    """Return the parsed xai_config.yaml dict (auto-loads if needed)."""
    load_xai_model()  # Ensure config is loaded
    return _xai_config


# ─────────────────────────────────────────────────────────────────────────────
# Backward compatibility (kept for existing code that calls init_xai_model)
# ─────────────────────────────────────────────────────────────────────────────
def init_xai_model(device=None) -> None:
    """Legacy function - now just calls load_xai_model() for backward compatibility."""
    load_xai_model(device)


# ─────────────────────────────────────────────────────────────────────────────
# Public runner — called from Celery tasks or routes
# ─────────────────────────────────────────────────────────────────────────────
def run_xai(
    b64_image: str,
    frame_probs: Optional[list] = None,
    frame_tensors: Optional[list] = None,
) -> list:
    """
    Decode a base64 image and run all enabled XAI techniques.
    Model is automatically loaded on first call.

    Args:
        b64_image     : Base64-encoded image string (bare or data URI).
        frame_probs   : List[float] — per-frame fake probability for SHAP
                        TimeShap. Pass values already computed by GenD.
        frame_tensors : List[Tensor] — preprocessed frame tensors (optional,
                        also for SHAP TimeShap).

    Returns:
        List[dict] — one result dict per technique.
    """
    import base64
    import io

    from PIL import Image
    from .pipeline import run_xai_pipeline

    # ── Decode base64 → PIL.Image ─────────────────────────────────────────
    if b64_image.startswith("data:"):
        b64_image = b64_image.split(",", 1)[1]

    raw_bytes = base64.b64decode(b64_image.strip())
    pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    # ── Fetch singletons (will auto-load if needed) ───────────────────────
    model  = get_xai_model()      # This will load the model if not loaded
    config = get_xai_config()      # This will load the config
    device = get_xai_device()      # This will load the device

    # ── Run pipeline ──────────────────────────────────────────────────────
    return run_xai_pipeline(
        model=model,
        pil_image=pil_image,
        device=device,
        frame_tensors=frame_tensors,
        frame_probs=frame_probs,
        config=config,
    )


# 1 GRADCAP HEAT MAP
# 2 SHAP TimeShap — Temporal Frame Attribution