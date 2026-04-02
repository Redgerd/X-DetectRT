# backend/services/explaination/explaination.py
"""
XAI Explanation Service for X-DetectRT.

Loads and initialises the XAI pipeline for image/video (GenD-based) analysis.
Audio XAI is handled separately by services/audio/xai_audio.py but is
referenced from pipeline.py for unified orchestration.
"""

import os
import logging
import yaml

logger = logging.getLogger(__name__)

_XAI_CONFIG: dict = {}
_XAI_INITIALISED = False


def _load_config() -> dict:
    """Load xai_config.yaml from the same directory as this file."""
    config_path = os.environ.get(
        "XAI_CONFIG_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "xai_config.yaml"),
    )
    if not os.path.isfile(config_path):
        logger.warning(
            f"[XAI] xai_config.yaml not found at {config_path}. "
            "Using default empty config — XAI techniques will be skipped."
        )
        return {}
    try:
        with open(config_path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        logger.info(f"[XAI] Config loaded from {config_path}")
        return raw.get("xai", {})
    except Exception as e:
        logger.error(f"[XAI] Failed to parse xai_config.yaml: {e}", exc_info=True)
        return {}


def load_xai_model():
    """
    Initialise the XAI subsystem.

    Called once at FastAPI startup (main.py). For image/video, this reads the
    config so that individual XAI functions know which techniques are enabled.
    Audio XAI (WavLM-based IG + SHAP) does not require extra initialisation
    beyond the audio models themselves being loaded.
    """
    global _XAI_CONFIG, _XAI_INITIALISED

    if _XAI_INITIALISED:
        logger.debug("[XAI] Already initialised — skipping.")
        return

    _XAI_CONFIG = _load_config()
    enabled = _XAI_CONFIG.get("enabled", False)
    techniques = _XAI_CONFIG.get("techniques", [])

    logger.info(
        f"[XAI] Initialised — enabled={enabled}, techniques={techniques}"
    )
    _XAI_INITIALISED = True


def get_xai_config() -> dict:
    """Return the parsed XAI configuration dict."""
    return _XAI_CONFIG


def is_technique_enabled(technique: str) -> bool:
    """Return True if *technique* appears in the configured techniques list."""
    return technique in _XAI_CONFIG.get("techniques", [])
