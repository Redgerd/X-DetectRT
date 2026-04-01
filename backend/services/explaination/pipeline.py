# backend/xai/pipeline.py
"""
XAI Pipeline — orchestrates all 7 XAI techniques and aggregates results.

Each technique is called independently. Failures are isolated — a failing
technique logs its error and returns a standard error dict without blocking
the others.

Usage:
    from xai.pipeline import run_xai_pipeline

    results = run_xai_pipeline(
        model=model,
        pil_image=pil_image,
        frame_tensors=frame_tensors,   # optional, for SHAP TimeShap
        frame_probs=frame_probs,        # optional, for SHAP TimeShap
        device="cuda",
        config=xai_config,             # optional dict from xai_config.yaml
    )
"""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Technique module registry — lazy imports so missing deps don't crash import
TECHNIQUE_MODULES = {
    "shap_timeshap":        "services.explaination.shap_timeshap",
    "lime_superpixels":     "services.explaination.lime_superpixels",
    "integrated_gradients": "services.explaination.integrated_gradients",
    "sam_attribution":      "services.explaination.sam_attribution",
    "counterfactual":       "services.explaination.counterfactual",
    "tcav":                 "services.explaination.tcav",
    "prototype_analysis":   "services.explaination.prototype_analysis",
}

DEFAULT_ENABLED = [
    "shap_timeshap",        # Fast - uses pre-computed probabilities
    "tcav",                 # Fast - uses simulated concepts
    "prototype_analysis",   # Fast - cosine similarity only
    # Disabled slow techniques:
    # "lime_superpixels",     # Too slow
    # "integrated_gradients", # Too slow
    # "counterfactual",       # Too slow
]


def _run_one_technique(name: str, module_path: str, model, inputs, kwargs: dict) -> dict:
    """
    Dynamically import and call the explain() function of a technique module.
    Returns a result dict guaranteed to have 'technique' and 'figure_base64' keys.
    Errors are caught and returned as { technique, error, figure_base64: None }.
    """
    try:
        import importlib
        mod = importlib.import_module(module_path)
        t0 = time.perf_counter()
        result = mod.explain(model, inputs, **kwargs)
        elapsed = round(time.perf_counter() - t0, 2)
        result["elapsed_seconds"] = elapsed
        logger.info(f"[XAI Pipeline] '{name}' completed in {elapsed}s")
        return result
    except Exception as exc:
        logger.exception(f"[XAI Pipeline] Technique '{name}' failed: {exc}")
        return {
            "technique": name,
            "error": str(exc),
            "figure_base64": None,
        }


def run_xai_pipeline(
    model: Any,
    pil_image: Any,
    device: str = "cpu",
    frame_tensors: Optional[list] = None,
    frame_probs: Optional[list] = None,
    config: Optional[dict] = None,
) -> list:
    """
    Run all enabled XAI techniques and return a list of result dicts.

    Args:
        model        : Loaded GenD model (eval mode).
        pil_image    : PIL.Image of the frame to explain.
        device       : Torch device string.
        frame_tensors: List of preprocessed frame tensors (for SHAP TimeShap).
        frame_probs  : List of float fake_prob per frame (for SHAP TimeShap).
        config       : Optional dict from xai_config.yaml xai section.

    Returns:
        List[dict] — one dict per technique with keys:
            technique, scores, figure_base64, [error], [narrative], [elapsed_seconds]
    """
    cfg = config or {}
    enabled = cfg.get("techniques", DEFAULT_ENABLED)

    results = []

    for name in enabled:
        if name not in TECHNIQUE_MODULES:
            logger.warning(f"[XAI Pipeline] Unknown technique '{name}', skipping.")
            continue

        module_path = TECHNIQUE_MODULES[name]

        # Build per-technique kwargs
        kwargs: dict = {"device": device}

        if name == "shap_timeshap":
            if frame_tensors is not None:
                kwargs["frame_tensors"] = frame_tensors
            if frame_probs is not None:
                kwargs["frame_probs"] = frame_probs

        elif name == "lime_superpixels":
            kwargs["num_samples"] = cfg.get("lime_num_samples", 300)

        elif name == "integrated_gradients":
            kwargs["ig_steps"] = cfg.get("ig_steps", 50)

        # Inputs: for SHAP we pass None (uses frame_probs from kwargs)
        # For all other techniques we pass pil_image
        inputs = None if name == "shap_timeshap" else pil_image

        result = _run_one_technique(name, module_path, model, inputs, kwargs)
        results.append(result)

    return results
