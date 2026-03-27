# backend/xai/__init__.py
"""
XAI (Explainable AI) techniques for X-DetectRT deepfake detection.

Techniques:
    03 - shap_timeshap       : Temporal SHAP per-frame contribution
    07 - lime_superpixels    : LIME with anatomical facial superpixels
    08 - integrated_gradients: Integrated Gradients with facial zone masks
    09 - sam_attribution     : SAM-guided semantic attribution
    10 - counterfactual      : Counterfactual minimum perturbations
    11 - tcav                : Concept Activation Vectors (TCAV)
    12 - prototype_analysis  : Cosine similarity vs known deepfake prototypes
"""

from .pipeline import run_xai_pipeline

__all__ = ["run_xai_pipeline"]
