# backend/services/audio/xai_audio.py
"""
READ – Module C: Explanation Engine (XAI)

Implements two complementary explanation methods:

    1. Integrated Gradients (captum) – gradient-based temporal attribution.
       Tells us which WavLM time-frames pushed the decision toward "Fake".

    2. SHAP DeepExplainer – game-theory–based attribution over the feature
       sequence. More stable for segment-level explanations.

Both produce a 1-D temporal score vector that is:
    - Upsampled to raw sample resolution (64 600)
    - Overlaid on the audio waveform as a red/blue heatmap PNG
    - Returned as a base64-encoded string
"""

import io
import base64
import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tensor_to_heatmap_png(
    waveform: np.ndarray,
    attribution: np.ndarray,
    sample_rate: int = 16_000,
    title: str = "Temporal Attribution Heatmap",
) -> str:
    """
    Render a waveform overlaid with a red/blue attribution heatmap.

    Args:
        waveform:     1-D float array of raw audio samples.
        attribution:  1-D float array, same length as *waveform*.
                      Positive values → "Fake" evidence; Negative → "Real".
        sample_rate:  For computing the time axis.
        title:        Plot title.

    Returns:
        Base64-encoded PNG (no data-URI prefix).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import matplotlib.colors as mcolors
    except ImportError:
        raise RuntimeError("matplotlib is required: pip install matplotlib")

    times = np.linspace(0, len(waveform) / sample_rate, num=len(waveform))

    # Normalise attribution to [-1, 1]
    abs_max = np.abs(attribution).max()
    if abs_max > 1e-8:
        norm_attr = attribution / abs_max
    else:
        norm_attr = np.zeros_like(attribution)

    fig, axes = plt.subplots(2, 1, figsize=(12, 4), dpi=100, sharex=True)
    fig.suptitle(title, fontsize=10, fontweight="bold")

    # Top panel: raw waveform
    axes[0].plot(times, waveform, color="#4A90E2", linewidth=0.5, alpha=0.85)
    axes[0].set_ylabel("Amplitude", fontsize=8)
    axes[0].set_xlim(0, times[-1])
    axes[0].tick_params(labelsize=7)

    # Bottom panel: attribution heatmap as a coloured fill
    # Positive (red) = evidence of "Fake"; Negative (blue) = evidence of "Real"
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    axes[1].fill_between(
        times,
        norm_attr,
        where=(norm_attr >= 0),
        color="#E74C3C",
        alpha=0.7,
        label="↑ Fake evidence",
    )
    axes[1].fill_between(
        times,
        norm_attr,
        where=(norm_attr < 0),
        color="#3498DB",
        alpha=0.7,
        label="↓ Real evidence",
    )
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_ylabel("Attribution", fontsize=8)
    axes[1].set_xlabel("Time (s)", fontsize=8)
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].tick_params(labelsize=7)

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _upsample_scores(scores: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate a short attribution vector to *target_len*."""
    src_idx = np.linspace(0, len(scores) - 1, num=len(scores))
    tgt_idx = np.linspace(0, len(scores) - 1, num=target_len)
    return np.interp(tgt_idx, src_idx, scores)


# ---------------------------------------------------------------------------
# Method 1: Integrated Gradients (captum)
# ---------------------------------------------------------------------------

def generate_integrated_gradients_xai(
    audio_tensor: torch.FloatTensor,
    detector_model: torch.nn.Module,
    wavlm_model: torch.nn.Module,
    device: str,
    target_class: int = 1,          # 1 = Fake
    n_steps: int = 50,
) -> str:
    """
    Generate a temporal attribution heatmap using Integrated Gradients.

    IG attributes the "Fake" class score to each WavLM time-frame by
    accumulating gradients along the straight-line path from a zero
    baseline to the actual feature sequence.

    Args:
        audio_tensor:   Pre-processed 1-D float tensor (64 600 samples).
        detector_model: Loaded DeepFakeDetector (eval mode).
        wavlm_model:    Loaded WavLM Base+ (frozen, eval mode).
        device:         Torch device string.
        target_class:   Which logit to attribute (1 = Fake).
        n_steps:        Number of IG integration steps (more = more accurate
                        but slower; 50 is a good balance).

    Returns:
        Base64-encoded PNG of waveform + heatmap.
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise RuntimeError("captum is required: pip install captum")

    detector_model.eval()

    # Extract WavLM features (frozen, no grad needed here)
    with torch.no_grad():
        inp = audio_tensor.unsqueeze(0).to(device)         # (1, 64600)
        features = wavlm_model(input_values=inp).last_hidden_state  # (1, T, 768)

    features = features.detach().requires_grad_(True)

    # Wrap detector so IG can call it with feature tensors directly
    def _forward(feat: torch.Tensor) -> torch.Tensor:
        logits = detector_model(feat)                       # (B, 2)
        return logits

    ig = IntegratedGradients(_forward)

    baseline = torch.zeros_like(features)

    attributions = ig.attribute(
        features,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=False,
    )  # (1, T, 768)

    # Reduce over hidden dim → (T,)
    attr_scores = attributions[0].detach().cpu().numpy().mean(axis=-1)  # (T,)

    # Upsample to raw sample resolution
    waveform_np = audio_tensor.cpu().numpy()
    attr_upsampled = _upsample_scores(attr_scores, len(waveform_np))

    return _tensor_to_heatmap_png(
        waveform_np,
        attr_upsampled,
        title="Integrated Gradients – Temporal Attribution (Fake Class)",
    )


# ---------------------------------------------------------------------------
# Method 2: SHAP DeepExplainer
# ---------------------------------------------------------------------------

def generate_shap_xai(
    audio_tensor: torch.FloatTensor,
    detector_model: torch.nn.Module,
    wavlm_model: torch.nn.Module,
    device: str,
    target_class: int = 1,
    n_background: int = 5,
) -> str:
    """
    Generate a temporal attribution heatmap using SHAP DeepExplainer.

    SHAP assigns each WavLM time-frame a Shapley value that reflects its
    marginal contribution to the "Fake" prediction relative to a set of
    background (reference) samples synthesised from Gaussian noise.

    Args:
        audio_tensor:    Pre-processed 1-D float tensor (64 600 samples).
        detector_model:  Loaded DeepFakeDetector (eval mode).
        wavlm_model:     Loaded WavLM Base+ (frozen).
        device:          Torch device string.
        target_class:    Which output neuron to explain (1 = Fake).
        n_background:    Number of noise-based background samples for SHAP.

    Returns:
        Base64-encoded PNG of waveform + heatmap.
    """
    try:
        import shap
    except ImportError:
        raise RuntimeError("shap is required: pip install shap")

    detector_model.eval()

    # Extract WavLM features
    with torch.no_grad():
        inp = audio_tensor.unsqueeze(0).to(device)
        features = wavlm_model(input_values=inp).last_hidden_state   # (1, T, 768)

    seq_len = features.shape[1]

    # Build background set: small Gaussian-noise feature tensors
    background = torch.randn(n_background, seq_len, 768, device=device) * 0.01

    # SHAP wrapper: take (B, T, 768) → (B, 2)
    def _model_fn(feat_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return detector_model(feat_batch)

    try:
        explainer = shap.DeepExplainer(_model_fn, background)
        shap_values = explainer.shap_values(features)   # list of (1, T, 768) per class
    except Exception as e:
        logger.warning(
            f"[XAI-SHAP] DeepExplainer failed ({e}); falling back to GradientExplainer."
        )
        try:
            explainer = shap.GradientExplainer(_model_fn, background)
            shap_values = explainer.shap_values(features)
        except Exception as e2:
            logger.error(f"[XAI-SHAP] GradientExplainer also failed: {e2}", exc_info=True)
            raise RuntimeError(f"SHAP explanation failed: {e2}") from e2

    # shap_values is a list indexed by class; select target_class
    if isinstance(shap_values, list):
        sv = shap_values[target_class]          # (1, T, 768)
    else:
        sv = shap_values                        # some versions return ndarray

    if isinstance(sv, np.ndarray):
        attr_scores = sv[0].mean(axis=-1)       # (T,)
    else:
        attr_scores = sv[0].detach().cpu().numpy().mean(axis=-1)

    waveform_np = audio_tensor.cpu().numpy()
    attr_upsampled = _upsample_scores(attr_scores, len(waveform_np))

    return _tensor_to_heatmap_png(
        waveform_np,
        attr_upsampled,
        title="SHAP – Temporal Attribution (Fake Class)",
    )


# ---------------------------------------------------------------------------
# Convenience wrapper — runs both and returns a dict
# ---------------------------------------------------------------------------

def generate_audio_xai(
    audio_tensor: torch.FloatTensor,
    detector_model: torch.nn.Module,
    wavlm_model: torch.nn.Module,
    device: str,
) -> dict:
    """
    Run both XAI methods and return base64 PNG strings in a dict.

    Returns:
        {
            "ig_heatmap_b64":   str | None,
            "shap_heatmap_b64": str | None,
        }
    """
    from typing import Dict, Optional as Opt
    results: Dict[str, Opt[str]] = {"ig_heatmap_b64": None, "shap_heatmap_b64": None}

    # Integrated Gradients
    try:
        results["ig_heatmap_b64"] = generate_integrated_gradients_xai(
            audio_tensor, detector_model, wavlm_model, device
        )
        logger.info("[XAI-Audio] ✅ Integrated Gradients heatmap generated.")
    except Exception as e:
        logger.error(f"[XAI-Audio] IG failed: {e}", exc_info=True)

    # SHAP
    try:
        results["shap_heatmap_b64"] = generate_shap_xai(
            audio_tensor, detector_model, wavlm_model, device
        )
        logger.info("[XAI-Audio] ✅ SHAP heatmap generated.")
    except Exception as e:
        logger.error(f"[XAI-Audio] SHAP failed: {e}", exc_info=True)

    return results
