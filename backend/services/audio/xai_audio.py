# backend/services/audio/xai_audio.py
"""
Module C: Explanation Engine (XAI) for Audio Deepfake Detection.

Returns RAW attribution score vectors (not PNG images).
The React frontend renders these as Canvas overlays.

Two complementary methods:

    1. Integrated Gradients (captum)
       Gradient path from a zero-baseline feature tensor.
       Tells us which WavLM time-frames pushed the decision toward "Fake".
       Returns: list[float], one score per WavLM time-frame.

    2. SHAP KernelExplainer (model-agnostic)
       Uses a numpy-callable wrapper — no Keras / TF dependency.
       Replaced DeepExplainer / GradientExplainer which crash on newer SHAP
       versions when wrapping PyTorch nn.Module callables.
       Returns: list[float], one Shapley value per WavLM time-frame.
"""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Hard-coded for speed (KernelExplainer cost scales with n_background)
SHAP_N_BACKGROUND = 3


# ---------------------------------------------------------------------------
# Method 1: Integrated Gradients (captum)
# ---------------------------------------------------------------------------

def generate_integrated_gradients_xai(
    audio_tensor:   torch.FloatTensor,
    detector_model: torch.nn.Module,
    wavlm_model:    torch.nn.Module,
    device:         str,
    target_class:   int = 1,   # 1 = Fake
    n_steps:        int = 50,
) -> list:
    """
    Run Integrated Gradients over WavLM feature frames.

    Args:
        audio_tensor:   1-D float tensor (64 600 samples at 16 kHz).
        detector_model: DeepFakeDetector (eval mode).
        wavlm_model:    WavLM Base+ (frozen, eval mode).
        device:         Torch device string.
        target_class:   Output index to attribute (1 = Fake).
        n_steps:        IG interpolation steps.

    Returns:
        list[float] — per-frame attribution (mean over 768 hidden dims),
        length = number of WavLM output frames (≈ 201 for 64 600-sample input).
        Positive = evidence of Fake; Negative = evidence of Real.
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise RuntimeError("captum is required: pip install captum")

    detector_model.eval()

    # Extract WavLM features (frozen — no grad)
    with torch.no_grad():
        inp      = audio_tensor.unsqueeze(0).to(device)          # (1, N)
        features = wavlm_model(input_values=inp).last_hidden_state  # (1, T, 768)

    # IG needs requires_grad on the input features
    features = features.detach().requires_grad_(True)

    def _forward(feat: torch.Tensor) -> torch.Tensor:
        return detector_model(feat)   # (B, 2)

    ig        = IntegratedGradients(_forward)
    baseline  = torch.zeros_like(features)

    attributions = ig.attribute(
        features,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=False,
    )  # (1, T, 768)

    # Mean over hidden dim → (T,)
    attr_scores = attributions[0].detach().cpu().numpy().mean(axis=-1)

    # Normalise to [-1, 1] for consistent frontend rendering
    abs_max = np.abs(attr_scores).max()
    if abs_max > 1e-8:
        attr_scores = attr_scores / abs_max

    logger.info(
        f"[XAI-IG] frames={len(attr_scores)}, "
        f"max={attr_scores.max():.3f}, min={attr_scores.min():.3f}"
    )
    return attr_scores.tolist()


# ---------------------------------------------------------------------------
# Method 2: SHAP KernelExplainer (model-agnostic, no TF/Keras dependency)
# ---------------------------------------------------------------------------

def generate_shap_xai(
    audio_tensor:   torch.FloatTensor,
    detector_model: torch.nn.Module,
    wavlm_model:    torch.nn.Module,
    device:         str,
    target_class:   int = 1,
    n_background:   int = SHAP_N_BACKGROUND,
) -> list:
    """
    Run SHAP KernelExplainer over WavLM feature frames.

    KernelExplainer is model-agnostic — it calls the model as a black-box
    numpy function, so there are no Keras 3 / TF 2.16 compatibility issues.

    Strategy:
        - Flatten (1, T, 768) → (1, T*768) for KernelExplainer input.
        - Background = *n_background* small Gaussian-noise tensors.
        - After shap_values() reshape back to (T, 768) and mean over hidden dim.

    Args:
        audio_tensor:   1-D float tensor (64 600 samples).
        detector_model: DeepFakeDetector.
        wavlm_model:    WavLM Base+ (frozen).
        device:         Torch device string.
        target_class:   Output index to explain (1 = Fake).
        n_background:   Background samples (3 = fast, ~3–5 s on CPU).

    Returns:
        list[float] — per-frame SHAP values, same length as IG output.
        Positive = pushed toward Fake; Negative = pushed toward Real.
    """
    try:
        import shap
    except ImportError:
        raise RuntimeError("shap is required: pip install shap")

    detector_model.eval()

    # Extract WavLM features
    with torch.no_grad():
        inp      = audio_tensor.unsqueeze(0).to(device)
        features = wavlm_model(input_values=inp).last_hidden_state  # (1, T, 768)

    # Calculate dimensions
    seq_len    = features.shape[1]
    hidden_dim = features.shape[2]
    
    # We want to explain seq_len features (one per time frame)
    # The SHAP explainer will generate binary masks of shape (N, seq_len)
    # For a mask of 1s and 0s, we reconstruct the tensor by taking either the original frame (1) or the background (0).
    feat_np = features[0].cpu().numpy() # (seq_len, hidden_dim)
    
    # ── Model wrapper: numpy (N, seq_len) → numpy (N, 2) ────────────────────
    # x_mask contains values between 0 and 1 indicating how much of the original feature vs background to use.
    def _predict_np(x_mask: np.ndarray) -> np.ndarray:
        N = x_mask.shape[0]
        # Expand mask to (N, seq_len, 1) to broadcast over hidden_dim
        mask_tensor = torch.tensor(x_mask, dtype=torch.float32, device=device).unsqueeze(-1)
        
        # Original features broadcasted to N
        orig = features.expand(N, -1, -1)
        
        # Create a simple zero background or mean background
        # Since SHAP requires a background, we just use 0 here for the background tensor
        bg = torch.zeros_like(orig)
        
        # Blend features according to SHAP mask
        blended = orig * mask_tensor + bg * (1 - mask_tensor)
        
        with torch.no_grad():
            logits = detector_model(blended)
        return logits.cpu().numpy()

    logger.info(f"[XAI-SHAP] KernelExplainer: Explaining {seq_len} temporal frames...")
    
    # The reference 'base' is an array of 0s (meaning fully background)
    # The instance to explain is an array of 1s (meaning fully original)
    background_mask = np.zeros((1, seq_len))
    instance_mask = np.ones((1, seq_len))
    
    explainer = shap.KernelExplainer(_predict_np, background_mask)
    shap_values = explainer.shap_values(instance_mask, nsamples=128, silent=True)
    # shap_values is list of arrays: one for each output class. Shape is (1, seq_len)
    
    if isinstance(shap_values, list):
        sv_frame = shap_values[target_class][0]   # (seq_len,)
    else:
        # Some versions return 3D arrays (1, seq_len, 2)
        if len(shap_values.shape) == 3:
            sv_frame = shap_values[0, :, target_class]
        else:
            sv_frame = shap_values[0]

    # Normalise
    abs_max = np.abs(sv_frame).max()
    if abs_max > 1e-8:
        sv_frame = sv_frame / abs_max

    logger.info(f"[XAI-SHAP] done — max={sv_frame.max():.3f}, min={sv_frame.min():.3f}")
    return sv_frame.tolist()


# ---------------------------------------------------------------------------
# Convenience wrapper — orchestrates both methods
# ---------------------------------------------------------------------------

def generate_audio_xai(
    audio_tensor:   torch.FloatTensor,
    detector_model: torch.nn.Module,
    wavlm_model:    torch.nn.Module,
    device:         str,
) -> dict:
    """
    Run both XAI methods and return score vectors.

    Returns:
        {
            "ig_scores":   list[float] | None,
            "shap_scores": list[float] | None,
        }
    """
    results: dict = {"ig_scores": None, "shap_scores": None}

    # Integrated Gradients
    try:
        results["ig_scores"] = generate_integrated_gradients_xai(
            audio_tensor, detector_model, wavlm_model, device
        )
        logger.info("[XAI-Audio] ✅ Integrated Gradients scores generated.")
    except Exception as e:
        logger.error(f"[XAI-Audio] IG failed: {e}", exc_info=True)

    # SHAP KernelExplainer
    try:
        results["shap_scores"] = generate_shap_xai(
            audio_tensor, detector_model, wavlm_model, device
        )
        logger.info("[XAI-Audio] ✅ SHAP KernelExplainer scores generated.")
    except Exception as e:
        logger.error(f"[XAI-Audio] SHAP failed: {e}", exc_info=True)

    return results