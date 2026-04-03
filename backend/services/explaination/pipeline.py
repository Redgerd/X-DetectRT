# backend/services/explaination/pipeline.py
"""
XAI Pipeline — real Grad-CAM spatial heatmaps + SHAP TimeShap temporal attribution.

Techniques implemented
──────────────────────
1. Grad-CAM (ViT-adapted)
   Hooks the last transformer encoder block, back-propagates from the "fake"
   class logit, and reshapes patch-level gradient weights into a spatial
   heatmap overlaid on the original image (returned as a base64 PNG).

2. SHAP TimeShap
   Uses the `timeshap` library (or falls back to `shap.KernelExplainer`) to
   compute per-frame Shapley-value attributions for a sequence of frames.
   Requires either raw `frame_tensors` (preferred) or scalar `frame_probs`.

Supported GenD backbones
──────────────────────────
  • CLIPEncoder      — last layer: vision_model.encoder.layers[-1]
  • DINOEncoder      — last layer: backbone.encoder.layer[-1]
  • PerceptionEncoder— last layer: backbone.blocks[-1]
"""

from __future__ import annotations

import base64
import io
import logging
import math
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_last_encoder_layer(model):
    """
    Return the last transformer encoder block for any supported GenD backbone.
    Raises ValueError if the backbone type is unrecognised.
    """
    fe = model.feature_extractor
    backbone_cls = type(fe).__name__

    if backbone_cls == "CLIPEncoder":
        # HuggingFace CLIP vision model
        return fe.vision_model.encoder.layers[-1]

    elif backbone_cls == "DINOEncoder":
        # HuggingFace DINOv2
        return fe.backbone.encoder.layer[-1]

    elif backbone_cls == "PerceptionEncoder":
        # timm EVA-ViT
        return fe.backbone.blocks[-1]

    else:
        # Generic fallback — try common attribute names
        for attr in ("blocks", "layers", "layer"):
            seq = getattr(fe, attr, None)
            if seq is not None and hasattr(seq, "__len__") and len(seq) > 0:
                logger.warning(
                    "[GradCAM] Unknown backbone '%s'; using '.%s[-1]' as last layer.",
                    backbone_cls, attr,
                )
                return seq[-1]

        raise ValueError(
            f"[GradCAM] Cannot locate last encoder layer for backbone '{backbone_cls}'. "
            "Supported types: CLIPEncoder, DINOEncoder, PerceptionEncoder."
        )


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1]. Returns zero array if constant."""
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr, dtype=np.float32)


def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Grad-CAM (ViT-adapted)
# ─────────────────────────────────────────────────────────────────────────────

def run_gradcam(
    model,
    pil_image: Image.Image,
    device: str,
    target_class: int = 1,          # 1 = "fake"
    alpha_overlay: float = 0.45,    # heatmap blend strength
) -> dict:
    """
    Compute a Grad-CAM heatmap for a single PIL image using the GenD backbone.

    Algorithm
    ─────────
    1. Forward pass with a hook on the last encoder block output.
    2. Back-propagate from model.model(features)[0, target_class].
    3. Capture gradient of that block's output tensor.
    4. Weight each patch token by its mean gradient (channel dimension).
    5. Apply ReLU, reshape to square spatial grid, resize to original image.
    6. Apply JET colormap and blend with original image.

    Parameters
    ──────────
    model        : GenD instance (eval mode expected).
    pil_image    : Input PIL image (any size/mode).
    device       : 'cuda' | 'cpu'.
    target_class : Logit index to back-prop from (default 1 = fake).
    alpha_overlay: Weight of the heatmap in the blended overlay [0, 1].

    Returns
    ───────
    dict:
        technique    : "gradcam"
        heatmap_b64  : Base64-encoded PNG of the overlay image.
        raw_scores   : List[float] — per-patch importance scores (flattened).
        grid_size    : int — side length of the patch grid (G × G).
        target_class : int — class used for back-propagation.
        status       : "ok" | "error"
        error        : str (only present if status == "error")
    """
    model.eval()

    fe = model.feature_extractor

    # ── 1. Preprocess image ───────────────────────────────────────────────────
    img_tensor = fe.preprocess(pil_image).unsqueeze(0).to(device)   # [1, C, H, W]
    img_tensor.requires_grad_(True)

    # ── 2. Register hooks ─────────────────────────────────────────────────────
    _activations: dict = {}
    _gradients:   dict = {}

    try:
        last_layer = _get_last_encoder_layer(model)
    except ValueError as exc:
        logger.error("[GradCAM] %s", exc)
        return {"technique": "gradcam", "status": "error", "error": str(exc),
                "heatmap_b64": "", "raw_scores": [], "grid_size": 0, "target_class": target_class}

    def _fwd_hook(_module, _inp, output):
        # Some HF layers return tuples (hidden_states, attn_weights, ...)
        feat = output[0] if isinstance(output, tuple) else output
        _activations["feat"] = feat  # do NOT .detach() — keep in graph

    def _bwd_hook(_module, _grad_in, grad_out):
        grad = grad_out[0] if isinstance(grad_out, tuple) else grad_out
        _gradients["feat"] = grad.detach()

    fwd_handle = last_layer.register_forward_hook(_fwd_hook)
    bwd_handle = last_layer.register_full_backward_hook(_bwd_hook)

    try:
        # ── 3. Forward pass ───────────────────────────────────────────────────
        features = fe(img_tensor)           # [1, D]
        logits   = model.model(features)    # [1, num_classes]
        score    = logits[0, target_class]

        # ── 4. Backward pass ──────────────────────────────────────────────────
        model.zero_grad()
        score.backward()

    except Exception as exc:
        logger.error("[GradCAM] Forward/backward pass failed: %s", exc, exc_info=True)
        return {"technique": "gradcam", "status": "error", "error": str(exc),
                "heatmap_b64": "", "raw_scores": [], "grid_size": 0, "target_class": target_class}
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    feat = _activations.get("feat")     # [1, seq_len, D]
    grad = _gradients.get("feat")       # [1, seq_len, D]

    if feat is None or grad is None:
        msg = "Hooks captured no data — check that the last encoder layer is differentiable."
        logger.warning("[GradCAM] %s", msg)
        return {"technique": "gradcam", "status": "error", "error": msg,
                "heatmap_b64": "", "raw_scores": [], "grid_size": 0, "target_class": target_class}

    # ── 5. Compute per-patch importance ───────────────────────────────────────
    # weight each token by the mean absolute gradient magnitude across channels
    weights   = grad.mean(dim=-1)                           # [1, seq_len]
    cam_token = (weights.unsqueeze(-1) * feat).sum(dim=-1)  # [1, seq_len]
    cam_token = cam_token[0]                                # [seq_len]

    # Drop CLS token at index 0 — patch tokens are 1 …
    patch_cam = cam_token[1:]                               # [num_patches]
    patch_cam = F.relu(patch_cam).detach().cpu()

    num_patches = patch_cam.shape[0]
    grid_size   = int(math.isqrt(num_patches))              # exact square?

    if grid_size * grid_size != num_patches:
        # Non-square grid (e.g. rectangular or register tokens): pad to next square
        grid_size = math.ceil(math.sqrt(num_patches))
        pad_len   = grid_size * grid_size - num_patches
        patch_cam = F.pad(patch_cam, (0, pad_len))

    spatial_cam = patch_cam.reshape(grid_size, grid_size).numpy()
    spatial_cam = _normalise(spatial_cam)

    # ── 6. Overlay on original image ──────────────────────────────────────────
    orig_np = np.array(pil_image.convert("RGB"))            # [H, W, 3]
    H, W    = orig_np.shape[:2]

    heatmap_resized = cv2.resize(
        spatial_cam.astype(np.float32), (W, H),
        interpolation=cv2.INTER_CUBIC,
    )
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    heatmap_bgr   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay_np  = cv2.addWeighted(orig_np, 1 - alpha_overlay, heatmap_rgb, alpha_overlay, 0)
    overlay_pil = Image.fromarray(overlay_np)
    heatmap_b64 = _pil_to_b64(overlay_pil)

    logger.info(
        "[GradCAM] Done — grid=%dx%d, patches=%d, max_score=%.4f",
        grid_size, grid_size, num_patches, float(patch_cam.max()),
    )

    return {
        "technique":    "gradcam",
        "status":       "ok",
        "heatmap_b64":  heatmap_b64,
        "raw_scores":   patch_cam.tolist(),
        "grid_size":    grid_size,
        "target_class": target_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHAP TimeShap — Temporal Frame Attribution
# ─────────────────────────────────────────────────────────────────────────────

def _build_model_fn(model, frame_tensors: list, device: str, baseline: torch.Tensor):
    """
    Build a numpy-callable f(X) → probabilities for KernelSHAP.

    X shape: [n_samples, n_frames] — each entry is 0 (masked) or 1 (present).
    When a frame is masked, it is replaced by `baseline` (mean-frame tensor).
    Returns an array [n_samples, 1] — fake probability for each sample.
    """
    stacked = torch.stack(frame_tensors, dim=0)             # [T, C, H, W]

    def model_fn(coalition_matrix: np.ndarray) -> np.ndarray:
        """coalition_matrix: [n_samples, T]  (float 0/1)."""
        model.eval()
        results = []
        with torch.no_grad():
            for row in coalition_matrix:
                masked_frames = []
                for t, present in enumerate(row):
                    frame_t = stacked[t].to(device)
                    masked_frames.append(frame_t if present > 0.5 else baseline)

                batch = torch.stack(masked_frames, dim=0)   # [T, C, H, W]

                # Run each frame independently through the encoder, then average
                logits_list = []
                for frame in batch:
                    feat   = model.feature_extractor(frame.unsqueeze(0))
                    logits = model.model(feat)               # [1, 2]
                    logits_list.append(logits)

                avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)  # [1, 2]
                probs = torch.softmax(avg_logits, dim=-1)
                results.append(probs[0, 1].item())           # fake probability

        return np.array(results).reshape(-1, 1)

    return model_fn


def _prob_sequence_fn(frame_probs: np.ndarray):
    """
    Lightweight model function when only per-frame probabilities are available.
    Returns a coalition-aware aggregation of the probabilities.

    X: [n_samples, T]  (float 0/1 coalition mask)
    """
    def model_fn(coalition_matrix: np.ndarray) -> np.ndarray:
        results = []
        for row in coalition_matrix:
            active = row > 0.5
            if active.any():
                # Coalition mean of active frame probabilities
                val = frame_probs[active].mean()
            else:
                # All masked → global mean (baseline)
                val = frame_probs.mean()
            results.append(val)
        return np.array(results).reshape(-1, 1)

    return model_fn


def run_timeshap(
    model,
    frame_tensors: Optional[List],
    frame_probs:   Optional[List[float]],
    device: str,
    config: dict,
    num_samples:   int = 256,
) -> dict:
    """
    Compute per-frame Shapley value attributions (SHAP TimeShap).

    Strategy
    ────────
    Priority 1 — `timeshap` library:
        Uses TimeSHAP's `local_report` for rigorous temporal Shapley values.
    Priority 2 — `shap.KernelExplainer`:
        Falls back to KernelSHAP when TimeSHAP is unavailable; treats each
        frame position as a binary coalition feature.
    Priority 3 — Ablation (if neither library is available):
        Leave-one-out (LOO) per-frame attribution using model inference.

    Parameters
    ──────────
    model         : GenD instance.
    frame_tensors : List[Tensor] — preprocessed per-frame tensors. If None,
                    attribution is computed from `frame_probs` only.
    frame_probs   : List[float]  — per-frame fake probabilities from detection.
    device        : 'cuda' | 'cpu'.
    config        : Parsed xai_config.yaml dict.
    num_samples   : Number of coalition samples for KernelSHAP.

    Returns
    ───────
    dict:
        technique      : "timeshap"
        attributions   : List[float] — Shapley value per frame (positive →
                         frame increases fake score, negative → reduces it).
        frame_probs    : List[float] — original per-frame probabilities.
        n_frames       : int
        baseline_prob  : float — prediction with all frames masked (φ_0).
        method         : "timeshap" | "kernelshap" | "loo" | "error"
        status         : "ok" | "error"
        error          : str (only if status == "error")
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    if frame_tensors is None and frame_probs is None:
        return {
            "technique": "timeshap", "status": "error",
            "error": "Both frame_tensors and frame_probs are None.",
            "attributions": [], "frame_probs": [], "n_frames": 0,
            "baseline_prob": 0.0, "method": "error",
        }

    fp_array = np.array(frame_probs, dtype=np.float32) if frame_probs is not None else None
    n_frames = len(frame_tensors) if frame_tensors is not None else len(fp_array)

    if n_frames == 0:
        return {
            "technique": "timeshap", "status": "error",
            "error": "Zero frames provided.",
            "attributions": [], "frame_probs": [], "n_frames": 0,
            "baseline_prob": 0.0, "method": "error",
        }

    logger.info("[TimeShap] Running temporal attribution on %d frames …", n_frames)

    # ── Build baseline tensor (mean of all frame tensors) ─────────────────────
    baseline_tensor = None
    if frame_tensors is not None and len(frame_tensors) > 0:
        stacked  = torch.stack([t.to(device) for t in frame_tensors], dim=0)  # [T, C, H, W]
        baseline_tensor = stacked.mean(dim=0, keepdim=True)                    # [1, C, H, W]

    # Baseline probability (all frames masked → use mean frame tensor)
    if baseline_tensor is not None:
        with torch.no_grad():
            model.eval()
            feat    = model.feature_extractor(baseline_tensor)
            logits  = model.model(feat)
            baseline_prob = torch.softmax(logits, dim=-1)[0, 1].item()
    else:
        baseline_prob = float(fp_array.mean()) if fp_array is not None else 0.5

    # ── Identity input: all frames present ────────────────────────────────────
    full_coalition = np.ones((1, n_frames), dtype=np.float32)

    # ── Strategy 1: timeshap library ─────────────────────────────────────────
    try:
        import timeshap.explainer as ts_explainer
        import timeshap.plot      as ts_plot        # noqa – validates install

        logger.info("[TimeShap] Using `timeshap` library.")

        if frame_tensors is not None:
            model_fn = _build_model_fn(model, frame_tensors, device, baseline_tensor.squeeze(0))
        else:
            model_fn = _prob_sequence_fn(fp_array)

        # TimeSHAP expects: model_fn, data as 2-D array [1, T], baseline vector
        data      = np.ones((1, n_frames), dtype=np.float32)
        baseline  = np.zeros((1, n_frames), dtype=np.float32)  # all-masked coalition

        pruning_dict, shap_dict = ts_explainer.local_report(
            model   = model_fn,
            data    = data,
            baseline= baseline,
            random_seed = 42,
            num_instances = num_samples,
        )
        # timeshap returns a DataFrame; extract values in frame order
        attributions = shap_dict.sort_values("t").set_index("t")["SHAP Value"].tolist()

        return {
            "technique":     "timeshap",
            "status":        "ok",
            "method":        "timeshap",
            "n_frames":      n_frames,
            "attributions":  attributions,
            "frame_probs":   fp_array.tolist() if fp_array is not None else [],
            "baseline_prob": baseline_prob,
        }

    except ImportError:
        logger.info("[TimeShap] `timeshap` not available — falling back to KernelSHAP.")
    except Exception as exc:
        logger.warning("[TimeShap] timeshap.local_report failed (%s) — falling back.", exc)

    # ── Strategy 2: shap.KernelExplainer ─────────────────────────────────────
    try:
        import shap

        logger.info("[TimeShap] Using `shap.KernelExplainer`.")

        if frame_tensors is not None:
            model_fn = _build_model_fn(model, frame_tensors, device, baseline_tensor.squeeze(0))
        else:
            model_fn = _prob_sequence_fn(fp_array)

        # Background = all-masked coalition (Shapley null coalition)
        background = np.zeros((1, n_frames), dtype=np.float32)
        explainer  = shap.KernelExplainer(model_fn, background, silent=True)

        # Explain full coalition (all frames present)
        shap_values = explainer.shap_values(
            full_coalition,
            nsamples = num_samples,
            silent   = True,
        )  # shape: [1, n_frames] or list of those

        # KernelExplainer may return a list (one array per output)
        if isinstance(shap_values, list):
            attrs = shap_values[0][0].tolist()
        else:
            attrs = shap_values[0].tolist()

        return {
            "technique":     "timeshap",
            "status":        "ok",
            "method":        "kernelshap",
            "n_frames":      n_frames,
            "attributions":  attrs,
            "frame_probs":   fp_array.tolist() if fp_array is not None else [],
            "baseline_prob": baseline_prob,
        }

    except ImportError:
        logger.info("[TimeShap] `shap` not available — falling back to LOO attribution.")
    except Exception as exc:
        logger.warning("[TimeShap] KernelExplainer failed (%s) — falling back to LOO.", exc)

    # ── Strategy 3: Leave-One-Out (LOO) ablation ─────────────────────────────
    logger.info("[TimeShap] Using Leave-One-Out (LOO) ablation.")

    try:
        # Full model probability (all frames)
        if frame_tensors is not None:
            model_fn = _build_model_fn(model, frame_tensors, device, baseline_tensor.squeeze(0))
            full_prob = float(model_fn(full_coalition)[0, 0])
        else:
            full_prob = float(fp_array.mean())

        loo_attrs = []
        for i in range(n_frames):
            mask = np.ones((1, n_frames), dtype=np.float32)
            mask[0, i] = 0.0    # mask out frame i

            if frame_tensors is not None:
                without_i = float(model_fn(mask)[0, 0])
            else:
                active = mask[0] > 0.5
                without_i = float(fp_array[active].mean()) if active.any() else baseline_prob

            loo_attrs.append(full_prob - without_i)   # positive → frame i raises fake score

        return {
            "technique":     "timeshap",
            "status":        "ok",
            "method":        "loo",
            "n_frames":      n_frames,
            "attributions":  loo_attrs,
            "frame_probs":   fp_array.tolist() if fp_array is not None else [],
            "baseline_prob": baseline_prob,
        }

    except Exception as exc:
        logger.error("[TimeShap] LOO attribution failed: %s", exc, exc_info=True)
        return {
            "technique":     "timeshap",
            "status":        "error",
            "error":         str(exc),
            "method":        "error",
            "n_frames":      n_frames,
            "attributions":  [],
            "frame_probs":   fp_array.tolist() if fp_array is not None else [],
            "baseline_prob": baseline_prob,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_xai_pipeline(
    model,
    pil_image:     Image.Image,
    device:        str,
    frame_tensors: Optional[List],
    frame_probs:   Optional[List[float]],
    config:        dict,
) -> list:
    """
    Orchestrate all enabled XAI techniques and return a list of result dicts.

    Always runs:
        • Grad-CAM (spatial heatmap over the single representative frame)

    Runs when temporal data is available (frame_tensors or frame_probs):
        • SHAP TimeShap (per-frame Shapley value attribution)

    Parameters
    ──────────
    model         : Loaded GenD instance.
    pil_image     : Representative PIL image (full frame or face crop).
    device        : 'cuda' | 'cpu'.
    frame_tensors : Optional list of pre-processed frame tensors.
    frame_probs   : Optional list of per-frame fake probabilities.
    config        : Parsed xai_config.yaml (currently used for future gates).

    Returns
    ───────
    List[dict] — one result dict per technique, in order:
        [gradcam_result, timeshap_result?]
    """
    results: list = []

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    logger.info("[XAI Pipeline] Running Grad-CAM …")
    try:
        gradcam_result = run_gradcam(
            model        = model,
            pil_image    = pil_image,
            device       = device,
            target_class = 1,                  # always explain "fake" class
            alpha_overlay= 0.45,
        )
    except Exception as exc:
        logger.error("[XAI Pipeline] Grad-CAM raised unhandled exception: %s", exc, exc_info=True)
        gradcam_result = {
            "technique": "gradcam", "status": "error", "error": str(exc),
            "heatmap_b64": "", "raw_scores": [], "grid_size": 0, "target_class": 1,
        }
    results.append(gradcam_result)

    # ── SHAP TimeShap (only when temporal data exists) ────────────────────────
    has_temporal = (
        (frame_tensors is not None and len(frame_tensors) > 0) or
        (frame_probs   is not None and len(frame_probs)   > 0)
    )

    if has_temporal:
        logger.info("[XAI Pipeline] Running SHAP TimeShap …")
        try:
            timeshap_result = run_timeshap(
                model         = model,
                frame_tensors = frame_tensors,
                frame_probs   = frame_probs,
                device        = device,
                config        = config,
                num_samples   = config.get("timeshap_samples", 256),
            )
        except Exception as exc:
            logger.error(
                "[XAI Pipeline] TimeShap raised unhandled exception: %s", exc, exc_info=True
            )
            timeshap_result = {
                "technique": "timeshap", "status": "error", "error": str(exc),
                "method": "error", "n_frames": 0, "attributions": [],
                "frame_probs": frame_probs or [], "baseline_prob": 0.0,
            }
        results.append(timeshap_result)
    else:
        logger.info(
            "[XAI Pipeline] Skipping TimeShap — no frame_tensors or frame_probs provided."
        )

    logger.info("[XAI Pipeline] Completed. Techniques run: %s",
                [r["technique"] for r in results])
    return results
