# backend/xai/integrated_gradients.py
"""
Technique 08 — Integrated Gradients (Anatomically Masked)
Apply Integrated Gradients using anatomically-informed baselines.
Constrain attribution output per facial zone using binary masks
derived from MediaPipe landmarks.

Key zones: nasolabial folds, hairline boundaries, forehead-eye transition,
eye sockets, jawline, chin, cheeks.

Output: horizontal bar chart of attribution strength per zone.
"""

import io
import base64
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


from .style import (
    PALETTE, apply_dark_style, fig_to_base64,
    styled_barh, set_axis_labels, interpolate_colors
)


def _compute_integrated_gradients(model, input_tensor, baseline_tensor,
                                   target_class=1, steps=50, device="cpu"):
    """
    Compute Integrated Gradients via the Riemann sum approximation.
    Returns the attribution tensor of the same shape as input_tensor.
    """
    import torch

    input_tensor = input_tensor.to(device)
    baseline_tensor = baseline_tensor.to(device)

    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, steps + 1, device=device)
    grads = []

    for alpha in alphas:
        interp = baseline_tensor + alpha * (input_tensor - baseline_tensor)
        interp = interp.detach().requires_grad_(True)
        logits = model(interp)
        score = logits[0, target_class]
        model.zero_grad()
        score.backward()
        grads.append(interp.grad.detach().clone())

    # Trapezoidal integration
    grads = torch.stack(grads, dim=0)   # (steps+1, 1, C, H, W)
    avg_grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = avg_grads.mean(dim=0)   # (1, C, H, W)

    integrated_grads = (input_tensor - baseline_tensor) * avg_grads
    return integrated_grads.squeeze(0)  # (C, H, W)


def _get_mediapipe_masks(img_rgb: np.ndarray):
    """Return dict of zone_name -> binary mask using MediaPipe Face Mesh."""
    from lime_superpixels import FACIAL_ZONES, _get_zone_mask
    try:
        import mediapipe as mp
        h, w = img_rgb.shape[:2]
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.3) as fm:
            results = fm.process(img_rgb)
        if not results.multi_face_landmarks:
            return None
        face_lm = results.multi_face_landmarks[0]
        landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in face_lm.landmark]
        masks = {}
        for zone_name, indices in FACIAL_ZONES.items():
            masks[zone_name] = _get_zone_mask(landmarks_px, indices, h, w)
        return masks
    except Exception as e:
        logger.warning(f"[IG] MediaPipe mask failed: {e}")
        return None


def explain(model, inputs: Any, **kwargs) -> dict:
    """
    Args:
        model : GenD model.
        inputs: PIL.Image of a video frame.
        kwargs:
            device     (str) : torch device
            ig_steps   (int) : IG interpolation steps (default 50)

    Returns:
        dict: technique, scores {zone: attribution_strength}, figure_base64
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
        from PIL import Image

        pil_image = inputs
        if not isinstance(pil_image, Image.Image):
            raise TypeError("inputs must be a PIL.Image")

        device = kwargs.get("device", "cpu")
        steps = kwargs.get("ig_steps", 50)

        img_rgb = np.array(pil_image.convert("RGB"))
        h, w = img_rgb.shape[:2]

        # Preprocess input and baseline (black image)
        input_tensor = model.feature_extractor.preprocess(pil_image).unsqueeze(0)
        black_img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
        baseline_tensor = model.feature_extractor.preprocess(black_img).unsqueeze(0)

        # Enable gradients on model params only temporarily
        was_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)

        attrs = _compute_integrated_gradients(model, input_tensor, baseline_tensor,
                                              target_class=1, steps=steps, device=device)

        # Restore grad state
        for p in model.parameters():
            p.requires_grad_(False)
        if was_training:
            model.train()

        # attrs shape: (C, H, W) — aggregate to single attribution map
        attr_map = attrs.abs().sum(dim=0).cpu().numpy()   # (H, W)

        # Resize attr_map to img_rgb spatial dims if needed
        import cv2
        attr_map_resized = cv2.resize(attr_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Get anatomical masks
        zone_masks = _get_mediapipe_masks(img_rgb)
        if zone_masks is None:
            # Equal-grid fallback: quarter image
            zone_masks = {
                "Top-Left":     np.zeros((h, w), dtype=np.uint8),
                "Top-Right":    np.zeros((h, w), dtype=np.uint8),
                "Bottom-Left":  np.zeros((h, w), dtype=np.uint8),
                "Bottom-Right": np.zeros((h, w), dtype=np.uint8),
            }
            zone_masks["Top-Left"][:h//2, :w//2] = 1
            zone_masks["Top-Right"][:h//2, w//2:] = 1
            zone_masks["Bottom-Left"][h//2:, :w//2] = 1
            zone_masks["Bottom-Right"][h//2:, w//2:] = 1

        scores_raw = {}
        for zone_name, mask in zone_masks.items():
            zone_attrs = attr_map_resized[mask == 1]
            scores_raw[zone_name] = float(zone_attrs.mean()) if zone_attrs.size > 0 else 0.0

        # Normalise to [0, 1]
        vals = np.array(list(scores_raw.values()))
        v_min, v_max = vals.min(), vals.max()
        if v_max - v_min > 1e-8:
            scores = {k: round(float((v - v_min) / (v_max - v_min)), 4)
                      for k, v in scores_raw.items()}
        else:
            scores = {k: 0.0 for k in scores_raw}

        sorted_scores = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

        # --- Professional Horizontal Bar Chart ---
        zones = list(sorted_scores.keys())
        vals_sorted = list(sorted_scores.values())
        bar_colors = interpolate_colors(vals_sorted, PALETTE["ACCENT"], PALETTE["FAKE"])

        row_h = 0.55
        n_rows = len(zones)
        fig_h = max(5, n_rows * (row_h + 0.35) + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_h))

        styled_barh(ax, zones, vals_sorted, bar_colors, show_pct=True)

        apply_dark_style(fig, ax)
        set_axis_labels(ax,
                        xlabel="Attribution Strength (normalised)",
                        ylabel="Facial Zone",
                        title="Integrated Gradients — Anatomical Zone Attribution")

        figure_base64 = fig_to_base64(fig, dpi=300)
        plt.close(fig)

        return {
            "technique": "integrated_gradients",
            "scores": sorted_scores,
            "figure_base64": figure_base64,
        }

    except Exception as exc:
        logger.exception(f"[Integrated Gradients] Failed: {exc}")
        return {"technique": "integrated_gradients", "error": str(exc), "figure_base64": None}
