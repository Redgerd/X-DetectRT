# backend/xai/counterfactual.py
"""
Technique 10 — Counterfactual Explanations
Find the minimum feature perturbation required to flip prediction FAKE → REAL.
Targets facial features: eye symmetry, skin texture, blink frequency,
hairline edge, jaw sharpness, lip sync, background.

Output: horizontal bar chart of perturbation magnitude per feature.
        Example text: "If eye symmetry were 82% more natural, this would be REAL."
"""

import io
import base64
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

# Feature definitions with their perturbation strategies
COUNTERFACTUAL_FEATURES = {
    "Eye Symmetry":         {"axis": "horizontal_flip_eyes",   "weight": 1.0},
    "Skin Texture":         {"axis": "gaussian_blur",           "weight": 0.8},
    "Hairline Edge":        {"axis": "top_crop_smooth",         "weight": 0.7},
    "Jaw Sharpness":        {"axis": "bottom_blur",             "weight": 0.9},
    "Lip Sync":             {"axis": "mouth_region_blend",      "weight": 0.85},
    "Background Coherence": {"axis": "background_normalize",    "weight": 0.6},
    "Blink Frequency":      {"axis": "eye_region_interpolate",  "weight": 0.75},
}


def _apply_perturbation(img_rgb: np.ndarray, feature: str, strength: float) -> np.ndarray:
    """Apply a perturbation to the image for the given feature at given strength."""
    import cv2
    h, w = img_rgb.shape[:2]
    perturbed = img_rgb.copy().astype(np.float32)
    strategy = COUNTERFACTUAL_FEATURES[feature]["axis"]

    if strategy == "horizontal_flip_eyes":
        # Flip top half horizontally and blend
        top = perturbed[:h // 3, :, :]
        top_flipped = top[:, ::-1, :]
        perturbed[:h // 3, :, :] = top * (1 - strength) + top_flipped * strength

    elif strategy == "gaussian_blur":
        # Gaussian blur entire skin texture
        ksize = max(1, int(strength * 15)) | 1  # must be odd
        blur = cv2.GaussianBlur(perturbed, (ksize, ksize), 0)
        perturbed = perturbed * (1 - strength) + blur * strength

    elif strategy == "top_crop_smooth":
        # Smooth hairline area (top 15%)
        y_end = int(h * 0.15)
        region = perturbed[:y_end, :, :]
        blur = cv2.GaussianBlur(region, (21, 21), 0)
        perturbed[:y_end, :, :] = region * (1 - strength) + blur * strength

    elif strategy == "bottom_blur":
        # Blur lower jawline area
        y_start = int(h * 0.75)
        region = perturbed[y_start:, :, :]
        ksize = max(1, int(strength * 11)) | 1
        blur = cv2.GaussianBlur(region, (ksize, ksize), 0)
        perturbed[y_start:, :, :] = region * (1 - strength) + blur * strength

    elif strategy == "mouth_region_blend":
        # Blend mouth region toward 128 (neutral)
        y0, y1 = int(h * 0.65), int(h * 0.85)
        perturbed[y0:y1, :, :] = (perturbed[y0:y1, :, :] * (1 - strength)
                                   + 128 * strength)

    elif strategy == "background_normalize":
        # Normalise background colour
        perturbed = perturbed * (1 - strength * 0.3) + perturbed.mean() * strength * 0.3

    elif strategy == "eye_region_interpolate":
        # Interpolate eye region to mean brightness
        y0, y1 = int(h * 0.2), int(h * 0.45)
        eye_mean = perturbed[y0:y1, :, :].mean()
        perturbed[y0:y1, :, :] = (perturbed[y0:y1, :, :] * (1 - strength)
                                   + eye_mean * strength)

    return np.clip(perturbed, 0, 255).astype(np.uint8)


from .style import (
    PALETTE, apply_dark_style, fig_to_base64,
    styled_barh, set_axis_labels, interpolate_colors
)


def explain(model, inputs: Any, **kwargs) -> dict:
    """
    Args:
        model : GenD model.
        inputs: PIL.Image of a video frame.
        kwargs:
            device         (str)   : torch device
            target_prob    (float) : decision threshold to cross (default 0.5)
            n_steps        (int)   : binary search steps per feature (default 20)

    Returns:
        dict: technique, scores {feature: perturbation_magnitude},
              narrative (list of strings), figure_base64
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
        import torch.nn.functional as F
        from PIL import Image

        pil_image = inputs
        if not isinstance(pil_image, Image.Image):
            raise TypeError("inputs must be a PIL.Image")

        device = kwargs.get("device", "cpu")
        target_prob = kwargs.get("target_prob", 0.5)
        n_steps = kwargs.get("n_steps", 20)

        img_rgb = np.array(pil_image.convert("RGB"))

        model.eval()

        def _predict(img_arr: np.ndarray) -> float:
            pil = Image.fromarray(img_arr.astype(np.uint8))
            with torch.no_grad():
                t = model.feature_extractor.preprocess(pil).unsqueeze(0).to(device)
                logits = model(t)
                return F.softmax(logits, dim=-1)[0, 1].item()

        baseline_fake_prob = _predict(img_rgb)

        scores = {}
        narrative = []

        for feature_name in COUNTERFACTUAL_FEATURES:
            # Binary search for minimum strength to flip prediction
            lo, hi = 0.0, 1.0
            found_strength = 1.0
            for _ in range(n_steps):
                mid = (lo + hi) / 2.0
                perturbed = _apply_perturbation(img_rgb, feature_name, mid)
                new_prob = _predict(perturbed)
                if new_prob < target_prob:
                    found_strength = mid
                    hi = mid
                else:
                    lo = mid

            scores[feature_name] = round(found_strength, 4)
            pct_improvement = round((1.0 - found_strength) * 100, 1)
            narrative.append(
                f"If '{feature_name}' were {pct_improvement:.0f}% more natural, "
                f"this video would classify as REAL."
            )

        # Sort by lowest perturbation (most impactful feature first)
        sorted_scores = dict(sorted(scores.items(), key=lambda kv: kv[1]))

        # --- Professional Horizontal Bar Chart ---
        features = list(sorted_scores.keys())
        magnitudes = list(sorted_scores.values())

        # Low perturbation (most impactful) → FAKE orange; high → REAL teal
        # Invert mapping: most impactful feature has smallest magnitude
        inverted = [1.0 - m for m in magnitudes]
        bar_colors = interpolate_colors(inverted, PALETTE["REAL"], PALETTE["FAKE"])

        row_h = 0.55
        n_rows = len(features)
        fig_h = max(5, n_rows * (row_h + 0.35) + 1.8)
        fig, ax = plt.subplots(figsize=(10, fig_h))

        styled_barh(ax, features, magnitudes, bar_colors)

        # Threshold line at 0.5
        ax.axvline(0.5, color=PALETTE["NEUTRAL"], linewidth=1.0,
                   linestyle=":", alpha=0.6,
                   label="Mid perturbation (0.5)")

        # Narrative box — top-3 sentences
        top_narrative = "\n".join(narrative[:3])
        fig.text(0.5, -0.03, top_narrative,
                 ha="center", va="top", fontsize=8,
                 color=PALETTE["NEUTRAL"],
                 wrap=True, transform=fig.transFigure)

        apply_dark_style(fig, ax)
        set_axis_labels(ax,
                        xlabel="Minimum Perturbation Magnitude  (0 = easy flip, 1 = hard)",
                        ylabel="Facial Feature",
                        title="Counterfactual Explanations — Minimum Feature Perturbations")

        figure_base64 = fig_to_base64(fig, dpi=300)
        plt.close(fig)

        return {
            "technique": "counterfactual",
            "scores": sorted_scores,
            "narrative": narrative,
            "baseline_fake_prob": round(baseline_fake_prob, 4),
            "figure_base64": figure_base64,
        }

    except Exception as exc:
        logger.exception(f"[Counterfactual] Failed: {exc}")
        return {"technique": "counterfactual", "error": str(exc), "figure_base64": None}
