# backend/xai/lime_superpixels.py
"""
Technique 07 — LIME with Facial Superpixels
Use MediaPipe to extract anatomical facial landmarks and define superpixels
based on facial zones (eyes, nose, jaw, lips, hairline, ears, cheeks,
forehead, skin texture, nasolabial folds).
Apply LIME perturbation on these anatomical superpixels.

Output: horizontal bar chart of importance scores per facial zone (0.0–1.0).
"""

import io
import base64
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

# Anatomical facial zone definitions (MediaPipe 468-landmark indices)
# Based on MediaPipe Face Mesh canonical map
FACIAL_ZONES = {
    "Left Eye":          [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "Right Eye":         [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "Nose":              [1, 2, 5, 4, 6, 168, 197, 195, 5, 64, 98, 97, 2, 326, 327, 294],
    "Lips":              [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    "Jaw & Chin":        [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
    "Left Cheek":        [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365],
    "Right Cheek":       [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136],
    "Forehead":          [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378],
    "Hairline":          [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377],
    "Nasolabial Folds":  [92, 165, 167, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106],
}


def _get_zone_mask(landmarks_px, zone_indices, img_h, img_w):
    """Create a binary mask for a set of landmark indices via convex hull."""
    import cv2
    pts = []
    for idx in zone_indices:
        if idx < len(landmarks_px):
            pts.append(landmarks_px[idx])
    if len(pts) < 3:
        return np.zeros((img_h, img_w), dtype=np.uint8)
    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


def _build_zone_segment_map(img_rgb: np.ndarray):
    """
    Run MediaPipe Face Mesh and return a per-pixel zone-label array.
    Pixels not assigned to any zone get label 0 ("Background").
    """
    try:
        import mediapipe as mp
        import cv2

        h, w = img_rgb.shape[:2]
        segment_map = np.zeros((h, w), dtype=int)  # 0 = background

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.3) as face_mesh:
            results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            logger.warning("[LIME] No face landmarks detected — using SLIC fallback.")
            return None, None

        face_lm = results.multi_face_landmarks[0]
        landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in face_lm.landmark]

        zone_labels = {}  # zone_name -> int label
        for label_id, (zone_name, indices) in enumerate(FACIAL_ZONES.items(), start=1):
            mask = _get_zone_mask(landmarks_px, indices, h, w)
            segment_map[mask == 1] = label_id
            zone_labels[label_id] = zone_name

        return segment_map, zone_labels

    except Exception as e:
        logger.warning(f"[LIME] MediaPipe zone detection failed: {e}")
        return None, None


def _slic_fallback(img_rgb: np.ndarray, n_segments=10):
    """SLIC superpixels fallback when MediaPipe unavailable."""
    from skimage.segmentation import slic
    segments = slic(img_rgb, n_segments=n_segments, compactness=10, start_label=1)
    zone_labels = {i: f"Region {i}" for i in np.unique(segments)}
    return segments, zone_labels


from .style import (
    PALETTE, apply_dark_style, fig_to_base64,
    styled_barh, set_axis_labels, interpolate_colors
)


def explain(model, inputs: Any, **kwargs) -> dict:
    """
    Args:
        model : GenD model (callable).
        inputs: PIL.Image of the face frame.
        kwargs:
            device       (str)  : torch device
            num_samples  (int)  : LIME perturbation samples (default 300)

    Returns:
        dict: technique, scores {zone: importance}, figure_base64
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
        num_samples = kwargs.get("num_samples", 300)

        img_rgb = np.array(pil_image.convert("RGB"))
        h, w = img_rgb.shape[:2]

        # --- Build anatomical segment map ---
        segment_map, zone_labels = _build_zone_segment_map(img_rgb)
        if segment_map is None:
            segment_map, zone_labels = _slic_fallback(img_rgb)

        unique_labels = [l for l in np.unique(segment_map) if l != 0]
        n_zones = len(unique_labels)

        if n_zones == 0:
            raise ValueError("No zones found in segment map.")

        # --- LIME perturbation ---
        # Each sample: random binary mask over zones → perturbed image → model prob
        model.eval()
        probs_list = []
        masks_list = []

        for _ in range(num_samples):
            zone_on = np.random.randint(0, 2, size=n_zones)
            masks_list.append(zone_on)

            perturbed = img_rgb.copy()
            for i, label in enumerate(unique_labels):
                if zone_on[i] == 0:
                    perturbed[segment_map == label] = 127  # grey baseline

            pil_perturbed = Image.fromarray(perturbed.astype(np.uint8))
            with torch.no_grad():
                tensor = model.feature_extractor.preprocess(pil_perturbed).unsqueeze(0).to(device)
                logits = model(tensor)
                import torch.nn.functional as F
                prob_fake = F.softmax(logits, dim=-1)[0, 1].item()
            probs_list.append(prob_fake)

        X = np.array(masks_list, dtype=float)   # (num_samples, n_zones)
        y = np.array(probs_list, dtype=float)    # (num_samples,)

        # Fit weighted ridge regression (LIME kernel: exp distance)
        distances = np.sqrt(((X - 1.0) ** 2).sum(axis=1))
        kernel_width = np.sqrt(n_zones) * 0.75
        weights = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))

        # Weighted least squares
        W = np.diag(weights)
        XtW = X.T @ W
        coeffs = np.linalg.lstsq(XtW @ X + 1e-4 * np.eye(n_zones), XtW @ y, rcond=None)[0]

        # Normalise to [0, 1]
        c_min, c_max = coeffs.min(), coeffs.max()
        if c_max - c_min > 1e-8:
            coeffs_norm = (coeffs - c_min) / (c_max - c_min)
        else:
            coeffs_norm = np.zeros_like(coeffs)

        scores = {}
        for i, label in enumerate(unique_labels):
            zone_name = zone_labels.get(label, f"Zone {label}")
            scores[zone_name] = round(float(coeffs_norm[i]), 4)

        # Sort by importance descending
        sorted_scores = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

        # --- Professional Horizontal Bar Chart ---
        zones = list(sorted_scores.keys())
        importances = list(sorted_scores.values())
        bar_colors = interpolate_colors(importances, PALETTE["REAL"], PALETTE["FAKE"])

        row_h = 0.55
        n_rows = len(zones)
        fig_h = max(5, n_rows * (row_h + 0.35) + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_h))

        styled_barh(ax, zones, importances, bar_colors, show_pct=True)

        apply_dark_style(fig, ax)
        set_axis_labels(ax,
                        xlabel="Importance Score",
                        ylabel="Facial Zone",
                        title="LIME — Facial Zone Importance (Superpixel Attribution)")

        figure_base64 = fig_to_base64(fig, dpi=300)
        plt.close(fig)

        return {
            "technique": "lime_superpixels",
            "scores": sorted_scores,
            "figure_base64": figure_base64,
        }

    except Exception as exc:
        logger.exception(f"[LIME Superpixels] Failed: {exc}")
        return {"technique": "lime_superpixels", "error": str(exc), "figure_base64": None}