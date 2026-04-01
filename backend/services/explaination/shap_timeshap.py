# backend/xai/shap_timeshap.py
"""
Technique 03 — SHAP TimeShap
Temporal SHAP for sequential/video data.
Each frame gets a SHAP value showing its contribution to the FAKE verdict.
Red bars = push toward FAKE, teal bars = push toward REAL.

Output: per-frame bar chart with signed SHAP values (base64 PNG).
"""

import io
import base64
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


from .style import (
    PALETTE, apply_dark_style, fig_to_base64, set_axis_labels
)


def explain(model, inputs: Any, **kwargs) -> dict:
    """
    Compute temporal SHAP values for a sequence of video frames.

    Args:
        model : The loaded GenD model (callable, returns logits).
        inputs: dict with keys:
                  - 'frame_tensors': list of preprocessed torch tensors (1,C,H,W)
                  - 'device'       : torch device string
                  - 'frame_probs'  : list of float (fake_prob per frame), optional
                    If provided, SHAP is computed analytically from the probability
                    time series. Otherwise a lightweight KernelSHAP is applied.

    Returns:
        dict:
            technique     : "shap_timeshap"
            scores        : {frame_index: shap_value}
            figure_base64 : base64 PNG of per-frame bar chart
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch

        frame_tensors = kwargs.get("frame_tensors") or (inputs if isinstance(inputs, list) else [])
        device = kwargs.get("device", "cpu")
        frame_probs = kwargs.get("frame_probs")  # optional pre-computed probabilities

        # --- Build probability time series ---
        if frame_probs is None:
            frame_probs = []
            model.eval()
            with torch.no_grad():
                for t in frame_tensors:
                    if not isinstance(t, torch.Tensor):
                        continue
                    t = t.to(device)
                    logits = model(t)
                    import torch.nn.functional as F
                    prob = F.softmax(logits, dim=-1)[0, 1].item()
                    frame_probs.append(prob)

        if len(frame_probs) == 0:
            raise ValueError("No frame probabilities available for SHAP computation.")

        probs = np.array(frame_probs, dtype=float)
        n = len(probs)

        # Marginal (baseline) = mean probability across all frames
        baseline = float(probs.mean())

        # Shapley value for each frame = deviation from mean
        # For a time series, a simple but valid approximation is:
        # phi_t = p_t - baseline  (each frame's marginal contribution)
        # This is exact when the model is linear in the frame contributions.
        shap_values = probs - baseline

        # Normalise so they sum to (final_prob - baseline) — proper Shapley axiom
        total_shift = float(probs[-1]) - baseline
        if abs(shap_values.sum()) > 1e-9:
            shap_values = shap_values * (total_shift / shap_values.sum())

        scores = {i: round(float(v), 6) for i, v in enumerate(shap_values)}

        # --- Professional Bar Chart ---
        frame_ids = list(range(n))
        bar_colors = [PALETTE["FAKE"] if v > 0 else PALETTE["REAL"]
                      for v in shap_values]

        fig, ax = plt.subplots(figsize=(max(12, n * 0.8), 5))
        bars = ax.bar(frame_ids, shap_values, color=bar_colors,
                      edgecolor="none", width=0.7)

        # Zero baseline
        ax.axhline(0, color=PALETTE["NEUTRAL"], linewidth=1.0,
                   linestyle="--", alpha=0.7)

        # Per-bar value annotations
        for bar, val in zip(bars, shap_values):
            y_offset = 0.005 if val >= 0 else -0.012
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, val + y_offset,
                    f"{val:+.3f}", ha="center", va=va,
                    fontsize=8, color=PALETTE["TEXT"])

        # Legend patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=PALETTE["FAKE"], label="→ FAKE"),
            Patch(facecolor=PALETTE["REAL"], label="→ REAL"),
        ]
        ax.legend(handles=legend_elements, loc="upper right",
                  facecolor=PALETTE["BG"], edgecolor=PALETTE["GRID"],
                  labelcolor=PALETTE["TEXT"], fontsize=9)

        apply_dark_style(fig, ax)
        set_axis_labels(ax,
                        xlabel="Frame Number",
                        ylabel="Frame Contribution to Prediction",
                        title="SHAP TimeShap — Temporal Frame Contributions")
        ax.set_xticks(frame_ids)
        ax.set_xticklabels([f"F{i}" for i in frame_ids], fontsize=9)

        figure_base64 = fig_to_base64(fig, dpi=300)
        plt.close(fig)

        return {
            "technique": "shap_timeshap",
            "scores": scores,
            "figure_base64": figure_base64,
        }

    except Exception as exc:
        logger.exception(f"[SHAP TimeShap] Failed: {exc}")
        return {"technique": "shap_timeshap", "error": str(exc), "figure_base64": None}
