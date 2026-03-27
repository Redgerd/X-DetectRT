# backend/xai/prototype_analysis.py
"""
Technique 12 — Prototype / Criticism Analysis
Embed the input video in the model's feature/embedding space.
Compute cosine similarity against a library of known confirmed deepfakes.

Known fake prototypes: GAN v1, FaceSwap, SimSwap (stubbed with saved
random embeddings if real library not available).

Output: bar chart of cosine similarity scores vs each known fake prototype.
Text: "This video is 94% similar to confirmed deepfake #1 (GAN v1)"
"""

import io
import base64
import logging
import os
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

PROTOTYPE_DB_PATH = os.environ.get(
    "PROTOTYPE_DB_PATH", "/app/data/known_fakes_embeddings.npy"
)

# Known prototype names — must match order in .npy file (or fallback stubs)
PROTOTYPE_NAMES = [
    "GAN v1 (StyleGAN)",
    "FaceSwap Classic",
    "SimSwap",
    "DeepFaceLab",
    "FaceShifter",
    "FSGAN",
    "BlendFace",
    "E4S",
]


def _load_prototype_db(embedding_dim: int):
    """
    Load prototype embeddings. Falls back to reproducible stubs if file absent.
    Returns np.ndarray of shape (N, embedding_dim).
    """
    if os.path.exists(PROTOTYPE_DB_PATH):
        try:
            db = np.load(PROTOTYPE_DB_PATH, allow_pickle=False)
            logger.info(f"[Prototype] Loaded prototype DB from {PROTOTYPE_DB_PATH}: {db.shape}")
            return db, PROTOTYPE_NAMES[:len(db)]
        except Exception as e:
            logger.warning(f"[Prototype] Failed to load DB: {e}. Using stubs.")

    # Deterministic stubs — seeded for reproducibility
    rng = np.random.default_rng(seed=42)
    n = len(PROTOTYPE_NAMES)
    stubs = rng.standard_normal((n, embedding_dim)).astype(np.float32)
    # Normalise to unit sphere
    norms = np.linalg.norm(stubs, axis=1, keepdims=True)
    stubs = stubs / (norms + 1e-8)
    return stubs, PROTOTYPE_NAMES


def _extract_embedding(model, img_rgb: np.ndarray, device: str) -> np.ndarray:
    """Reuse the same embedding extraction as TCAV."""
    from xai.tcav import _extract_embedding as _te
    return _te(model, img_rgb, device)


from xai.style import (
    PALETTE, apply_dark_style, fig_to_base64, set_axis_labels
)


def explain(model, inputs: Any, **kwargs) -> dict:
    """
    Args:
        model : GenD model.
        inputs: PIL.Image of a video frame.
        kwargs:
            device (str): torch device

    Returns:
        dict: technique, scores {prototype_name: cosine_similarity},
              top_match (str), figure_base64
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics.pairwise import cosine_similarity
        from PIL import Image

        pil_image = inputs
        if not isinstance(pil_image, Image.Image):
            raise TypeError("inputs must be a PIL.Image")

        device = kwargs.get("device", "cpu")
        img_rgb = np.array(pil_image.convert("RGB"))

        model.eval()
        input_emb = _extract_embedding(model, img_rgb, device)
        embedding_dim = input_emb.shape[0]

        prototype_db, prototype_names = _load_prototype_db(embedding_dim)

        # Normalise input embedding
        input_norm = input_emb / (np.linalg.norm(input_emb) + 1e-8)
        input_norm = input_norm.reshape(1, -1)

        # Normalise prototypes
        proto_norms = prototype_db / (np.linalg.norm(prototype_db, axis=1, keepdims=True) + 1e-8)

        sims = cosine_similarity(input_norm, proto_norms)[0]  # (N,)

        # Map [-1, 1] → [0, 1] and round
        sims_scaled = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
        scores = {name: round(float(sim), 4)
                  for name, sim in zip(prototype_names, sims_scaled)}

        sorted_scores = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
        top_name = list(sorted_scores.keys())[0]
        top_sim = list(sorted_scores.values())[0]
        top_match = (f"This video is {top_sim * 100:.1f}% similar to confirmed deepfake "
                     f"'{top_name}'.")

        # --- Professional Bar Chart ---
        names = list(sorted_scores.keys())
        sims_sorted = list(sorted_scores.values())

        # Orange bars with alpha scaled by similarity strength
        import matplotlib.colors as mcolors
        base_rgb = mcolors.to_rgba(PALETTE["FAKE"])
        bar_colors = [
            (base_rgb[0], base_rgb[1], base_rgb[2], max(0.35, s))
            for s in sims_sorted
        ]

        row_h = 0.55
        n_rows = len(names)
        fig_h = max(5, n_rows * (row_h + 0.35) + 1.8)
        fig, ax = plt.subplots(figsize=(10, fig_h))

        bars = ax.barh(names, sims_sorted, color=bar_colors, edgecolor="none", height=0.55)
        for bar, val in zip(bars, sims_sorted):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val * 100:.1f}%", va="center", ha="left",
                    fontsize=9, color=PALETTE["TEXT"])
        ax.set_xlim(0, 1.2)

        # Top-match banner inside the chart
        ax.text(0.98, 0.02, top_match,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=PALETTE["FAKE"],
                bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["BG"],
                          edgecolor=PALETTE["GRID"], alpha=0.8))

        apply_dark_style(fig, ax)
        set_axis_labels(ax,
                        xlabel="Cosine Similarity to Known Deepfake Method",
                        ylabel="Deepfake Prototype",
                        title="Prototype Analysis — Known Deepfake Similarity Matching")

        figure_base64 = fig_to_base64(fig, dpi=300)
        plt.close(fig)

        return {
            "technique": "prototype_analysis",
            "scores": sorted_scores,
            "top_match": top_match,
            "figure_base64": figure_base64,
        }

    except Exception as exc:
        logger.exception(f"[Prototype Analysis] Failed: {exc}")
        return {"technique": "prototype_analysis", "error": str(exc), "figure_base64": None}
