# backend/api/audio/schemas.py
"""
Pydantic schemas for the Audio Deepfake Detection REST API.

The response now carries raw data arrays instead of pre-rendered PNGs.
The frontend Canvas renders the waveform, STFT spectrogram, and XAI overlays.
"""

from typing import List, Optional
from pydantic import BaseModel


class StftData(BaseModel):
    """2-D STFT power spectrogram payload for the spectrogram Canvas layer."""
    matrix: List[List[float]]   # [freq_bins][time_frames] — dB values
    times:  List[float]         # time axis in seconds (one per frame)
    freqs:  List[float]         # frequency axis in Hz (one per bin)
    db_min: float               # global dB minimum (for colormap range)
    db_max: float               # global dB maximum


class AudioAnalysisResult(BaseModel):
    """
    Full synchronous response returned by POST /audio/analyze.

    Raw data arrays are returned so the React frontend can:
      - render a zoomable waveform with Canvas
      - render an Inferno-colourmap STFT spectrogram
      - overlay IG / SHAP temporal hotspots as glowing contours
      - sync a playhead to the HTML5 <audio> element
    """
    verdict:          str            # "FAKE" | "REAL"
    is_fake:          bool
    confidence:       float          # 0–100
    fake_prob:        float          # 0–1
    real_prob:        float          # 0–1
    duration_seconds: float          # actual audio duration in seconds

    # ── Canvas data layers ────────────────────────────────────────────────────
    waveform_samples: List[float]           # ~2 000 downsampled amplitude points
    stft:             Optional[StftData] = None  # STFT spectrogram

    # ── XAI attribution vectors (one score per STFT time-frame) ──────────────
    ig_scores:   Optional[List[float]] = None   # Integrated Gradients per-frame
    shap_scores: Optional[List[float]] = None   # SHAP KernelExplainer per-frame