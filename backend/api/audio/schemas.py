# backend/api/audio/schemas.py
"""
Pydantic schemas for the Audio Deepfake Detection API.
"""

from typing import Optional
from pydantic import BaseModel


class AudioAnalysisResponse(BaseModel):
    """Response returned immediately after submitting an audio file for analysis."""
    task_id: str
    celery_task_id: str
    message: str


class AudioDetectionResult(BaseModel):
    """Detection verdict and probabilities (available after the detection Celery task finishes)."""
    task_id: str
    verdict: str                    # "FAKE" or "REAL"
    is_fake: bool
    confidence: float               # 0–100
    fake_prob: float                # 0–1
    real_prob: float                # 0–1
    waveform_b64: Optional[str] = None   # base64 PNG of raw waveform


class AudioXAIResult(BaseModel):
    """XAI heatmaps (available after the XAI Celery task finishes)."""
    task_id: str
    ig_heatmap_b64: Optional[str] = None    # Integrated Gradients temporal heatmap
    shap_heatmap_b64: Optional[str] = None  # SHAP temporal heatmap
    fake_prob: float
    real_prob: float


class AudioResultResponse(BaseModel):
    """
    Combined polling response — merges detection result and XAI result.
    XAI fields will be None if the XAI task has not completed yet.
    """
    task_id: str
    status: str                             # "pending" | "detection_complete" | "complete"
    verdict: Optional[str] = None
    is_fake: Optional[bool] = None
    confidence: Optional[float] = None
    fake_prob: Optional[float] = None
    real_prob: Optional[float] = None
    waveform_b64: Optional[str] = None
    ig_heatmap_b64: Optional[str] = None
    shap_heatmap_b64: Optional[str] = None
