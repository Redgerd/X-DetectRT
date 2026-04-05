# backend/api/report/schemas.py
"""
Pydantic request/response schemas for the forensic PDF report generation API.
Supports three module types: image, video, audio.
"""

from typing import Optional, Literal, List, Any
from pydantic import BaseModel, Field


# ─── Shared ───────────────────────────────────────────────────────────────────

class StftDataSchema(BaseModel):
    """STFT spectrogram data forwarded from the audio analysis result."""
    matrix: List[List[float]]
    times:  List[float]
    freqs:  List[float]
    db_min: float
    db_max: float


# ─── Module A: Image ──────────────────────────────────────────────────────────

class ImageDataSchema(BaseModel):
    task_id:      Optional[str]   = None
    file_name:    str             = "Unknown"
    verdict:      Optional[str]   = None
    is_fake:      bool            = False
    confidence:   float           = 0.0
    fake_prob:    float           = 0.0
    real_prob:    float           = 0.0
    anomaly_type: Optional[str]   = None
    sha256_hash:  Optional[str]   = None
    # Base64 images (with or without data URI prefix)
    thumbnail_b64: Optional[str]  = None
    gradcam_b64:   Optional[str]  = None
    ela_b64:       Optional[str]  = None
    # JSON XAI payloads
    fft_data:      Optional[Any]  = None
    lime_data:     Optional[Any]  = None


# ─── Module B: Video ──────────────────────────────────────────────────────────

class FlaggedFrameSchema(BaseModel):
    frame_index:  int             = 0
    timestamp:    str             = "00:00:00"
    is_anomaly:   bool            = True
    confidence:   float           = 0.0
    fake_prob:    float           = 0.0
    real_prob:    float           = 0.0
    anomaly_type: Optional[str]   = None
    # Base64 frame image data (raw base64, NO data URI prefix needed)
    frame_data:   Optional[str]   = None
    gradcam_b64:  Optional[str]   = None
    ela_b64:      Optional[str]   = None


class VideoDataSchema(BaseModel):
    task_id:       Optional[str]  = None
    file_name:     str            = "Unknown"
    verdict:       Optional[str]  = None
    is_fake:       bool           = False
    confidence:    float          = 0.0
    fake_prob:     float          = 0.0
    real_prob:     float          = 0.0
    total_frames:  int            = 0
    anomaly_count: int            = 0
    duration_seconds: float       = 0.0
    detected_type: Optional[str]  = None


# ─── Module C: Audio ──────────────────────────────────────────────────────────

class AudioDataSchema(BaseModel):
    task_id:          Optional[str]         = None
    file_name:        str                   = "Unknown"
    verdict:          str                   = "REAL"
    is_fake:          bool                  = False
    confidence:       float                 = 0.0
    fake_prob:        float                 = 0.0
    real_prob:        float                 = 0.0
    duration_seconds: float                 = 0.0


# ─── Unified Report Request ───────────────────────────────────────────────────

class GenerateReportRequest(BaseModel):
    case_id:           Optional[str]                      = None
    module_type:       Literal["image", "video", "audio"]
    executive_summary: Optional[str]                      = None

    # Module-specific payloads (only one should be filled per call)
    image_data:        Optional[ImageDataSchema]          = None
    video_data:        Optional[VideoDataSchema]          = None
    audio_data:        Optional[AudioDataSchema]          = None

    # Video: list of flagged frames with base64 images
    flagged_frames:    Optional[List[FlaggedFrameSchema]] = None

    # Audio: XAI score vectors and spectrogram
    ig_scores:         Optional[List[float]]              = None
    shap_scores:       Optional[List[float]]              = None
    stft:              Optional[StftDataSchema]           = None


# ─── Response ─────────────────────────────────────────────────────────────────

class GenerateReportResponse(BaseModel):
    report_id:  str
    file_path:  str
    module_type: str
    message:    str = "Report generated successfully"
