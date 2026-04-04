# backend/services/pdf_generator/schemas.py
"""
Pydantic schemas for PDF report generation.

Defines the data structures expected from the detection pipelines.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class XAIResultData(BaseModel):
    """XAI visualization data for a single frame."""
    frame_index: int = Field(default=0, description="Frame number in video")
    timestamp: Optional[str] = Field(default=None, description="Timestamp in video")
    gradcam_b64: Optional[str] = Field(default=None, description="Grad-CAM heatmap base64")
    ela_b64: Optional[str] = Field(default=None, description="ELA heatmap base64")
    lime_data: Optional[Dict[str, Any]] = Field(default=None, description="LIME attribution data")
    is_anomaly: bool = Field(default=False, description="Whether anomaly detected")
    fake_prob: float = Field(default=0.0, description="Deepfake probability")
    real_prob: float = Field(default=0.0, description="Real probability")
    confidence: float = Field(default=0.0, description="Confidence score")


class AnomalySegment(BaseModel):
    """Detected anomaly segment in video."""
    start_frame: int = Field(description="Start frame index")
    end_frame: int = Field(description="End frame index")
    start_timestamp: str = Field(description="Start timestamp (MM:SS)")
    end_timestamp: str = Field(description="End timestamp (MM:SS)")
    fake_prob: float = Field(description="Average fake probability")
    severity: str = Field(description="Severity: low, medium, high")


class VideoAnalysisData(BaseModel):
    """Video analysis results from detection pipeline."""
    task_id: str = Field(description="Unique task identifier")
    file_name: str = Field(description="Original video file name")
    total_frames: int = Field(description="Total frames analyzed")
    duration_seconds: float = Field(description="Video duration in seconds")
    verdict: str = Field(description="overall verdict: FAKE or REAL")
    is_fake: bool = Field(description="Whether deepfake detected")
    confidence: float = Field(description="Overall confidence score")
    fake_prob: float = Field(description="Average fake probability")
    real_prob: float = Field(description="Average real probability")
    anomaly_frames: List[int] = Field(default_factory=list, description="List of anomaly frame indices")
    anomaly_segments: List[AnomalySegment] = Field(default_factory=list, description="Detected anomaly segments")
    anomaly_frame_data: List[XAIResultData] = Field(default_factory=list, description="XAI data for anomaly frames")
    detected_type: Optional[str] = Field(default=None, description="Type of manipulation detected")


class ImageAnalysisData(BaseModel):
    """Image analysis results from detection pipeline."""
    task_id: str = Field(description="Unique task identifier")
    file_name: str = Field(description="Original image file name")
    verdict: str = Field(description="FAKE or REAL")
    is_fake: bool = Field(description="Whether deepfake detected")
    confidence: float = Field(description="Confidence score")
    fake_prob: float = Field(description="Deepfake probability")
    real_prob: float = Field(description="Real probability")
    xai_result: Optional[XAIResultData] = Field(default=None, description="XAI visualization data")
    detected_type: Optional[str] = Field(default=None, description="Type of manipulation detected")


class AudioAnalysisData(BaseModel):
    """Audio analysis results from detection pipeline."""
    task_id: str = Field(description="Unique task identifier")
    file_name: str = Field(description="Original audio file name")
    duration_seconds: float = Field(description="Audio duration in seconds")
    verdict: str = Field(description="FAKE or REAL")
    is_fake: bool = Field(description="Whether deepfake detected")
    confidence: float = Field(description="Confidence score")
    fake_prob: float = Field(description="Deepfake probability")
    real_prob: float = Field(description="Real probability")
    waveform_b64: Optional[str] = Field(default=None, description="Waveform visualization base64")
    spectrogram_b64: Optional[str] = Field(default=None, description="Spectrogram visualization base64")
    ig_scores: Optional[List[float]] = Field(default=None, description="Integrated gradients scores")
    shap_scores: Optional[List[float]] = Field(default=None, description="SHAP scores")
    anomaly_timestamps: List[str] = Field(default_factory=list, description="Timestamps with anomalies")
    detected_type: Optional[str] = Field(default=None, description="Type of manipulation detected")


class AnalysisReportData(BaseModel):
    """Complete analysis report data for PDF generation."""
    case_id: str = Field(description="Unique case identifier")
    report_date: datetime = Field(default_factory=datetime.now, description="Report generation date")
    module_type: str = Field(description="Module type: video, image, or audio")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    video_data: Optional[VideoAnalysisData] = Field(default=None, description="Video analysis data")
    image_data: Optional[ImageAnalysisData] = Field(default=None, description="Image analysis data")
    audio_data: Optional[AudioAnalysisData] = Field(default=None, description="Audio analysis data")
    
    executive_summary: str = Field(default="", description="LLM-generated forensic summary")
    technical_findings: str = Field(default="", description="Technical breakdown text")
    
    analyst_name: Optional[str] = Field(default=None, description="Analyst name")
    analyst_notes: Optional[str] = Field(default=None, description="Additional analyst notes")