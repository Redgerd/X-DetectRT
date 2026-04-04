# backend/api/report/routes.py
"""
API Routes for PDF Report Generation.

Provides endpoints to generate forensic PDF reports from detection results.
"""

import os
import uuid
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from services.pdf_generator import generate_forensic_report, AnalysisReportData
from services.pdf_generator.schemas import (
    VideoAnalysisData,
    ImageAnalysisData,
    AudioAnalysisData,
    AnomalySegment,
    XAIResultData,
)

router = APIRouter(prefix="/report", tags=["report"])


class AnomalySegmentRequest(BaseModel):
    """Request model for anomaly segment."""
    start_frame: int
    end_frame: int
    start_timestamp: str
    end_timestamp: str
    fake_prob: float
    severity: str


class XAIResultRequest(BaseModel):
    """Request model for XAI result."""
    frame_index: int
    timestamp: Optional[str] = None
    gradcam_b64: Optional[str] = None
    ela_b64: Optional[str] = None
    is_anomaly: bool = False
    fake_prob: float = 0.0
    real_prob: float = 0.0
    confidence: float = 0.0


class VideoAnalysisRequest(BaseModel):
    """Request model for video analysis report."""
    task_id: str
    file_name: str
    total_frames: int
    duration_seconds: float
    verdict: str
    is_fake: bool
    confidence: float
    fake_prob: float
    real_prob: float
    anomaly_frames: List[int] = []
    anomaly_segments: List[AnomalySegmentRequest] = []
    anomaly_frame_data: List[XAIResultRequest] = []
    detected_type: Optional[str] = None


class ImageAnalysisRequest(BaseModel):
    """Request model for image analysis report."""
    task_id: str
    file_name: str
    verdict: str
    is_fake: bool
    confidence: float
    fake_prob: float
    real_prob: float
    xai_result: Optional[XAIResultRequest] = None
    detected_type: Optional[str] = None


class AudioAnalysisRequest(BaseModel):
    """Request model for audio analysis report."""
    task_id: str
    file_name: str
    duration_seconds: float
    verdict: str
    is_fake: bool
    confidence: float
    fake_prob: float
    real_prob: float
    waveform_b64: Optional[str] = None
    spectrogram_b64: Optional[str] = None
    anomaly_timestamps: List[str] = []
    detected_type: Optional[str] = None


class ReportGenerationRequest(BaseModel):
    """Request model for report generation."""
    case_id: Optional[str] = None
    module_type: str
    executive_summary: str = ""
    analyst_name: Optional[str] = None
    analyst_notes: Optional[str] = None
    
    video_data: Optional[VideoAnalysisRequest] = None
    image_data: Optional[ImageAnalysisRequest] = None
    audio_data: Optional[AudioAnalysisRequest] = None


class ReportGenerationResponse(BaseModel):
    """Response model for report generation."""
    case_id: str
    file_path: str
    message: str


def _convert_video_data(data: VideoAnalysisRequest) -> Optional[VideoAnalysisData]:
    """Convert video request to VideoAnalysisData."""
    if not data:
        return None
    
    segments = [
        AnomalySegment(
            start_frame=s.start_frame,
            end_frame=s.end_frame,
            start_timestamp=s.start_timestamp,
            end_timestamp=s.end_timestamp,
            fake_prob=s.fake_prob,
            severity=s.severity,
        )
        for s in data.anomaly_segments
    ]
    
    frame_data = [
        XAIResultData(
            frame_index=x.frame_index,
            timestamp=x.timestamp,
            gradcam_b64=x.gradcam_b64,
            ela_b64=x.ela_b64,
            is_anomaly=x.is_anomaly,
            fake_prob=x.fake_prob,
            real_prob=x.real_prob,
            confidence=x.confidence,
        )
        for x in data.anomaly_frame_data
    ]
    
    return VideoAnalysisData(
        task_id=data.task_id,
        file_name=data.file_name,
        total_frames=data.total_frames,
        duration_seconds=data.duration_seconds,
        verdict=data.verdict,
        is_fake=data.is_fake,
        confidence=data.confidence,
        fake_prob=data.fake_prob,
        real_prob=data.real_prob,
        anomaly_frames=data.anomaly_frames,
        anomaly_segments=segments,
        anomaly_frame_data=frame_data,
        detected_type=data.detected_type,
    )


def _convert_image_data(data: ImageAnalysisRequest) -> Optional[ImageAnalysisData]:
    """Convert image request to ImageAnalysisData."""
    if not data:
        return None
    
    xai_result = None
    if data.xai_result:
        xai_result = XAIResultData(
            frame_index=data.xai_result.frame_index,
            timestamp=data.xai_result.timestamp,
            gradcam_b64=data.xai_result.gradcam_b64,
            ela_b64=data.xai_result.ela_b64,
            is_anomaly=data.xai_result.is_anomaly,
            fake_prob=data.xai_result.fake_prob,
            real_prob=data.xai_result.real_prob,
            confidence=data.xai_result.confidence,
        )
    
    return ImageAnalysisData(
        task_id=data.task_id,
        file_name=data.file_name,
        verdict=data.verdict,
        is_fake=data.is_fake,
        confidence=data.confidence,
        fake_prob=data.fake_prob,
        real_prob=data.real_prob,
        xai_result=xai_result,
        detected_type=data.detected_type,
    )


def _convert_audio_data(data: AudioAnalysisRequest) -> Optional[AudioAnalysisData]:
    """Convert audio request to AudioAnalysisData."""
    if not data:
        return None
    
    return AudioAnalysisData(
        task_id=data.task_id,
        file_name=data.file_name,
        duration_seconds=data.duration_seconds,
        verdict=data.verdict,
        is_fake=data.is_fake,
        confidence=data.confidence,
        fake_prob=data.fake_prob,
        real_prob=data.real_prob,
        waveform_b64=data.waveform_b64,
        spectrogram_b64=data.spectrogram_b64,
        anomaly_timestamps=data.anomaly_timestamps,
        detected_type=data.detected_type,
    )


@router.post("/generate", response_model=ReportGenerationResponse)
async def generate_report(request: ReportGenerationRequest):
    """Generate a forensic PDF report."""
    case_id = request.case_id or f"CASE-{uuid.uuid4().hex[:8].upper()}"
    
    video_data = _convert_video_data(request.video_data)
    image_data = _convert_image_data(request.image_data)
    audio_data = _convert_audio_data(request.audio_data)
    
    report_data = AnalysisReportData(
        case_id=case_id,
        report_date=datetime.now(),
        module_type=request.module_type,
        executive_summary=request.executive_summary,
        analyst_name=request.analyst_name,
        analyst_notes=request.analyst_notes,
        video_data=video_data,
        image_data=image_data,
        audio_data=audio_data,
    )
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    
    try:
        generate_forensic_report(report_data, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
    
    return ReportGenerationResponse(
        case_id=case_id,
        file_path=output_path,
        message="Report generated successfully"
    )


@router.get("/download/{filename}")
async def download_report(filename: str):
    """Download a generated PDF report."""
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")
    file_path = os.path.join(reports_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        file_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )