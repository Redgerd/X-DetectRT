# backend/services/pdf_generator/__init__.py
"""
PDF Generation Module for Deepfake Detection Reports.

This module now re-exports the report generator classes directly 
from the real reporting module (`services.reports`) to maintain
backward compatibility if needed.
"""

from ..reports import (
    ImageForensicReport,
    VideoForensicReport,
    AudioForensicReport
)

# If you need to re-export the schemas, they are now consolidated in api.report:
from ...api.report.schemas import (
    ImageDataSchema as ImageAnalysisData,
    VideoDataSchema as VideoAnalysisData,
    AudioDataSchema as AudioAnalysisData,
    GenerateReportRequest as AnalysisReportData
)

__all__ = [
    "ImageForensicReport",
    "VideoForensicReport",
    "AudioForensicReport",
    "ImageAnalysisData",
    "VideoAnalysisData",
    "AudioAnalysisData",
    "AnalysisReportData"
]