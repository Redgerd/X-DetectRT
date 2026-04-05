# backend/services/pdf_generator/__init__.py
"""
PDF Generation Module for Deepfake Detection Reports.

Supports generation of professional forensic PDF reports for:
- Video Analysis
- Image Analysis
- Audio Analysis

Each report includes:
- Executive Summary (LLM-generated)
- Technical Breakdown (anomaly scores, confidence levels)
- XAI Visualizations (Grad-CAM, ELA, spectrograms)
- Forensic Evidence (side-by-side comparisons)
"""

from .generator import PDFGenerator, generate_forensic_report
from .schemas import (
    AnalysisReportData,
    VideoAnalysisData,
    ImageAnalysisData,
    AudioAnalysisData,
    XAIResultData,
)

__all__ = [
    "PDFGenerator",
    "generate_forensic_report",
    "AnalysisReportData",
    "VideoAnalysisData",
    "ImageAnalysisData",
    "AudioAnalysisData",
    "XAIResultData",
]