# backend/api/report/routes.py
"""
Synaptic Shield — Report Generation API
POST /report/generate  — generate a PDF for image, video, or audio analysis
GET  /report/download/{filename} — download a previously generated PDF
"""

import os
import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .schemas import GenerateReportRequest, GenerateReportResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/report", tags=["Report Generation"])

REPORTS_DIR = "/app/backend/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─── POST /report/generate ────────────────────────────────────────────────────

@router.post("/generate", response_model=GenerateReportResponse)
async def generate_report(payload: GenerateReportRequest):
    """
    Generate a professional forensic PDF report.

    module_type:
      - "image"  → Module A: Single-image deepfake analysis report
      - "video"  → Module B: Video frame-by-frame analysis report
      - "audio"  → Module C: Acoustic deepfake analysis report

    Returns the file path and report ID. Use GET /report/download/{filename}
    to retrieve the actual PDF binary.
    """
    case_id = payload.case_id or f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    module  = payload.module_type

    logger.info(f"[ReportAPI] Generating {module.upper()} report for case: {case_id}")

    try:
        if module == "image":
            file_path, report_id = _generate_image_report(payload, case_id)
        elif module == "video":
            file_path, report_id = _generate_video_report(payload, case_id)
        elif module == "audio":
            file_path, report_id = _generate_audio_report(payload, case_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown module_type: {module}")

        logger.info(f"[ReportAPI] ✅ Generated: {file_path}")
        return GenerateReportResponse(
            report_id=report_id,
            file_path=file_path,
            module_type=module,
            message=f"{module.upper()} forensic report generated successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ReportAPI] ❌ Report generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )


# ─── GET /report/download/{filename} ─────────────────────────────────────────

@router.get("/download/{filename}")
async def download_report(filename: str):
    """
    Download a previously generated forensic PDF report by filename.
    Security: only alphanumeric, dash, underscore, dot allowed in filename.
    """
    import re
    if not re.match(r'^[\w\-. ]+\.pdf$', filename, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Invalid filename format")

    file_path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Report not found: {filename}")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Report-Source": "Synaptic Shield XAI Platform",
        }
    )


# ─── GET /report/list ─────────────────────────────────────────────────────────

@router.get("/list")
async def list_reports():
    """List all generated reports in the reports directory."""
    try:
        files = [
            f for f in os.listdir(REPORTS_DIR)
            if f.lower().endswith(".pdf")
        ]
        files.sort(reverse=True)  # newest first
        return {
            "reports": files,
            "count": len(files),
            "reports_dir": REPORTS_DIR,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _generate_image_report(payload: GenerateReportRequest, case_id: str):
    from services.reports.image_report import ImageForensicReport

    img = payload.image_data
    if img is None:
        img = {}
    else:
        img = img.model_dump()

    data = {
        "case_id":           case_id,
        "executive_summary": payload.executive_summary or "",
        # Flatten image_data fields into root
        **img,
    }

    report = ImageForensicReport(data)
    file_path = report.generate(REPORTS_DIR)
    return file_path, report.report_id


def _generate_video_report(payload: GenerateReportRequest, case_id: str):
    from services.reports.video_report import VideoForensicReport

    vid = payload.video_data
    vid_dict = vid.model_dump() if vid else {}

    # Build flagged_frames list
    flagged = []
    if payload.flagged_frames:
        for f in payload.flagged_frames:
            fd = f.model_dump()
            flagged.append(fd)

    # If anomaly_count not set, derive from flagged frames
    if not vid_dict.get("anomaly_count") and flagged:
        vid_dict["anomaly_count"] = len([f for f in flagged if f.get("is_anomaly", True)])

    data = {
        "case_id":           case_id,
        "executive_summary": payload.executive_summary or "",
        "video_data":        vid_dict,
        "flagged_frames":    flagged,
    }

    report = VideoForensicReport(data)
    file_path = report.generate(REPORTS_DIR)
    return file_path, report.report_id


def _generate_audio_report(payload: GenerateReportRequest, case_id: str):
    from services.reports.audio_report import AudioForensicReport

    aud = payload.audio_data
    aud_dict = aud.model_dump() if aud else {}

    stft_dict = payload.stft.model_dump() if payload.stft else None

    data = {
        "case_id":           case_id,
        "executive_summary": payload.executive_summary or "",
        "audio_data":        aud_dict,
        "ig_scores":         payload.ig_scores   or [],
        "shap_scores":       payload.shap_scores or [],
        "stft":              stft_dict,
    }

    report = AudioForensicReport(data)
    file_path = report.generate(REPORTS_DIR)
    return file_path, report.report_id
