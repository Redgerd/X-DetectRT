# backend/services/pdf_generator/generator.py
"""
Professional PDF Report Generator for Deepfake Detection.

Generates forensic-style PDF reports with:
- Professional header with case ID and detection date
- Executive summary from LLM analysis
- Technical breakdown with anomaly scores
- Forensic evidence with XAI visualizations
- Module-specific sections (video timeline, audio spectrogram)
"""

import base64
import io
import logging
from datetime import datetime
from typing import Optional, List, Tuple

from fpdf import FPDF
from PIL import Image

from .schemas import (
    AnalysisReportData,
    VideoAnalysisData,
    ImageAnalysisData,
    AudioAnalysisData,
    XAIResultData,
)

logger = logging.getLogger(__name__)


class PDFGenerator:
    """Generates professional forensic PDF reports for deepfake detection."""

    PAGE_WIDTH = 210
    PAGE_HEIGHT = 297
    MARGIN = 15
    CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
    
    TITLE_COLOR = (31, 41, 59)
    HEADER_COLOR = (59, 130, 246)
    DANGER_COLOR = (220, 38, 38)
    SUCCESS_COLOR = (22, 163, 74)
    WARNING_COLOR = (217, 119, 6)
    TABLE_HEADER_BG = (243, 244, 246)
    TABLE_BORDER = (156, 163, 175)

    def __init__(self):
        self.pdf = None

    def generate(self, data: AnalysisReportData, output_path: str) -> str:
        """Generate PDF report and save to output_path."""
        self.pdf = FPDF(unit="mm", format="A4")
        self.pdf.set_auto_page_break(auto=True, margin=15)
        
        self._add_header_section(data)
        
        if data.module_type == "video":
            self._add_video_report(data.video_data)
        elif data.module_type == "image":
            self._add_image_report(data.image_data)
        elif data.module_type == "audio":
            self._add_audio_report(data.audio_data)
        
        self._add_footer()
        
        self.pdf.output(output_path)
        logger.info(f"PDF report generated: {output_path}")
        return output_path

    def _add_header_section(self, data: AnalysisReportData) -> None:
        """Add professional header with title, case ID, and date."""
        self.pdf.add_page()
        
        self.pdf.set_fill_color(*self.HEADER_COLOR)
        self.pdf.rect(0, 0, self.PAGE_WIDTH, 40, "F")
        
        self.pdf.set_font("Helvetica", "B", 24)
        self.pdf.set_text_color(255, 255, 255)
        self.pdf.cell(0, 15, "DEEPFAKE DETECTION REPORT", align="C", ln=True)
        
        self.pdf.set_font("Helvetica", "", 12)
        self.pdf.ln(8)
        self.pdf.cell(0, 8, f"Case ID: {data.case_id}", align="C", ln=True)
        
        self.pdf.set_xy(self.MARGIN, 45)
        self.pdf.set_text_color(*self.TITLE_COLOR)
        
        info_y = self.pdf.get_y()
        self.pdf.set_font("Helvetica", "B", 11)
        self.pdf.cell(self.CONTENT_WIDTH / 2, 6, f"Module Type: {data.module_type.upper()}")
        self.pdf.set_font("Helvetica", "", 11)
        self.pdf.cell(self.CONTENT_WIDTH / 2, 6, f"Report Date: {data.report_date.strftime('%Y-%m-%d %H:%M:%S')}", align="R", ln=True)
        
        self.pdf.set_line_width(0.5)
        self.pdf.line(self.MARGIN, info_y + 8, self.PAGE_WIDTH - self.MARGIN, info_y + 8)
        self.pdf.ln(12)

    def _add_executive_summary(self, summary: str) -> None:
        """Add executive summary section."""
        self.pdf.set_font("Helvetica", "B", 14)
        self.pdf.set_text_color(*self.TITLE_COLOR)
        self.pdf.cell(0, 8, "EXECUTIVE SUMMARY", ln=True)
        
        self.pdf.ln(2)
        self.pdf.set_line_width(0.3)
        self.pdf.line(self.MARGIN, self.pdf.get_y(), self.PAGE_WIDTH - self.MARGIN, self.pdf.get_y())
        self.pdf.ln(4)
        
        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.set_text_color(55, 65, 81)
        
        summary_lines = self._wrap_text(summary, self.CONTENT_WIDTH, 10)
        for line in summary_lines:
            self.pdf.multi_cell(self.CONTENT_WIDTH, 5, line)
        
        self.pdf.ln(8)

    def _add_verdict_box(self, verdict: str, confidence: float, is_fake: bool) -> None:
        """Add verdict highlight box."""
        box_y = self.pdf.get_y()
        
        if is_fake:
            bg_color = (254, 226, 226)
            text_color = self.DANGER_COLOR
            verdict_text = "DEEPFAKE DETECTED"
        else:
            bg_color = (220, 252, 231)
            text_color = self.SUCCESS_COLOR
            verdict_text = "AUTHENTIC CONTENT"
        
        self.pdf.set_fill_color(*bg_color)
        self.pdf.rect(self.MARGIN, box_y, self.CONTENT_WIDTH, 20, "F")
        
        self.pdf.set_xy(self.MARGIN + 5, box_y + 3)
        self.pdf.set_font("Helvetica", "B", 14)
        self.pdf.set_text_color(*text_color)
        self.pdf.cell(self.CONTENT_WIDTH - 10, 6, verdict_text, ln=True)
        
        self.pdf.set_font("Helvetica", "", 11)
        self.pdf.set_text_color(55, 65, 81)
        self.pdf.cell(self.CONTENT_WIDTH - 10, 5, f"Confidence: {confidence:.1f}%", ln=True)
        
        self.pdf.ln(25)

    def _add_video_report(self, data: Optional[VideoAnalysisData]) -> None:
        """Add video-specific report sections."""
        if not data:
            return
        
        self._add_executive_summary(data.task_id or "Video analysis complete")
        self._add_verdict_box(data.verdict, data.confidence, data.is_fake)
        
        self._add_technical_breakdown_video(data)
        
        if data.anomaly_frame_data:
            self._add_forensic_evidence_video(data)
        
        if data.anomaly_segments:
            self._add_anomaly_timeline(data)
        
        self.pdf.ln(10)

    def _add_image_report(self, data: Optional[ImageAnalysisData]) -> None:
        """Add image-specific report sections."""
        if not data:
            return
        
        self._add_executive_summary(data.file_name or "Image analysis complete")
        self._add_verdict_box(data.verdict, data.confidence, data.is_fake)
        
        self._add_technical_breakdown_image(data)
        
        if data.xai_result:
            self._add_forensic_evidence_image(data.xai_result)
        
        self.pdf.ln(10)

    def _add_audio_report(self, data: Optional[AudioAnalysisData]) -> None:
        """Add audio-specific report sections."""
        if not data:
            return
        
        self._add_executive_summary(data.file_name or "Audio analysis complete")
        self._add_verdict_box(data.verdict, data.confidence, data.is_fake)
        
        self._add_technical_breakdown_audio(data)
        
        if data.waveform_b64 or data.spectrogram_b64:
            self._add_audio_visualizations(data)
        
        if data.anomaly_timestamps:
            self._add_audio_anomaly_timestamps(data)
        
        self.pdf.ln(10)

    def _add_technical_breakdown_video(self, data: VideoAnalysisData) -> None:
        """Add technical breakdown table for video analysis."""
        self.pdf.set_font("Helvetica", "B", 12)
        self.pdf.set_text_color(*self.TITLE_COLOR)
        self.pdf.cell(0, 8, "TECHNICAL BREAKDOWN", ln=True)
        self.pdf.ln(3)
        
        col_widths = [50, 40, 40, self.CONTENT_WIDTH - 130]
        headers = ["Metric", "Value", "Score", "Notes"]
        
        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.set_fill_color(*self.TABLE_HEADER_BG)
        
        x_start = self.MARGIN
        for i, header in enumerate(headers):
            self.pdf.cell(col_widths[i], 7, header, border=1, fill=True)
        self.pdf.ln()
        
        self.pdf.set_font("Helvetica", "", 9)
        self.pdf.set_text_color(55, 65, 81)
        
        metrics = [
            ("File Name", data.file_name, "", ""),
            ("Total Frames", str(data.total_frames), "", ""),
            ("Duration", f"{data.duration_seconds:.2f}s", "", ""),
            ("Anomaly Frames", str(len(data.anomaly_frames)), "", f"{len(data.anomaly_frames) / data.total_frames * 100:.1f}% of total"),
            ("Fake Probability", "", f"{data.fake_prob:.2%}", "Mean across frames"),
            ("Real Probability", "", f"{data.real_prob:.2%}", "Mean across frames"),
            ("Detection Type", data.detected_type or "N/A", "", ""),
        ]
        
        for i, (metric, value, score, notes) in enumerate(metrics):
            if i % 2 == 1:
                self.pdf.set_fill_color(249, 250, 251)
            else:
                self.pdf.set_fill_color(255, 255, 255)
            
            for j, content in enumerate([metric, value, score, notes]):
                self.pdf.cell(col_widths[j], 6, content, border=1, fill=True)
            self.pdf.ln()
        
        self.pdf.ln(10)

    def _add_technical_breakdown_image(self, data: ImageAnalysisData) -> None:
        """Add technical breakdown table for image analysis."""
        self.pdf.set_font("Helvetica", "B", 12)
        self.pdf.set_text_color(*self.TITLE_COLOR)
        self.pdf.cell(0, 8, "TECHNICAL BREAKDOWN", ln=True)
        self.pdf.ln(3)
        
        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.set_fill_color(*self.TABLE_HEADER_BG)
        
        col_widths = [60, 50, self.CONTENT_WIDTH - 110]
        for header in ["Metric", "Value", "Notes"]:
            self.pdf.cell(col_widths.pop(0), 7, header, border=1, fill=True)
        self.pdf.ln()
        
        self.pdf.set_font("Helvetica", "", 9)
        self.pdf.set_text_color(55, 65, 81)
        
        metrics = [
            ("File Name", data.file_name, ""),
            ("Verdict", data.verdict, ""),
            ("Confidence", f"{data.confidence:.1f}%", ""),
            ("Fake Probability", f"{data.fake_prob:.4f}", ""),
            ("Real Probability", f"{data.real_prob:.4f}", ""),
            ("Detection Type", data.detected_type or "N/A", ""),
        ]
        
        for i, (metric, value, notes) in enumerate(metrics):
            if i % 2 == 1:
                self.pdf.set_fill_color(249, 250, 251)
            else:
                self.pdf.set_fill_color(255, 255, 255)
            
            self.pdf.cell(60, 6, metric, border=1, fill=True)
            self.pdf.cell(50, 6, value, border=1, fill=True)
            self.pdf.cell(self.CONTENT_WIDTH - 110, 6, notes, border=1, fill=True)
            self.pdf.ln()
        
        self.pdf.ln(10)

    def _add_technical_breakdown_audio(self, data: AudioAnalysisData) -> None:
        """Add technical breakdown table for audio analysis."""
        self.pdf.set_font("Helvetica", "B", 12)
        self.pdf.set_text_color(*self.TITLE_COLOR)
        self.pdf.cell(0, 8, "TECHNICAL BREAKDOWN", ln=True)
        self.pdf.ln(3)
        
        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.set_fill_color(*self.TABLE_HEADER_BG)
        
        col_widths = [60, 50, self.CONTENT_WIDTH - 110]
        for header in ["Metric", "Value", "Notes"]:
            self.pdf.cell(col_widths.pop(0), 7, header, border=1, fill=True)
        self.pdf.ln()
        
        self.pdf.set_font("Helvetica", "", 9)
        self.pdf.set_text_color(55, 65, 81)
        
        metrics = [
            ("File Name", data.file_name, ""),
            ("Duration", f"{data.duration_seconds:.2f}s", ""),
            ("Verdict", data.verdict, ""),
            ("Confidence", f"{data.confidence:.1f}%", ""),
            ("Fake Probability", f"{data.fake_prob:.4f}", ""),
            ("Real Probability", f"{data.real_prob:.4f}", ""),
            ("Detection Type", data.detected_type or "N/A", ""),
            ("Anomaly Segments", str(len(data.anomaly_timestamps)), ""),
        ]
        
        for i, (metric, value, notes) in enumerate(metrics):
            if i % 2 == 1:
                self.pdf.set_fill_color(249, 250, 251)
            else:
                self.pdf.set_fill_color(255, 255, 255)
            
            self.pdf.cell(60, 6, metric, border=1, fill=True)
            self.pdf.cell(50, 6, value, border=1, fill=True)
            self.pdf.cell(self.CONTENT_WIDTH - 110, 6, notes, border=1, fill=True)
            self.pdf.ln()
        
        self.pdf.ln(10)

    def _add_forensic_evidence_video(self, data: VideoAnalysisData) -> None:
        """Add forensic evidence section with XAI visualizations for video."""
        self.pdf.set_font("Helvetica", "B", 12)
        self.pdf.set_text_color(*self.TITLE_COLOR)
        self.pdf.cell(0, 8, "FORENSIC EVIDENCE", ln=True)
        self.pdf.ln(3)
        
        for i, xai_data in enumerate(data.anomaly_frame_data[:3]):
            self.pdf.set_font("Helvetica", "B", 10)
            self.pdf.set_text_color(55, 65, 81)
            self.pdf.cell(0, 6, f"Anomaly Frame {xai_data.frame_index}", ln=True)
            if xai_data.timestamp:
                self.pdf.set_font("Helvetica", "", 9)
                self.pdf.cell(0, 5, f"Timestamp: {xai_data.timestamp}", ln=True)
            
            self.pdf.ln(2)
            
            self._add_dual_image_comparison(xai_data.gradcam_b64, xai_data.ela_b64)
            self.pdf.ln(8)
        
        self.pdf.ln(5)

    def _add_forensic_evidence_image(self, xai_data: XAIResultData) -> None:
        """Add forensic evidence section for image with XAI visualizations."""
        self.pdf.set_font("Helvetica", "B", 12)
        self.pdf.set_text_color(*self.TITLE_COLOR)
        self.pdf.cell(0, 8, "FORENSIC EVIDENCE", ln=True)
        self.pdf.ln(3)
        
        self._add_dual_image_comparison(xai_data.gradcam_b64, xai_data.ela_b64)
        self.pdf.ln(10)

    def _add_dual_image_comparison(self, image1_b64: Optional[str], image2_b64: Optional[str], 
                            label1: str = "Original Frame / Grad-CAM", 
                            label2: str = "ELA Heatmap") -> None:
        """Add side-by-side image comparison."""
        img_width = (self.CONTENT_WIDTH - 10) / 2
        img_height = img_width * 0.75
        
        x1 = self.MARGIN
        x2 = self.MARGIN + img_width + 10
        
        if image1_b64:
            try:
                self._add_image_from_base64(image1_b64, x1, self.pdf.get_y(), img_width, img_height)
                self.pdf.set_xy(x1, self.pdf.get_y() + img_height + 2)
                self.pdf.set_font("Helvetica", "B", 8)
                self.pdf.set_text_color(55, 65, 81)
                self.pdf.cell(img_width, 4, label1, align="C", ln=True)
            except Exception as e:
                logger.warning(f"Failed to add image1: {e}")
        
        if image2_b64:
            try:
                self._add_image_from_base64(image2_b64, x2, self.pdf.get_y() - img_height - 6, img_width, img_height)
                self.pdf.set_xy(x2, self.pdf.get_y() + img_height + 2)
                self.pdf.set_font("Helvetica", "B", 8)
                self.pdf.set_text_color(55, 65, 81)
                self.pdf.cell(img_width, 4, label2, align="C", ln=True)
            except Exception as e:
                logger.warning(f"Failed to add image2: {e}")
        
        self.pdf.ln(img_height + 10)

    def _add_anomaly_timeline(self, data: VideoAnalysisData) -> None:
        """Add timeline of detected anomalies for video."""
        self.pdf.set_font("Helvetica", "B", 12)
        self.pdf.set_text_color(*self.TITLE_COLOR)
        self.pdf.cell(0, 8, "ANOMALY TIMELINE", ln=True)
        self.pdf.ln(3)
        
        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.set_fill_color(*self.TABLE_HEADER_BG)
        
        col_widths = [40, 40, 40, 40]
        for header in ["Start", "End", "Avg Fake %", "Severity"]:
            self.pdf.cell(col_widths.pop(0), 7, header, border=1, fill=True)
        self.pdf.ln()
        
        self.pdf.set_font("Helvetica", "", 9)
        self.pdf.set_text_color(55, 65, 81)
        
        for seg in data.anomaly_segments:
            if seg.severity.lower() == "high":
                self.pdf.set_text_color(*self.DANGER_COLOR)
            elif seg.severity.lower() == "medium":
                self.pdf.set_text_color(*self.WARNING_COLOR)
            else:
                self.pdf.set_text_color(55, 65, 81)
            
            self.pdf.cell(40, 6, seg.start_timestamp, border=1)
            self.pdf.cell(40, 6, seg.end_timestamp, border=1)
            self.pdf.cell(40, 6, f"{seg.fake_prob:.2%}", border=1)
            self.pdf.cell(40, 6, seg.severity.upper(), border=1)
            self.pdf.ln()
        
        self.pdf.set_text_color(55, 65, 81)
        self.pdf.ln(10)

    def _add_audio_visualizations(self, data: AudioAnalysisData) -> None:
        """Add audio waveform and spectrogram visualizations."""
        self.pdf.set_font("Helvetica", "B", 12)
        self.pdf.set_text_color(*self.TITLE_COLOR)
        self.pdf.cell(0, 8, "AUDIO VISUALIZATIONS", ln=True)
        self.pdf.ln(3)
        
        if data.waveform_b64:
            self._add_single_image(data.waveform_b64, "Waveform")
        
        if data.spectrogram_b64:
            self._add_single_image(data.spectrogram_b64, "Spectrogram")
        
        self.pdf.ln(10)

    def _add_audio_anomaly_timestamps(self, data: AudioAnalysisData) -> None:
        """Add audio anomaly timestamps section."""
        self.pdf.set_font("Helvetica", "B", 12)
        self.pdf.set_text_color(*self.TITLE_COLOR)
        self.pdf.cell(0, 8, "ANOMALY TIMESTAMPS", ln=True)
        self.pdf.ln(3)
        
        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.set_text_color(55, 65, 81)
        
        for ts in data.anomaly_timestamps:
            self.pdf.cell(0, 5, f"• {ts}", ln=True)
        
        self.pdf.ln(10)

    def _add_single_image(self, image_b64: Optional[str], label: str = "") -> None:
        """Add a single image to the PDF."""
        if not image_b64:
            return
        
        try:
            img_width = self.CONTENT_WIDTH
            img_height = img_width * 0.5
            
            self._add_image_from_base64(image_b64, self.MARGIN, self.pdf.get_y(), img_width, img_height)
            
            if label:
                self.pdf.set_xy(self.MARGIN, self.pdf.get_y() + img_height + 2)
                self.pdf.set_font("Helvetica", "B", 9)
                self.pdf.set_text_color(55, 65, 81)
                self.pdf.cell(img_width, 4, label, align="C", ln=True)
            
            self.pdf.ln(img_height + 10)
        except Exception as e:
            logger.warning(f"Failed to add image {label}: {e}")

    def _add_image_from_base64(self, b64_str: str, x: float, y: float, width: float, height: float) -> None:
        """Add an image from base64 string to the PDF."""
        try:
            encoded = b64_str.split(",")[1] if "," in b64_str else b64_str
            img_bytes = base64.b64decode(encoded)
            
            img = Image.open(io.BytesIO(img_bytes))
            img_io = io.BytesIO()
            img.save(img_io, format="PNG")
            img_io.seek(0)
            
            self.pdf.image(img_io, x=x, y=y, w=width, h=height)
        except Exception as e:
            logger.warning(f"Failed to decode image: {e}")
            self.pdf.rect(x, y, width, height)
            self.pdf.set_xy(x + width / 2, y + height / 2)
            self.pdf.set_font("Helvetica", "I", 8)
            self.pdf.set_text_color(156, 163, 175)
            self.pdf.cell(10, 5, "[Image unavailable]", align="C")

    def _add_footer(self) -> None:
        """Add footer with generation info."""
        self.pdf.set_y(-20)
        self.pdf.set_font("Helvetica", "I", 8)
        self.pdf.set_text_color(156, 163, 175)
        self.pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | X-DetectRT Deepfake Detection System", align="C", ln=True)

    def _wrap_text(self, text: str, max_width: float, font_size: int) -> List[str]:
        """Wrap text to fit within max_width."""
        if not text:
            return []
        
        chars_per_line = int(max_width / (font_size * 0.5))
        lines = []
        words = text.split()
        
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= chars_per_line:
                current_line = (current_line + " " + word).strip()
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines


def generate_forensic_report(data: AnalysisReportData, output_path: str) -> str:
    """Convenience function to generate a forensic report PDF."""
    generator = PDFGenerator()
    return generator.generate(data, output_path)