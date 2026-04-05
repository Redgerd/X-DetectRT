# backend/services/reports/base_report.py
"""
Synaptic Shield — Base Forensic Report
Shared branding, headers, footers, and utility drawing routines
used by all three report modules (Image, Video, Audio).
"""

import io
import uuid
import hashlib
import base64
from datetime import datetime
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, KeepTogether, Flowable, HRFlowable
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line

W, H = A4

# ─── PROFESSIONAL LIGHT PALETTE ───────────────────────────────────────────────
class BrandColors:
    WHITE        = colors.HexColor('#FFFFFF')
    BG_PAGE      = colors.HexColor('#F7F8FA')
    BG_CARD      = colors.HexColor('#FFFFFF')
    BG_HEADER    = colors.HexColor('#1B2A4A')
    BG_SUBHEADER = colors.HexColor('#E8EDF5')
    NAVY         = colors.HexColor('#1B2A4A')
    BLUE         = colors.HexColor('#2563EB')
    BLUE_LIGHT   = colors.HexColor('#DBEAFE')
    RED          = colors.HexColor('#DC2626')
    RED_LIGHT    = colors.HexColor('#FEE2E2')
    GREEN        = colors.HexColor('#16A34A')
    GREEN_LIGHT  = colors.HexColor('#DCFCE7')
    AMBER        = colors.HexColor('#D97706')
    AMBER_LIGHT  = colors.HexColor('#FEF3C7')
    GREY_900     = colors.HexColor('#111827')
    GREY_700     = colors.HexColor('#374151')
    GREY_500     = colors.HexColor('#6B7280')
    GREY_300     = colors.HexColor('#D1D5DB')
    GREY_100     = colors.HexColor('#F3F4F6')
    BORDER       = colors.HexColor('#E5E7EB')
    
    # Old mapping aliases to not break everything immediately
    SURFACE         = colors.HexColor('#1B2A4A')
    TABLE_HEADER    = colors.HexColor('#1B2A4A')
    TABLE_ROW       = colors.HexColor('#FFFFFF')
    TABLE_ROW_ALT   = colors.HexColor('#F9FAFB')
    TEXT_HIGH       = colors.HexColor('#1B2A4A')
    TEXT_MED        = colors.HexColor('#6B7280')
    ELECTRIC_TEAL   = colors.HexColor('#2563EB')
    ALERT_RED       = colors.HexColor('#DC2626')
    NEURAL_GREEN    = colors.HexColor('#16A34A')

PLT_NAVY  = '#1B2A4A'
PLT_BLUE  = '#2563EB'
PLT_RED   = '#DC2626'
PLT_GREEN = '#16A34A'
PLT_AMBER = '#D97706'
PLT_GREY  = '#6B7280'
PLT_LGREY = '#E5E7EB'


# ─── fig → ReportLab Image ────────────────────────────────────────────────────
def fig_to_img(fig, w, h):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=w, height=h)

# ─── CHARTS FROM SAMPLE.PY ────────────────────────────────────────────────────

def make_confidence_bar(fake=74.37, real=25.63, width_in=5.6, height_in=1.2):
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.barh([''], [fake], color=PLT_RED, height=0.45, label=f'Fake  {fake:.1f}%')
    ax.barh([''], [real], left=[fake], color=PLT_GREEN, height=0.45, label=f'Real  {real:.1f}%')

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel('Probability (%)', fontsize=8, color=PLT_GREY, fontfamily='DejaVu Sans')
    ax.tick_params(axis='x', colors=PLT_GREY, labelsize=7.5)
    ax.set_yticks([])

    # Removed decision threshold line and text per user request

    if fake > 10:
        ax.text(fake/2, 0, f'{fake:.1f}%', ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white', fontfamily='DejaVu Sans')
    if real > 10:
        ax.text(fake + real/2, 0, f'{real:.1f}%', ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='white', fontfamily='DejaVu Sans')

    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.35), fontsize=7.5, frameon=True,
              framealpha=1, edgecolor=PLT_LGREY)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(PLT_LGREY)
    ax.spines['bottom'].set_color(PLT_LGREY)

    fig.tight_layout(pad=0.4)
    return fig

def make_donut(fake=74.37, real=25.63, size=2.6):
    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    wedges, _ = ax.pie(
        [fake, real],
        colors=[PLT_RED, PLT_GREEN],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor='white', linewidth=2)
    )

    ax.text(0, 0.10, f'{fake:.1f}%', ha='center', va='center',
            fontsize=14, fontweight='bold', color=PLT_RED, fontfamily='DejaVu Sans')
    ax.text(0, -0.22, 'FAKE', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color=PLT_GREY, fontfamily='DejaVu Sans')

    ax.plot(-1.1, -1.12, 's', color=PLT_RED, markersize=7)
    ax.text(-0.88, -1.12, f'Fake  {fake:.1f}%', va='center', fontsize=7,
            color='#374151', fontfamily='DejaVu Sans')
    ax.plot(0.22, -1.12, 's', color=PLT_GREEN, markersize=7)
    ax.text(0.44, -1.12, f'Real  {real:.1f}%', va='center', fontsize=7,
            color='#374151', fontfamily='DejaVu Sans')

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.3, 1.2)
    ax.axis('off')
    fig.tight_layout(pad=0)
    return fig

def make_artifact_bars(labels, values, width_in=5.2, height_in=2.2):
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bar_colors = [PLT_RED if v >= 65 else PLT_AMBER for v in values]
    bars = ax.barh(labels, values, color=bar_colors, height=0.5, edgecolor='none')
    
    ax.barh(labels, [100]*len(labels), color=PLT_LGREY, height=0.5, edgecolor='none', zorder=0)
    bars2 = ax.barh(labels, values, color=bar_colors, height=0.5, edgecolor='none', zorder=1)

    for bar, v in zip(bars2, values):
        ax.text(v + 0.8, bar.get_y() + bar.get_height()/2,
                f'{v}%', va='center', fontsize=7.5, color=PLT_NAVY,
                fontweight='bold', fontfamily='DejaVu Sans')

    ax.axvline(50, color=PLT_AMBER, linestyle='--', linewidth=1, alpha=0.8)
    ax.set_xlim(0, 112)
    ax.set_xlabel('Confidence Score (%)', fontsize=7.5, color=PLT_GREY,
                  fontfamily='DejaVu Sans')
    ax.tick_params(axis='x', colors=PLT_GREY, labelsize=7)
    ax.tick_params(axis='y', colors=PLT_NAVY, labelsize=7.5)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(PLT_LGREY)
    ax.spines['bottom'].set_color(PLT_LGREY)
    ax.set_title('Artifact Sub-Score Breakdown', fontsize=8, color=PLT_NAVY,
                 fontweight='bold', pad=6, fontfamily='DejaVu Sans', loc='left')
    fig.tight_layout(pad=0.5)
    return fig

def make_timeline(frames, scores, threshold=50, width_in=6.0, height_in=1.6):
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    # Draw timeline area
    ax.fill_between(frames, [0]*len(frames), scores, alpha=0.12, color=PLT_RED)
    
    # Flags vs Auth
    colors_sc = [PLT_RED if s >= threshold else PLT_GREEN for s in scores]
    ax.scatter(frames, scores, color=colors_sc, s=30, zorder=5, edgecolors='white', linewidths=1.0)
    
    ax.axhline(threshold, color=PLT_AMBER, linestyle='--', linewidth=1, alpha=0.9)
    ax.text(max(frames)*1.02 if len(frames) else 1, threshold, 'Threshold', va='center', fontsize=6.5, color=PLT_AMBER,
            fontfamily='DejaVu Sans')

    ax.set_ylim(0, 100)
    if len(frames) > 0:
        ax.set_xlim(min(frames)-1, max(frames)+1)
    
    ax.set_xlabel('Frame Index', fontsize=7.5, color=PLT_GREY, fontfamily='DejaVu Sans')
    ax.set_ylabel('Fake Prob. (%)', fontsize=7.5, color=PLT_GREY, fontfamily='DejaVu Sans')
    ax.tick_params(colors=PLT_GREY, labelsize=7)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(PLT_LGREY)
    ax.spines['bottom'].set_color(PLT_LGREY)
    ax.set_title('Frame-Level Anomaly Timeline', fontsize=8, fontweight='bold',
                 color=PLT_NAVY, loc='left', pad=5, fontfamily='DejaVu Sans')
    fig.tight_layout(pad=0.5)
    return fig

# ─── CUSTOM FLOWABLES ─────────────────────────────────────────────────────────

class SectionBand(Flowable):
    def __init__(self, text, w=None, accent=None):
        super().__init__()
        self.text   = text
        self._w     = w or (W - 40*mm)
        self._accent = accent or '#2563EB'
        self.height = 24

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(colors.HexColor('#EEF2FF'))
        c.roundRect(0, 2, self._w, 20, 3, fill=1, stroke=0)
        c.setFillColor(colors.HexColor(self._accent))
        c.roundRect(0, 2, 4, 20, 2, fill=1, stroke=0)
        c.setFont('Helvetica-Bold', 8)
        c.setFillColor(colors.HexColor('#1B2A4A'))
        c.drawString(12, 8, self.text.upper())
        c.restoreState()

    def wrap(self, aW, aH):
        return self._w, self.height

class VerdictBanner(Flowable):
    def __init__(self, is_fake=True, confidence=100.0, module_name="GenD", w=None):
        super().__init__()
        self._w = w or (W - 40*mm)
        self.height = 56
        self.is_fake = is_fake
        self.confidence = confidence
        self.module_name = module_name

    def draw(self):
        c = self.canv
        c.saveState()
        
        if self.is_fake:
            bg_color = '#FFF1F1'
            accent_color = '#DC2626'
            border_color = '#FECACA'
            icon = '!'
            title = 'VERDICT: DEEPFAKE / SYNTHETIC CONTENT DETECTED'
            sub = f'{self.module_name} neural network detected AI-generated content fingerprints with {self.confidence:.1f}% confidence.'
        else:
            bg_color = '#F0FDF4'
            accent_color = '#16A34A'
            border_color = '#BBF7D0'
            icon = '✓'
            title = 'VERDICT: AUTHENTIC / NO MANIPULATION DETECTED'
            sub = f'{self.module_name} verified content authenticity with {self.confidence:.1f}% confidence.'

        c.setFillColor(colors.HexColor(bg_color))
        c.roundRect(0, 0, self._w, 54, 5, fill=1, stroke=0)
        
        c.setFillColor(colors.HexColor(accent_color))
        c.roundRect(0, 0, 5, 54, 3, fill=1, stroke=0)
        
        c.setStrokeColor(colors.HexColor(border_color))
        c.setLineWidth(1)
        c.roundRect(0, 0, self._w, 54, 5, fill=0, stroke=1)
        
        c.setFillColor(colors.HexColor(accent_color))
        c.circle(28, 27, 16, fill=1, stroke=0)
        c.setFont('Helvetica-Bold', 16 if self.is_fake else 18)
        c.setFillColor(colors.white)
        c.drawCentredString(28, 21, icon)
        
        c.setFont('Helvetica-Bold', 11)
        c.setFillColor(colors.HexColor('#991B1B' if self.is_fake else '#166534'))
        c.drawString(55, 34, title)
        
        c.setFont('Helvetica', 7.5)
        c.setFillColor(colors.HexColor('#6B7280'))
        c.drawString(55, 18, sub)
        c.restoreState()

    def wrap(self, aW, aH):
        return self._w, self.height

class MetricBox(Flowable):
    def __init__(self, label, value, sub, accent='#2563EB', w=100, h=62):
        super().__init__()
        self.label  = label
        self.value  = value
        self.sub    = sub
        self.accent = accent
        self._w     = w
        self._h     = h
        self.height = h

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(colors.white)
        c.roundRect(0, 0, self._w, self._h, 5, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor('#E5E7EB'))
        c.setLineWidth(0.8)
        c.roundRect(0, 0, self._w, self._h, 5, fill=0, stroke=1)
        
        c.setFillColor(colors.HexColor(self.accent))
        c.roundRect(0, self._h - 4, self._w, 4, 2, fill=1, stroke=0)
        
        c.setFont('Helvetica', 6.5)
        c.setFillColor(colors.HexColor('#6B7280'))
        c.drawCentredString(self._w/2, self._h - 18, self.label.upper())
        
        c.setFont('Helvetica-Bold', 16)
        c.setFillColor(colors.HexColor(self.accent))
        c.drawCentredString(self._w/2, self._h - 38, self.value)
        
        c.setFont('Helvetica', 6.5)
        c.setFillColor(colors.HexColor('#6B7280'))
        c.drawCentredString(self._w/2, 8, self.sub)
        c.restoreState()

    def wrap(self, aW, aH):
        return self._w, self._h

# ─── TABLES & STYLES ──────────────────────────────────────────────────────────

def pro_table(header_bg=None, stripe=True):
    hbg = header_bg or colors.HexColor('#1B2A4A')
    ts = TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), hbg),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 7.5),
        ('TOPPADDING',    (0, 0), (-1, 0), 7),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 7),
        ('LEFTPADDING',   (0, 0), (-1, 0), 9),
        ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 1), (-1, -1), 7.5),
        ('TEXTCOLOR',     (0, 1), (-1, -1), colors.HexColor('#374151')),
        ('TOPPADDING',    (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 9),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 9),
        ('LINEBELOW',     (0, 0), (-1, 0), 0, colors.white),
        ('LINEBELOW',     (0, 1), (-1, -1), 0.4, colors.HexColor('#E5E7EB')),
        ('BOX',           (0, 0), (-1, -1), 0.8, colors.HexColor('#D1D5DB')),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ])
    if stripe:
        ts.add('ROWBACKGROUNDS', (0, 1), (-1, -1),
               [colors.white, colors.HexColor('#F9FAFB')])
    return ts

def build_styles() -> dict:
    styles = getSampleStyleSheet()
    def S(name, **kw):
        return ParagraphStyle(name, **kw)
    custom = {
        'h1': S('h1', fontName='Helvetica-Bold', fontSize=18, leading=24,
                 textColor=BrandColors.NAVY, spaceAfter=3),
        'h2': S('h2', fontName='Helvetica-Bold', fontSize=10, leading=14,
                 textColor=BrandColors.NAVY, spaceAfter=5),
        'sub': S('sub', fontName='Helvetica', fontSize=8.5, leading=12,
                  textColor=BrandColors.GREY_500, spaceAfter=10),
        'body': S('body', fontName='Helvetica', fontSize=8.5, leading=13,
                   textColor=BrandColors.GREY_700, spaceAfter=6),
        'muted': S('muted', fontName='Helvetica', fontSize=7.5, leading=11,
                    textColor=BrandColors.GREY_500, spaceAfter=4),
        'caption': S('caption', fontName='Helvetica-Oblique', fontSize=7,
                      textColor=colors.HexColor('#9CA3AF'), alignment=TA_CENTER),
        'label': S('label', fontName='Helvetica-Bold', fontSize=7,
                    textColor=BrandColors.NAVY, spaceAfter=2),
        'cover_super': S('cover_super', fontName='Helvetica-Bold', fontSize=7,
                          textColor=BrandColors.BLUE, spaceAfter=2, letterSpacing=2),
        'cover_title': S('cover_title', fontName='Helvetica-Bold', fontSize=28,
                          textColor=BrandColors.NAVY, spaceAfter=6, leading=34),
        'cover_subtitle': S('cover_subtitle', fontName='Helvetica', fontSize=13,
                             textColor=BrandColors.GREY_700, spaceAfter=4),
        'section_heading': S('section_heading', fontName='Helvetica-Bold', fontSize=11,
                              textColor=BrandColors.NAVY, spaceBefore=14, spaceAfter=6),
        'fhdr': S('fhdr', fontName='Helvetica-Bold', fontSize=8,
                  textColor=BrandColors.NAVY, spaceAfter=6),
    }
    return custom


def metadata_table(rows: list, col_widths=None) -> Table:
    """Build a clean 2-column metadata table."""
    data = [
        ['FIELD', 'VALUE']
    ]
    for label, value in rows:
        data.append([label, str(value)])
    
    w = col_widths or [130, 385]
    t = Table(data, colWidths=w, hAlign='LEFT')
    t.setStyle(pro_table())
    return t

# ─── UTILS ────────────────────────────────────────────────────────────────────

def decode_b64_image(b64_data: str) -> Optional[io.BytesIO]:
    try:
        if not b64_data: return None
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
        return io.BytesIO(img_bytes)
    except Exception:
        return None

def make_rl_image(b64_data: str, max_width: float = None, max_height: float = None) -> Optional[RLImage]:
    buf = decode_b64_image(b64_data)
    if buf is None: return None
    try:
        img = RLImage(buf)
        if max_width and max_height:
            aspect = img.imageHeight / float(img.imageWidth)
            if img.imageWidth > max_width:
                img.drawWidth  = max_width
                img.drawHeight = max_width * aspect
            if img.drawHeight > max_height:
                img.drawHeight = max_height
                img.drawWidth  = max_height / aspect
        return img
    except Exception:
        return None

def sha256_from_b64(b64_data: str) -> str:
    buf = decode_b64_image(b64_data)
    if not buf: return "N/A"
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ─── PAGE DECORATOR ───────────────────────────────────────────────────────────

def make_on_page(report_id, timestamp, module_type):
    def on_page(canvas, doc):
        canvas.saveState()

        # Page background
        canvas.setFillColor(colors.HexColor('#F4F6FA'))
        canvas.rect(0, 0, W, H, fill=1, stroke=0)

        # White content
        canvas.setFillColor(colors.white)
        canvas.roundRect(14*mm, 32, W - 28*mm, H - 80, 4, fill=1, stroke=0)
        canvas.setStrokeColor(colors.HexColor('#E5E7EB'))
        canvas.setLineWidth(0.5)
        canvas.roundRect(14*mm, 32, W - 28*mm, H - 80, 4, fill=0, stroke=1)

        # Header bar
        canvas.setFillColor(colors.HexColor('#1B2A4A'))
        canvas.rect(0, H - 54, W, 54, fill=1, stroke=0)

        # Blue strip
        canvas.setFillColor(colors.HexColor('#2563EB'))
        canvas.rect(0, H - 4, W, 4, fill=1, stroke=0)

        # Logo & Sub
        canvas.setFont('Helvetica-Bold', 13)
        canvas.setFillColor(colors.white)
        canvas.drawString(18*mm, H - 28, 'SYNAPTIC SHIELD')
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor('#93C5FD'))
        subtxt = f'{module_type} FORENSIC ANALYSIS  ·  XAI PLATFORM'
        canvas.drawString(18*mm, H - 41, subtxt.upper())

        # Header Meta
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor('#CBD5E1'))
        canvas.drawRightString(W - 18*mm, H - 22, f'REPORT ID: {report_id}')
        canvas.drawRightString(W - 18*mm, H - 33, f'GENERATED: {timestamp}')
        canvas.drawRightString(W - 18*mm, H - 44, f'PAGE {doc.page}')

        # Footer
        canvas.setFillColor(colors.HexColor('#1B2A4A'))
        canvas.rect(0, 0, W, 30, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor('#2563EB'))
        canvas.rect(0, 0, W, 2, fill=1, stroke=0)

        canvas.setFont('Helvetica', 6.5)
        canvas.setFillColor(colors.HexColor('#94A3B8'))
        canvas.drawString(18*mm, 15, 'Synaptic Shield XAI Platform  ·  Auto-generated')
        canvas.drawCentredString(W/2, 12, 'FORENSIC ANALYSIS REPORT')
        canvas.drawRightString(W - 18*mm, 15, 'Level: CONFIDENTIAL')

        canvas.restoreState()
    return on_page

def build_doc(output_path: str, report_id: str, module_type: str, timestamp: str) -> SimpleDocTemplate:
    doc = SimpleDocTemplate(
        output_path, 
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=62, bottomMargin=40,
        title=f"Forensic Analysis Report — {report_id}",
        author="Synaptic Shield XAI Platform",
    )
    # Store dynamic props for building later
    doc.make_on_page = make_on_page(report_id, timestamp, module_type)
    return doc
