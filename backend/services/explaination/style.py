# backend/xai/style.py
"""
Shared professional styling for all XAI visualization modules.

Color Palette (dark-theme, publication-quality):
    BG      = #1a1a2e  — figure/axes background (very dark navy)
    FAKE    = #ff8c42  — bars/elements pushing toward FAKE
    REAL    = #2e9ca8  — bars/elements pushing toward REAL
    TEXT    = #ecf0f1  — all labels, ticks, annotations
    GRID    = #34495e  — subtle gridlines
    WARN    = #e74c3c  — high-severity / critical indicators
    NEUTRAL = #95a5a6  — neutral/zero reference lines
    ACCENT  = #3498db  — secondary data lines/series
"""

import io
import base64
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Palette constants
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "BG":       "#1a1a2e",
    "FAKE":     "#ff8c42",
    "REAL":     "#2e9ca8",
    "TEXT":     "#ecf0f1",
    "GRID":     "#34495e",
    "WARN":     "#e74c3c",
    "NEUTRAL":  "#95a5a6",
    "ACCENT":   "#3498db",
}

# Matplotlib rcParams applied globally once on import
_FONT_FAMILY = ["Arial", "DejaVu Sans", "sans-serif"]


# ──────────────────────────────────────────────────────────────────────────────
# Core styling helpers
# ──────────────────────────────────────────────────────────────────────────────

def apply_dark_style(fig, ax, *, gridlines: bool = True):
    """
    Apply the professional dark-navy style to a Matplotlib Figure + Axes pair.

    Args:
        fig        : matplotlib Figure object
        ax         : matplotlib Axes object (or list of Axes)
        gridlines  : whether to draw subtle horizontal gridlines (default True)
    """
    axes = ax if isinstance(ax, (list, tuple)) else [ax]

    fig.patch.set_facecolor(PALETTE["BG"])

    for a in axes:
        a.set_facecolor(PALETTE["BG"])

        # Spine styling — keep only bottom and left
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["left"].set_color(PALETTE["GRID"])
        a.spines["bottom"].set_color(PALETTE["GRID"])

        # Tick styling
        a.tick_params(colors=PALETTE["TEXT"], labelsize=10,
                      length=4, width=0.8, direction="out")
        a.xaxis.label.set_color(PALETTE["TEXT"])
        a.yaxis.label.set_color(PALETTE["TEXT"])
        a.title.set_color(PALETTE["TEXT"])

        # Gridlines
        if gridlines:
            a.grid(axis="x", color=PALETTE["GRID"], alpha=0.2,
                   linewidth=0.8, linestyle="--")
            a.set_axisbelow(True)


def fig_to_base64(fig, dpi: int = 300) -> str:
    """
    Render a Matplotlib figure to a base64-encoded PNG string.

    Args:
        fig : matplotlib Figure
        dpi : dots-per-inch (default 300 for publication quality)

    Returns:
        str : base64-encoded PNG
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                dpi=dpi, facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Color interpolation
# ──────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> np.ndarray:
    """Convert '#rrggbb' to normalised (3,) float array."""
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)])


def interpolate_colors(values: list, low_hex: str, high_hex: str) -> list:
    """
    Linearly interpolate between two hex colors based on normalised values.

    Args:
        values   : list of floats in [0, 1]
        low_hex  : hex color for value == 0  (e.g. REAL teal)
        high_hex : hex color for value == 1  (e.g. FAKE orange)

    Returns:
        list of '#rrggbb' strings, one per value
    """
    lo = _hex_to_rgb(low_hex)
    hi = _hex_to_rgb(high_hex)
    vals = np.asarray(values, dtype=float)
    # Normalise to [0, 1] if needed
    v_min, v_max = vals.min(), vals.max()
    if v_max - v_min > 1e-8:
        vals_n = (vals - v_min) / (v_max - v_min)
    else:
        vals_n = np.full_like(vals, 0.5)

    colors = []
    for v in vals_n:
        rgb = (lo * (1 - v) + hi * v)
        colors.append("#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    return colors


def tcav_tier_colors(values: list) -> list:
    """
    Three-tier coloring for TCAV:
        score >= 0.5  → FAKE orange
        0.3 <= score < 0.5 → WARN red
        score < 0.3   → REAL teal
    """
    colors = []
    for v in values:
        if v >= 0.5:
            colors.append(PALETTE["FAKE"])
        elif v >= 0.3:
            colors.append(PALETTE["WARN"])
        else:
            colors.append(PALETTE["REAL"])
    return colors


# ──────────────────────────────────────────────────────────────────────────────
# Shared chart builder
# ──────────────────────────────────────────────────────────────────────────────

def styled_barh(ax, labels: list, values: list, colors: list, *,
                xlim: float = 1.15, show_pct: bool = False):
    """
    Draw a professional horizontal bar chart.

    Args:
        ax        : matplotlib Axes
        labels    : Y-axis tick labels
        values    : bar lengths (0.0–1.0 expected)
        colors    : list of hex colors, one per bar
        xlim      : X-axis upper limit (default 1.15 to leave room for labels)
        show_pct  : if True show percentage values (e.g. "42%"), else decimals
    """
    bars = ax.barh(labels, values, color=colors,
                   edgecolor="none", height=0.55)

    for bar, val in zip(bars, values):
        label = f"{val * 100:.1f}%" if show_pct else f"{val:.3f}"
        ax.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center", ha="left",
            fontsize=9, color=PALETTE["TEXT"],
            fontfamily=_FONT_FAMILY,
        )

    ax.set_xlim(0, xlim)
    return bars


def set_axis_labels(ax, xlabel: str, ylabel: str, title: str,
                    title_size: int = 15, label_size: int = 11):
    """Apply bold title and descriptive axis labels."""
    ax.set_title(title, fontsize=title_size, fontweight="bold",
                 color=PALETTE["TEXT"], pad=14,
                 fontfamily=_FONT_FAMILY)
    ax.set_xlabel(xlabel, fontsize=label_size, color=PALETTE["TEXT"],
                  fontfamily=_FONT_FAMILY, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=label_size, color=PALETTE["TEXT"],
                  fontfamily=_FONT_FAMILY, labelpad=8)
