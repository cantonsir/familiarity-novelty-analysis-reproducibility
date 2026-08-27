"""Shared visual language and fixed-artboard export for manuscript figures.

The module deliberately keeps the public plotting API small.  Figure builders
should use the named semantic palettes rather than selecting ad-hoc colours,
and should save through :func:`save_figure` so that PDF, SVG, and PNG outputs
share one declared physical size.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Hashable, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


MM_PER_INCH = 25.4
FONT_FAMILY = "Arial"
PNG_DPI = 600

# Quiet publication neutrals.
TEXT_COLOR = "#1D1D1F"
MUTED_TEXT = "#6E6E73"
ZERO_COLOR = "#86868B"
HAIRLINE_COLOR = "#D2D2D7"
GRID_COLOR = "#E8E8ED"
SURFACE_COLOR = "#F5F5F7"
WHITE = "#FFFFFF"

# Semantic colours shared by every figure family.  Keys are presentation
# labels used by the release renderers; aliases retain convenient short names.
CATEGORY_COLORS = {
    "Face": "#FF9F0A",
    "Scenery": "#0071E3",
    "Geometry": "#AF52DE",
}

TRAIT_COLORS = {
    "FNI": "#0071E3",
    "AQ": "#AF52DE",
    "Maemuki": "#FF9F0A",
    "Flow": "#30B0C7",
}

MODE_COLORS = {
    "Laboratory": "#0A84FF",
    "Lab": "#0A84FF",
    "Online": "#30B0C7",
}

STATE_COLORS = {
    "Negative": "#0071E3",
    "Neutral": HAIRLINE_COLOR,
    "Positive": "#FF9F0A",
    "Novelty": "#0071E3",
    "Familiarity": "#FF9F0A",
    "Primary": "#0071E3",
    "Sensitivity": "#8E8E93",
    "Exploratory": "#AF52DE",
    "Excluded": "#D1D1D6",
}

NEGATIVE_COLOR = STATE_COLORS["Negative"]
POSITIVE_COLOR = STATE_COLORS["Positive"]
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "fni_negative_neutral_positive",
    (NEGATIVE_COLOR, WHITE, POSITIVE_COLOR),
    N=256,
)

CATEGORY_MARKERS = {"Face": "o", "Scenery": "s", "Geometry": "^"}
CATEGORY_LINESTYLES = {"Face": "-", "Scenery": "--", "Geometry": "-."}


def mm_to_inches(value: float) -> float:
    """Convert millimetres to inches without changing the requested size."""

    return float(value) / MM_PER_INCH


def set_publication_style() -> None:
    """Install the release-wide Arial Matplotlib style.

    TrueType PDF text and live SVG text are required by the release QA.  The
    sans-serif list intentionally contains only Arial so a missing font is
    visible during QA instead of silently becoming a mixed-font figure.
    """

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY],
            "font.size": 6.0,
            "font.weight": "normal",
            "text.color": TEXT_COLOR,
            "axes.titlesize": 7.0,
            "axes.titleweight": "bold",
            "axes.titlecolor": TEXT_COLOR,
            "axes.labelsize": 6.0,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": HAIRLINE_COLOR,
            "axes.linewidth": 0.5,
            "axes.axisbelow": True,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "xtick.color": MUTED_TEXT,
            "ytick.color": MUTED_TEXT,
            "xtick.major.size": 0.0,
            "ytick.major.size": 0.0,
            "xtick.minor.size": 0.0,
            "ytick.minor.size": 0.0,
            "legend.fontsize": 5.5,
            "legend.labelcolor": TEXT_COLOR,
            "legend.frameon": False,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.edgecolor": WHITE,
            "savefig.transparent": False,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.42,
            "grid.alpha": 0.9,
            "lines.color": TEXT_COLOR,
            "lines.linewidth": 1.25,
            "lines.markersize": 3.5,
            "patch.edgecolor": TEXT_COLOR,
            "patch.linewidth": 0.55,
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "f_n_manuscript_figures",
            "axes.unicode_minus": True,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def new_figure(width_mm: float, height_mm: float) -> Figure:
    """Return a publication-styled figure with a fixed millimetre artboard."""

    width = float(width_mm)
    height = float(height_mm)
    if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
        raise ValueError("Figure dimensions must be finite, positive millimetre values")
    set_publication_style()
    fig = plt.figure(figsize=(mm_to_inches(width), mm_to_inches(height)))
    # The declared size remains authoritative even if a backend perturbs the
    # current canvas while rendering another format.
    setattr(fig, "_manuscript_size_mm", (width, height))
    return fig


def add_title_block(
    fig: Figure,
    title: str,
    subtitle: str = "",
    *,
    left: float = 0.07,
    top: float = 0.965,
    title_size: float = 8.5,
    subtitle_size: float = 6.1,
    line_gap: float = 0.050,
) -> tuple[Any, Any | None]:
    """Add a consistent, left-aligned figure title and contextual subtitle."""

    title_artist = fig.text(
        left,
        top,
        str(title),
        ha="left",
        va="top",
        color=TEXT_COLOR,
        fontsize=title_size,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
    )
    subtitle_artist = None
    if subtitle:
        subtitle_artist = fig.text(
            left,
            top - line_gap,
            str(subtitle),
            ha="left",
            va="top",
            color=MUTED_TEXT,
            fontsize=subtitle_size,
            fontfamily=FONT_FAMILY,
            fontweight="normal",
            linespacing=1.18,
        )
    return title_artist, subtitle_artist


def add_figure_note(
    fig: Figure,
    text: str,
    *,
    left: float = 0.07,
    bottom: float = 0.018,
    size: float = 5.2,
) -> Any:
    """Add a quiet, left-aligned figure-level note inside the artboard."""

    return fig.text(
        left,
        bottom,
        str(text),
        ha="left",
        va="bottom",
        color=MUTED_TEXT,
        fontsize=size,
        fontfamily=FONT_FAMILY,
        linespacing=1.15,
    )


def style_axis(
    ax: Axes,
    *,
    grid_axis: str | None = "y",
    zero_x: bool = False,
    zero_y: bool = False,
) -> Axes:
    """Apply the shared quiet-axis treatment and optional zero references."""

    if grid_axis not in {None, "none", "x", "y", "both"}:
        raise ValueError("grid_axis must be one of None, 'none', 'x', 'y', or 'both'")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(HAIRLINE_COLOR)
    ax.spines["bottom"].set_color(HAIRLINE_COLOR)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="both", which="both", length=0, colors=MUTED_TEXT)
    ax.grid(False)
    if grid_axis not in {None, "none"}:
        ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.42, alpha=0.9)
    ax.set_axisbelow(True)
    if zero_x:
        ax.axvline(
            0,
            color=ZERO_COLOR,
            linewidth=0.7,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
    if zero_y:
        ax.axhline(
            0,
            color=ZERO_COLOR,
            linewidth=0.7,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
    return ax


def add_panel_label(
    ax: Axes,
    label: str,
    *,
    x: float = -0.12,
    y: float = 1.06,
    size: float = 8.0,
) -> Any:
    """Place one bold panel letter outside the plotting region."""

    return ax.text(
        x,
        y,
        str(label),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=TEXT_COLOR,
        fontsize=size,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
        clip_on=False,
    )


def add_panel_heading(ax: Axes, text: str, *, pad: float = 3.0) -> Any:
    """Set a short, left-aligned panel heading."""

    return ax.set_title(
        str(text),
        loc="left",
        pad=pad,
        color=TEXT_COLOR,
        fontsize=7.0,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
    )


def deterministic_jitter(
    key: Hashable | Sequence[Hashable],
    amplitude: float = 0.08,
) -> float | list[float]:
    """Return stable, order-independent jitter derived from one or more keys.

    Passing stable participant/item identifiers gives each observation the same
    displacement across figures and Python processes.  A sequence returns a
    list; a scalar returns one float.
    """

    width = float(amplitude)
    if not math.isfinite(width) or width < 0:
        raise ValueError("amplitude must be a finite, non-negative value")

    def one(value: Hashable) -> float:
        payload = f"f_n_manuscript_jitter|{type(value).__name__}|{value!r}".encode("utf-8")
        integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        unit = integer / float((1 << 64) - 1)
        return (2.0 * unit - 1.0) * width

    if isinstance(key, Sequence) and not isinstance(key, (str, bytes, bytearray)):
        return [one(value) for value in key]
    return one(key)


def _declared_size_mm(fig: Figure) -> tuple[float, float]:
    stored = getattr(fig, "_manuscript_size_mm", None)
    if stored is not None:
        return float(stored[0]), float(stored[1])
    width_in, height_in = fig.get_size_inches()
    return float(width_in) * MM_PER_INCH, float(height_in) * MM_PER_INCH


def _png_canvas_inches(pixels: int, dpi: int) -> float:
    # Matplotlib's Agg backend truncates the floating-point canvas product.
    # A tiny sub-pixel cushion survives backend arithmetic while remaining far
    # below the next whole pixel, so the exported canvas is exactly ``pixels``.
    return (float(pixels) + 1.0e-4) / float(dpi)


def save_figure(
    fig: Figure,
    base_path: str | Path,
    png_dpi: int = PNG_DPI,
) -> dict[str, str]:
    """Save fixed-size PDF/SVG masters and an exact-pixel PNG preview.

    ``base_path`` is extension-free (an existing extension is replaced).  The
    vector files retain the exact declared millimetre artboard and all SVG text
    remains live.  The PNG canvas is quantised to the nearest whole pixel at
    ``png_dpi`` and then the figure is restored to its exact physical size.

    The function never uses ``bbox_inches='tight'``: all titles, notes, and
    marks must fit inside the declared manuscript artboard.
    """

    if isinstance(png_dpi, bool) or int(png_dpi) != png_dpi or int(png_dpi) <= 0:
        raise ValueError("png_dpi must be a positive integer")
    dpi = int(png_dpi)
    base = Path(base_path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pdf": str(base.with_suffix(".pdf")),
        "svg": str(base.with_suffix(".svg")),
        "png": str(base.with_suffix(".png")),
    }

    width_mm, height_mm = _declared_size_mm(fig)
    exact_inches = (mm_to_inches(width_mm), mm_to_inches(height_mm))
    # ``forward=True`` asks the interactive manager to resize at its screen
    # DPI, which quantises some manuscript heights before export.  The saved
    # backend reads the figure inches directly, so keep manager forwarding off.
    fig.set_size_inches(*exact_inches, forward=False)
    setattr(fig, "_manuscript_size_mm", (width_mm, height_mm))

    pdf_metadata = {
        "Creator": "F_N manuscript figure pipeline",
        "Title": base.name,
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        outputs["pdf"],
        format="pdf",
        metadata=pdf_metadata,
        bbox_inches=None,
        transparent=False,
    )
    fig.savefig(
        outputs["svg"],
        format="svg",
        metadata={
            "Creator": "F_N manuscript figure pipeline",
            "Title": base.name,
            "Date": None,
        },
        bbox_inches=None,
        transparent=False,
    )

    target_width = round(width_mm / MM_PER_INCH * dpi)
    target_height = round(height_mm / MM_PER_INCH * dpi)
    fig.set_size_inches(
        _png_canvas_inches(target_width, dpi),
        _png_canvas_inches(target_height, dpi),
        forward=False,
    )
    try:
        fig.savefig(
            outputs["png"],
            format="png",
            dpi=dpi,
            bbox_inches=None,
            transparent=False,
            metadata={"Software": "F_N manuscript figure pipeline"},
        )
    finally:
        fig.set_size_inches(*exact_inches, forward=False)

    return outputs


def format_p(value: Any) -> str:
    """Format a p value compactly without converting missing values to zero."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(numeric):
        return "NA"
    if numeric < 0.001:
        return f"{numeric:.2e}"
    return f"{numeric:.3f}"


# Install the style at import so Seaborn calls made after importing this module
# inherit the same typography.  Builders still call new_figure explicitly.
set_publication_style()
