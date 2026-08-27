"""Reproduce the manuscript's complete Main Figures 2--5.

The renderers consume only the frozen, privacy-screened plot-data tables under
``data/main``.  They preserve the active manuscript artboards, panel geometry,
typography, colours, symbols, annotations, and displayed estimates.
"""

from __future__ import annotations

import copy
from io import BytesIO
import logging
from pathlib import Path
from typing import Iterable

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject
from scipy.stats import gaussian_kde


# Matplotlib's embedded TrueType font program triggers two harmless parser
# warnings in pypdf while Figure 2's vector panels are composed.  The finished
# page is validated independently, so keep the public reviewer run quiet.
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)

try:  # Notebook/top-level import.
    from style import (
        CATEGORY_COLORS,
        CATEGORY_LINESTYLES,
        CATEGORY_MARKERS,
        DIVERGING_CMAP,
        GRID_COLOR,
        HAIRLINE_COLOR,
        MUTED_TEXT,
        STATE_COLORS,
        TEXT_COLOR,
        WHITE,
        add_figure_note,
        add_title_block,
        new_figure,
        style_axis,
    )
except ImportError:  # Package-style import.
    from .style import (
        CATEGORY_COLORS,
        CATEGORY_LINESTYLES,
        CATEGORY_MARKERS,
        DIVERGING_CMAP,
        GRID_COLOR,
        HAIRLINE_COLOR,
        MUTED_TEXT,
        STATE_COLORS,
        TEXT_COLOR,
        WHITE,
        add_figure_note,
        add_title_block,
        new_figure,
        style_axis,
    )


CATEGORY_ORDER = ["Face", "Scenery", "Geometry"]
BEHAVIOR_VARIABLES = ["face_fni", "geometry_fni", "scenery_fni"]
QUESTIONNAIRE_VARIABLES = ["aq_score", "4_nf_daily", "maemuki_score", "10_flow_attitude"]
SCATTER_VARIABLES = ["aq_score", "maemuki_score", "10_flow_attitude"]
DAILY_VARIABLES = [
    "4_food/cuisine",
    "4_places/activities",
    "4_movies or videos",
    "4_restaurant",
    "4_clothes",
    "4_digital games",
    "4_books or magazines",
    "4_snacks or treats",
]
TRAITS = ["aq_score", "10_flow_attitude", "maemuki_score"]
PAIR_VARIABLES = ["aq_score", "4_nf_daily", "maemuki_score", "10_flow_attitude"]
SHORT_LABELS = {
    "aq_score": "AQ",
    "4_nf_daily": "Daily\nfamiliarity",
    "maemuki_score": "Maemuki",
    "10_flow_attitude": "Flow\nattitude",
}
LABELS = {
    "face_fni": "Face FNI",
    "geometry_fni": "Geometry FNI",
    "scenery_fni": "Scenery FNI",
    "aq_score": "AQ",
    "4_nf_daily": "Daily familiarity",
    "maemuki_score": "Maemuki",
    "10_flow_attitude": "Flow attitude",
    "4_food/cuisine": "Food/cuisine",
    "4_places/activities": "Places/activities",
    "4_movies or videos": "Movies/videos",
    "4_restaurant": "Restaurant",
    "4_clothes": "Clothes",
    "4_digital games": "Digital games",
    "4_books or magazines": "Books/magazines",
    "4_snacks or treats": "Snacks/treats",
}

PROFILE_FNI_VARIABLES = ["face_fni", "geometry_fni", "scenery_fni"]
PROFILE_VARIABLES = [*PROFILE_FNI_VARIABLES, *DAILY_VARIABLES]
PROFILE_LABELS = {
    "face_fni": "Face",
    "geometry_fni": "Geometry",
    "scenery_fni": "Scenery",
    "4_food/cuisine": "Food",
    "4_places/activities": "Places",
    "4_movies or videos": "Movies",
    "4_restaurant": "Restaurants",
    "4_clothes": "Clothes",
    "4_digital games": "Games",
    "4_books or magazines": "Books",
    "4_snacks or treats": "Snacks",
}
PROFILE_JITTER_COLUMNS = {
    "face_fni": "jitter_face",
    "geometry_fni": "jitter_geometry",
    "scenery_fni": "jitter_scenery",
    "4_food/cuisine": "jitter_food",
    "4_places/activities": "jitter_places",
    "4_movies or videos": "jitter_movies",
    "4_restaurant": "jitter_restaurant",
    "4_clothes": "jitter_clothes",
    "4_digital games": "jitter_games",
    "4_books or magazines": "jitter_books",
    "4_snacks or treats": "jitter_snacks",
}
EXEMPLARS = {
    "A": {"manuscript_profile_id": "P092", "color": STATE_COLORS["Familiarity"]},
    "B": {"manuscript_profile_id": "P101", "color": STATE_COLORS["Novelty"]},
}

POINT_COLOR = "#547A9A"
LINE_COLOR = CATEGORY_COLORS["Face"]
POINT_SIZE = 4.4
POINT_ALPHA = 0.29
LINE_WIDTH = 0.82
SCATTER_BOX_ASPECT = 0.60


def _read_csv(path: Path, required: Iterable[str] = ()) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required frozen plot-data table is unavailable: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Frozen plot-data table is empty: {path}")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return frame


def _save_pdf(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    declared = getattr(figure, "_manuscript_size_mm", None)
    if declared is not None:
        figure.set_size_inches(float(declared[0]) / 25.4, float(declared[1]) / 25.4, forward=False)
    figure.savefig(
        path,
        format="pdf",
        bbox_inches=None,
        transparent=False,
        metadata={
            "Creator": "F/N NHB reproducibility package",
            "Title": path.stem,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    return path


def _stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def _format_rho(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.2f}".replace("0.", ".")


def _matrix(
    source: pd.DataFrame,
    *,
    row: str,
    column: str,
    value: str,
    row_order: list[str],
    column_order: list[str],
) -> pd.DataFrame:
    result = source.pivot(index=row, columns=column, values=value).reindex(
        index=row_order, columns=column_order
    )
    if result.isna().any().any():
        raise ValueError(f"Matrix {value} contains missing cells")
    return result


def _panel_heading(
    figure: plt.Figure,
    label: str,
    title: str,
    subtitle: str,
    y: float,
) -> None:
    figure.text(
        0.012,
        y,
        label,
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.text(
        0.062,
        y,
        title,
        ha="left",
        va="top",
        fontsize=7.4,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.text(
        0.062,
        y - 0.027,
        subtitle,
        ha="left",
        va="top",
        fontsize=4.65,
        color=MUTED_TEXT,
    )


def _draw_colorbar(figure: plt.Figure, position: list[float], label_size: float = 4.1) -> None:
    axis = figure.add_axes(position)
    edges = np.linspace(-1.0, 1.0, 49)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        axis.add_patch(
            Rectangle(
                (0, low),
                1,
                high - low,
                facecolor=DIVERGING_CMAP((index + 0.5) / 48.0),
                edgecolor="none",
                linewidth=0,
            )
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(-1, 1)
    axis.set_xticks([])
    axis.set_yticks([-1, 0, 1])
    axis.yaxis.tick_right()
    axis.yaxis.set_label_position("right")
    axis.tick_params(axis="y", labelsize=3.7, length=1.0, colors=MUTED_TEXT, pad=1.5)
    axis.set_ylabel("Spearman ρ", fontsize=label_size, color=TEXT_COLOR, labelpad=2.0)
    for spine in axis.spines.values():
        spine.set_color(HAIRLINE_COLOR)
        spine.set_linewidth(0.45)


def _draw_heatmap(
    figure: plt.Figure,
    matrix: pd.DataFrame,
    q_values: pd.DataFrame,
    *,
    position: list[float],
    colorbar_position: list[float],
    x_labels: list[str],
    y_labels: list[str],
    rotation: float,
    annotation_size: float,
    tick_size: float,
    highlight_row: int | None = None,
) -> None:
    axis = figure.add_axes(position)
    row_count, column_count = matrix.shape
    for row_index in range(row_count):
        for column_index in range(column_count):
            value = float(matrix.iloc[row_index, column_index])
            axis.add_patch(
                Rectangle(
                    (column_index, row_index),
                    1,
                    1,
                    facecolor=DIVERGING_CMAP(np.clip((value + 1.0) / 2.0, 0.0, 1.0)),
                    edgecolor=WHITE,
                    linewidth=0.5,
                )
            )
            q_value = float(q_values.iloc[row_index, column_index])
            axis.text(
                column_index + 0.5,
                row_index + 0.5,
                f"{value:.2f}{_stars(q_value)}",
                ha="center",
                va="center",
                fontsize=annotation_size,
                fontweight="bold" if q_value < 0.05 else "normal",
                color="white" if abs(value) >= 0.52 else TEXT_COLOR,
                zorder=4,
            )
    if highlight_row is not None:
        axis.add_patch(
            Rectangle(
                (0, highlight_row),
                column_count,
                1,
                fill=False,
                edgecolor=LINE_COLOR,
                linewidth=0.9,
                zorder=5,
            )
        )
    axis.set_xlim(0, column_count)
    axis.set_ylim(row_count, 0)
    axis.set_xticks(np.arange(column_count) + 0.5, x_labels)
    axis.set_yticks(np.arange(row_count) + 0.5, y_labels)
    axis.tick_params(axis="x", labelsize=tick_size, rotation=rotation, pad=1.5, colors=TEXT_COLOR)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    axis.tick_params(axis="y", labelsize=tick_size, pad=2.5, colors=TEXT_COLOR)
    axis.tick_params(axis="both", length=0)
    for spine in axis.spines.values():
        spine.set_color(HAIRLINE_COLOR)
        spine.set_linewidth(0.45)
    _draw_colorbar(figure, colorbar_position)


def _draw_scatter_cell(
    axis: plt.Axes,
    *,
    plot_x: np.ndarray,
    plot_y: np.ndarray,
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    rho: float,
    q_value: float,
    zero_x: bool = False,
    annotation_size: float = 4.0,
) -> None:
    axis.scatter(
        plot_x,
        plot_y,
        s=POINT_SIZE,
        color=POINT_COLOR,
        alpha=POINT_ALPHA,
        edgecolors="none",
        rasterized=False,
        zorder=3,
    )
    design = np.column_stack([np.ones(len(observed_x)), observed_x])
    intercept, slope = np.linalg.lstsq(design, observed_y, rcond=None)[0]
    line_x = np.linspace(float(observed_x.min()), float(observed_x.max()), 160)
    axis.plot(line_x, intercept + slope * line_x, color=LINE_COLOR, linewidth=LINE_WIDTH, zorder=4)
    style_axis(axis, grid_axis="both", zero_x=zero_x)
    axis.set_box_aspect(SCATTER_BOX_ASPECT)
    axis.text(
        0.045,
        0.93,
        f"ρ = {_format_rho(float(rho))}{_stars(float(q_value))}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=annotation_size,
        fontweight="bold" if float(q_value) < 0.05 else "normal",
        color=TEXT_COLOR,
        bbox={
            "boxstyle": "round,pad=0.14",
            "facecolor": WHITE,
            "edgecolor": HAIRLINE_COLOR,
            "linewidth": 0.33,
            "alpha": 0.91,
        },
        zorder=6,
    )


def _draw_diagonal(axis: plt.Axes, values: np.ndarray, label: str) -> None:
    edges = np.histogram_bin_edges(values, bins="fd")
    if len(edges) < 6:
        edges = np.linspace(float(values.min()), float(values.max()), 7)
    if len(edges) > 13:
        edges = np.linspace(float(values.min()), float(values.max()), 13)
    axis.hist(
        values,
        bins=edges,
        density=True,
        color="#DCEAF6",
        edgecolor=WHITE,
        linewidth=0.3,
        zorder=2,
    )
    if np.unique(values).size > 2:
        grid = np.linspace(float(values.min()), float(values.max()), 180)
        axis.plot(grid, gaussian_kde(values)(grid), color=POINT_COLOR, linewidth=LINE_WIDTH, zorder=3)
    axis.text(
        0.045,
        0.90,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=4.25,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    style_axis(axis, grid_axis=None)
    axis.set_box_aspect(SCATTER_BOX_ASPECT)
    axis.spines["left"].set_visible(False)
    axis.set_yticks([])


# ---------------------------------------------------------------------------
# Main Figure 2


def _style_trajectory_axis(
    axis: plt.Axes,
    y_limits: tuple[float, float] | None = None,
    y_label: str = "Familiarity preference (−3 to +3)",
) -> None:
    style_axis(axis, grid_axis="y", zero_y=True)
    axis.set_xlim(0.7, 18.3)
    axis.set_xticks([1, 6, 12, 18])
    axis.set_xlabel("Comparison trial")
    axis.set_ylabel(y_label)
    if y_limits is not None:
        axis.set_ylim(*y_limits)


def _add_preference_direction_guide(axis: plt.Axes) -> None:
    arrow_x = -0.075
    text_x = -0.105
    arrow_style = {
        "arrowstyle": "-|>",
        "color": TEXT_COLOR,
        "linewidth": 0.9,
        "mutation_scale": 8,
    }
    axis.annotate(
        "",
        xy=(arrow_x, 0.94),
        xytext=(arrow_x, 0.53),
        xycoords=axis.transAxes,
        arrowprops=arrow_style,
        annotation_clip=False,
    )
    axis.annotate(
        "",
        xy=(arrow_x, 0.06),
        xytext=(arrow_x, 0.47),
        xycoords=axis.transAxes,
        arrowprops=arrow_style,
        annotation_clip=False,
    )
    axis.text(
        text_x,
        0.735,
        "Familiarity",
        rotation=90,
        ha="center",
        va="center",
        color=TEXT_COLOR,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.text(
        text_x,
        0.265,
        "Novelty",
        rotation=90,
        ha="center",
        va="center",
        color=TEXT_COLOR,
        transform=axis.transAxes,
        clip_on=False,
    )


def _draw_observed_trajectory(summary: pd.DataFrame) -> plt.Figure:
    figure = new_figure(180, 100)
    axis = figure.add_subplot(111)
    for category in CATEGORY_ORDER:
        subset = summary.loc[summary["broad_category"].astype(str).eq(category)].sort_values(
            "trial_within_subcategory"
        )
        x = subset["trial_within_subcategory"].to_numpy(float)
        color = CATEGORY_COLORS[category]
        axis.fill_between(
            x,
            subset["ci_low"].to_numpy(float),
            subset["ci_high"].to_numpy(float),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        axis.plot(
            x,
            subset["observed_mean"].to_numpy(float),
            color=color,
            linestyle=CATEGORY_LINESTYLES[category],
            marker=CATEGORY_MARKERS[category],
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            markersize=4.3,
            linewidth=1.8,
            label=category,
        )
    _style_trajectory_axis(axis, y_limits=(-0.65, 0.65), y_label="")
    axis.set_yticks(np.arange(-0.6, 0.61, 0.2))
    _add_preference_direction_guide(axis)
    axis.legend(loc="upper right", ncol=3, frameon=False, title=None)
    figure.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.82)
    add_title_block(
        figure,
        "Observed familiarity-preference trajectories",
        "Matched Experiments 3 and 5 · participant-balanced mean and 95% t CI",
        left=0.05,
    )
    return figure


def _badge(axis: plt.Axes, x: float, y: float, label: str, color: str, *, size: float = 5.2) -> None:
    axis.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=size,
        fontweight="bold",
        color="white",
        bbox={
            "boxstyle": "circle,pad=0.20",
            "facecolor": color,
            "edgecolor": "white",
            "linewidth": 0.65,
        },
        zorder=20,
        clip_on=False,
    )


def _draw_selected_profile_summary(average: pd.DataFrame) -> plt.Figure:
    """Draw the original 7.2 x 4.8 inch source artboard's upper section."""

    average = average.sort_values("plot_order").reset_index(drop=True)
    if len(average) != 155 or int(average["daily_line_order"].ge(0).sum()) != 70:
        raise ValueError("Figure 2 participant-profile source must contain 155 rows and 70 sampled lines")
    figure = new_figure(7.2 * 25.4, 4.8 * 25.4)
    add_title_block(
        figure,
        "Individual differences in familiarity/novelty profiles",
        "E3/E5 all-completer cohort (n = 155) · two author-selected illustrative cases marked A/B · descriptive",
        left=0.05,
    )
    top = figure.add_gridspec(
        1,
        3,
        left=0.05,
        right=0.985,
        bottom=0.516,
        top=0.825,
        width_ratios=(0.72, 1.35, 2.75),
        wspace=0.17,
    )
    ax_average = figure.add_subplot(top[0, 0])
    ax_fni = figure.add_subplot(top[0, 1], sharey=ax_average)
    ax_daily = figure.add_subplot(top[0, 2], sharey=ax_average)

    selected_rows = {}
    for exemplar, spec in EXEMPLARS.items():
        selected = average.loc[
            average["manuscript_profile_id"].astype(str).eq(spec["manuscript_profile_id"])
        ]
        if len(selected) != 1:
            raise ValueError(f"Figure 2 exemplar {exemplar} is unavailable")
        selected_rows[exemplar] = selected.iloc[0]

    average_values = average["average_category_fni_daily_nf"].to_numpy(float)
    score_grid = np.linspace(-3.0, 3.0, 300)
    density = gaussian_kde(average_values)(score_grid)
    histogram, edges = np.histogram(average_values, bins=np.linspace(-3.0, 3.0, 25), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    density_max = max(float(density.max()), float(histogram.max()))
    ax_average.axhspan(-3.0, 0.0, color="#EEF6FF", zorder=0)
    ax_average.axhspan(0.0, 3.0, color="#FFF4E8", zorder=0)
    ax_average.barh(
        centers,
        histogram,
        height=float(edges[1] - edges[0]) * 0.88,
        color=[
            STATE_COLORS["Novelty"] if center < 0 else STATE_COLORS["Familiarity"]
            for center in centers
        ],
        edgecolor="white",
        linewidth=0.35,
        alpha=0.40,
        zorder=1,
    )
    ax_average.plot(density, score_grid, color=TEXT_COLOR, linewidth=1.0, zorder=3)
    rug_x = (
        0.035 * density_max
        + np.abs(average["average_rug_unit"].to_numpy(float)) * 0.075 * density_max
    )
    ax_average.scatter(
        rug_x,
        average_values,
        s=3.0,
        color=[
            STATE_COLORS["Novelty"] if value < 0 else STATE_COLORS["Familiarity"]
            for value in average_values
        ],
        alpha=0.42,
        edgecolors="white",
        linewidth=0.15,
        zorder=4,
    )
    selected_marker_x = 0.035 * density_max
    for exemplar, row in selected_rows.items():
        value = float(row["average_category_fni_daily_nf"])
        color = str(EXEMPLARS[exemplar]["color"])
        ax_average.scatter(
            [selected_marker_x],
            [value],
            s=56,
            color=color,
            edgecolors=TEXT_COLOR,
            linewidth=0.65,
            zorder=12,
        )
        _badge(ax_average, selected_marker_x, value, exemplar, color, size=4.7)
    mean_value = float(np.mean(average_values))
    median_value = float(np.median(average_values))
    ax_average.axhline(mean_value, color=TEXT_COLOR, linewidth=0.7, zorder=5)
    ax_average.text(
        0.04,
        0.96,
        f"mean {mean_value:.2f}\nmedian {median_value:.2f}",
        transform=ax_average.transAxes,
        ha="left",
        va="top",
        fontsize=5.0,
        color=MUTED_TEXT,
    )
    ax_average.text(0.04, 0.84, "Familiarity", transform=ax_average.transAxes, fontsize=5.0, color="#9A5A16")
    ax_average.text(0.04, 0.055, "Novelty", transform=ax_average.transAxes, fontsize=5.0, color=STATE_COLORS["Novelty"])
    ax_average.set_xlim(density_max * 1.14, -density_max * 0.10)
    ax_average.set_ylim(-3.25, 3.25)
    ax_average.set_yticks(np.arange(-3, 4, 1))
    ax_average.set_xticks([])
    ax_average.tick_params(axis="y", labelleft=False)
    ax_average.set_xlabel("Density")
    ax_average.set_title("Average F/N index", loc="left", pad=4)
    style_axis(ax_average, grid_axis="y", zero_y=True)
    ax_average.spines["left"].set_visible(False)

    fni_x = np.arange(len(PROFILE_FNI_VARIABLES), dtype=float)
    for _, row in average.iterrows():
        ax_fni.plot(
            fni_x,
            row[PROFILE_FNI_VARIABLES].to_numpy(float),
            color=MUTED_TEXT,
            alpha=0.075,
            linewidth=0.38,
            zorder=1,
        )
    for position, variable in enumerate(PROFILE_FNI_VARIABLES):
        values = average[variable].dropna().to_numpy(float)
        color = CATEGORY_COLORS[PROFILE_LABELS[variable]]
        violin = ax_fni.violinplot(values, positions=[position], widths=0.78, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.18)
            body.set_linewidth(0.65)
        ax_fni.boxplot(
            values,
            positions=[position],
            widths=0.22,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": TEXT_COLOR, "linewidth": 1.0},
            whiskerprops={"color": color, "linewidth": 0.65},
            capprops={"color": color, "linewidth": 0.65},
            boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 0.75},
        )
        ax_fni.scatter(
            position + average[PROFILE_JITTER_COLUMNS[variable]].to_numpy(float),
            average[variable].to_numpy(float),
            s=3.4,
            color=color,
            alpha=0.14,
            edgecolors="none",
            zorder=3,
        )
    for exemplar, row in selected_rows.items():
        values = row[PROFILE_FNI_VARIABLES].to_numpy(float)
        color = str(EXEMPLARS[exemplar]["color"])
        ax_fni.plot(
            fni_x,
            values,
            color=color,
            linewidth=0.82,
            marker="o",
            markersize=3.1,
            markeredgecolor="white",
            markeredgewidth=0.42,
            zorder=12,
        )
        _badge(ax_fni, 2.25, float(values[-1]), exemplar, color)
    ax_fni.set_xlim(-0.45, 2.45)
    ax_fni.set_xticks(fni_x, [PROFILE_LABELS[item] for item in PROFILE_FNI_VARIABLES], rotation=25, ha="right")
    ax_fni.set_ylabel("Score (−3 novelty to +3 familiarity)")
    ax_fni.set_title("Category FNI", loc="left", pad=4)
    style_axis(ax_fni, grid_axis="y", zero_y=True)

    daily_x = np.arange(len(DAILY_VARIABLES), dtype=float)
    sampled_daily = average.loc[average["daily_line_order"].ge(0)].sort_values("daily_line_order")
    for _, row in sampled_daily.iterrows():
        ax_daily.plot(
            daily_x,
            row[DAILY_VARIABLES].to_numpy(float),
            color="#748394",
            alpha=0.11,
            linewidth=0.44,
            zorder=1,
        )
    daily_values = [average[column].dropna().to_numpy(float) for column in DAILY_VARIABLES]
    ax_daily.boxplot(
        daily_values,
        positions=daily_x,
        widths=0.54,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": TEXT_COLOR, "linewidth": 1.0},
        whiskerprops={"color": "#748394", "linewidth": 0.65},
        capprops={"color": "#748394", "linewidth": 0.65},
        boxprops={"facecolor": "#EEF2F6", "edgecolor": "#748394", "linewidth": 0.75},
    )
    for position, variable in enumerate(DAILY_VARIABLES):
        ax_daily.scatter(
            position + average[PROFILE_JITTER_COLUMNS[variable]].to_numpy(float),
            average[variable].to_numpy(float),
            s=2.8,
            color="#748394",
            alpha=0.10,
            edgecolors="none",
            zorder=3,
        )
    for exemplar, row in selected_rows.items():
        values = row[DAILY_VARIABLES].to_numpy(float)
        color = str(EXEMPLARS[exemplar]["color"])
        ax_daily.plot(
            daily_x,
            values,
            color=color,
            linewidth=0.78,
            marker="o",
            markersize=3.0,
            markeredgecolor="white",
            markeredgewidth=0.42,
            zorder=12,
        )
        _badge(ax_daily, 7.34, float(values[-1]), exemplar, color)
    ax_daily.set_xlim(-0.5, 7.58)
    ax_daily.set_xticks(daily_x, [PROFILE_LABELS[item] for item in DAILY_VARIABLES], rotation=25, ha="right")
    ax_daily.set_title("Daily familiarity/novelty items", loc="left", pad=4)
    ax_daily.tick_params(axis="y", labelleft=False)
    style_axis(ax_daily, grid_axis="y", zero_y=True)
    return figure


def _crop_page(page: PageObject, bottom_fraction: float, top_fraction: float) -> PageObject:
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    bottom = bottom_fraction * height
    top = top_fraction * height
    clipped_source = copy.deepcopy(page)
    clipped_source.cropbox.lower_left = (0.0, bottom)
    clipped_source.cropbox.upper_right = (width, top)
    cropped = PageObject.create_blank_page(width=width, height=top - bottom)
    cropped.merge_transformed_page(
        clipped_source,
        Transformation().translate(tx=0.0, ty=-bottom),
        over=True,
    )
    return cropped


def render_main_figure_02(data_root: Path, output_dir: Path) -> Path:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    trajectory = _read_csv(
        data_root / "figure02/observed_trajectory.csv",
        ["broad_category", "trial_within_subcategory", "observed_mean", "ci_low", "ci_high"],
    )
    profiles = _read_csv(
        data_root / "figure02/participant_profiles.csv",
        [
            "manuscript_profile_id",
            "analysis_record_id",
            "plot_order",
            "daily_line_order",
            "average_rug_unit",
            "average_category_fni_daily_nf",
            *PROFILE_VARIABLES,
            *PROFILE_JITTER_COLUMNS.values(),
        ],
    )

    observed_figure = _draw_observed_trajectory(trajectory)
    summary_figure = _draw_selected_profile_summary(profiles)
    for figure in (observed_figure, summary_figure):
        declared = getattr(figure, "_manuscript_size_mm", None)
        if declared is not None:
            figure.set_size_inches(float(declared[0]) / 25.4, float(declared[1]) / 25.4, forward=False)
    observed_buffer = BytesIO()
    summary_buffer = BytesIO()
    observed_figure.savefig(
        observed_buffer,
        format="pdf",
        bbox_inches=None,
        metadata={"CreationDate": None, "ModDate": None},
    )
    summary_figure.savefig(
        summary_buffer,
        format="pdf",
        bbox_inches=None,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(observed_figure)
    plt.close(summary_figure)
    observed_buffer.seek(0)
    summary_buffer.seek(0)
    source_observed = PdfReader(observed_buffer).pages[0]
    source_summary = _crop_page(PdfReader(summary_buffer).pages[0], 0.45, 1.0)

    target_width = 4.8 * 72.0
    gap = 0.08 * 72.0
    observed_scale = target_width / float(source_observed.mediabox.width)
    summary_scale = target_width / float(source_summary.mediabox.width)
    observed_height = float(source_observed.mediabox.height) * observed_scale
    summary_height = float(source_summary.mediabox.height) * summary_scale
    target_height = observed_height + gap + summary_height
    composite = PageObject.create_blank_page(width=target_width, height=target_height)
    composite.merge_transformed_page(
        source_summary,
        Transformation().scale(summary_scale).translate(tx=0.0, ty=0.0),
        over=True,
    )
    composite.merge_transformed_page(
        source_observed,
        Transformation().scale(observed_scale).translate(tx=0.0, ty=summary_height + gap),
        over=True,
    )

    label_figure = plt.figure(figsize=(target_width / 72.0, target_height / 72.0), facecolor="none")
    label_figure.patch.set_alpha(0.0)
    label_y_a = 1.0 - (0.035 * observed_height / target_height)
    label_y_b = 0.965 * summary_height / target_height
    for label, y_position in (("a", label_y_a), ("b", label_y_b)):
        label_figure.text(
            0.015,
            y_position,
            label,
            ha="left",
            va="top",
            fontsize=8.0,
            fontfamily="Arial",
            fontweight="bold",
            color="#222222",
        )
    label_buffer = BytesIO()
    label_figure.savefig(
        label_buffer,
        format="pdf",
        transparent=True,
        bbox_inches=None,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(label_figure)
    label_buffer.seek(0)
    composite.merge_page(PdfReader(label_buffer).pages[0], over=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "main_figure_02.pdf"
    writer = PdfWriter()
    writer.add_page(composite)
    writer.add_metadata({"/Title": "Observed trajectories and participant-level distributions"})
    with output.open("wb") as stream:
        writer.write(stream)
    return output


# ---------------------------------------------------------------------------
# Main Figures 3 and 4


def render_main_figure_03(data_root: Path, output_dir: Path) -> Path:
    data_root = Path(data_root)
    statistics_a = _read_csv(data_root / "figure03/category_questionnaire_bh12.csv")
    scatter = _read_csv(data_root / "figure03/face_scatter_display.csv")
    statistics_b = _read_csv(data_root / "figure03/category_daily_bh24.csv")
    if len(statistics_a) != 12 or len(statistics_b) != 24 or len(scatter) != 459:
        raise ValueError("Figure 3 frozen source row counts changed")

    figure = new_figure(4.8 * 25.4, 4.45 * 25.4)
    _panel_heading(
        figure,
        "a",
        "Category FNI and questionnaire associations",
        "Matched Tasks 3 + 5, n = 153 · Spearman ρ; BH-12 · supplementary",
        0.985,
    )
    rho_a = _matrix(
        statistics_a,
        row="behavior_variable",
        column="questionnaire_variable",
        value="spearman_rho",
        row_order=BEHAVIOR_VARIABLES,
        column_order=QUESTIONNAIRE_VARIABLES,
    )
    q_a = _matrix(
        statistics_a,
        row="behavior_variable",
        column="questionnaire_variable",
        value="q_value",
        row_order=BEHAVIOR_VARIABLES,
        column_order=QUESTIONNAIRE_VARIABLES,
    )
    _draw_heatmap(
        figure,
        rho_a,
        q_a,
        position=[0.185, 0.790, 0.690, 0.120],
        colorbar_position=[0.900, 0.790, 0.014, 0.120],
        x_labels=[LABELS[item] for item in QUESTIONNAIRE_VARIABLES],
        y_labels=[LABELS[item] for item in BEHAVIOR_VARIABLES],
        rotation=24,
        annotation_size=4.3,
        tick_size=4.15,
        highlight_row=0,
    )
    scatter_grid = figure.add_gridspec(
        1,
        3,
        left=0.075,
        right=0.985,
        bottom=0.505,
        top=0.690,
        wspace=0.34,
    )
    selected = statistics_a.loc[
        statistics_a["behavior_variable"].eq("face_fni")
        & statistics_a["questionnaire_variable"].isin(SCATTER_VARIABLES)
    ].set_index("questionnaire_variable")
    x_min = float(scatter["observed_face_fni"].min())
    x_max = float(scatter["observed_face_fni"].max())
    x_pad = 0.05 * (x_max - x_min)
    for index, variable in enumerate(SCATTER_VARIABLES):
        axis = figure.add_subplot(scatter_grid[0, index])
        panel = scatter.loc[scatter["questionnaire_variable"].eq(variable)]
        row = selected.loc[variable]
        _draw_scatter_cell(
            axis,
            plot_x=panel["plot_face_fni"].to_numpy(float),
            plot_y=panel["plot_questionnaire_value"].to_numpy(float),
            observed_x=panel["observed_face_fni"].to_numpy(float),
            observed_y=panel["observed_questionnaire_value"].to_numpy(float),
            rho=float(row["spearman_rho"]),
            q_value=float(row["q_value"]),
            zero_x=True,
            annotation_size=3.65,
        )
        axis.set_xlim(x_min - x_pad, x_max + x_pad)
        axis.set_xticks([-1.5, 0.0, 1.5])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.tick_params(axis="both", labelsize=3.65, pad=1.0)
        axis.set_title(LABELS[variable], loc="left", fontsize=4.8, fontweight="bold", pad=2.0)
    figure.text(
        0.53,
        0.455,
        "Face FNI  (novelty ← 0 → familiarity)",
        ha="center",
        va="bottom",
        fontsize=4.35,
        color=TEXT_COLOR,
    )
    _panel_heading(
        figure,
        "b",
        "Category FNI and daily familiarity items",
        "Matched Tasks 3 + 5, n = 153 · Spearman ρ; BH-24 · exploratory",
        0.415,
    )
    rho_b = _matrix(
        statistics_b,
        row="behavior_variable",
        column="daily_item_variable",
        value="spearman_rho",
        row_order=BEHAVIOR_VARIABLES,
        column_order=DAILY_VARIABLES,
    )
    q_b = _matrix(
        statistics_b,
        row="behavior_variable",
        column="daily_item_variable",
        value="q_value",
        row_order=BEHAVIOR_VARIABLES,
        column_order=DAILY_VARIABLES,
    )
    _draw_heatmap(
        figure,
        rho_b,
        q_b,
        position=[0.145, 0.145, 0.730, 0.145],
        colorbar_position=[0.900, 0.145, 0.014, 0.145],
        x_labels=[LABELS[item] for item in DAILY_VARIABLES],
        y_labels=[LABELS[item] for item in BEHAVIOR_VARIABLES],
        rotation=32,
        annotation_size=3.25,
        tick_size=3.55,
    )
    return _save_pdf(figure, Path(output_dir) / "main_figure_03.pdf")


def _diagonal_values(pair_data: pd.DataFrame, variable: str) -> np.ndarray:
    x_match = pair_data.loc[pair_data["x_variable"].eq(variable), "observed_x"]
    if not x_match.empty:
        first_pair = pair_data.loc[pair_data["x_variable"].eq(variable), "y_variable"].iloc[0]
        values = pair_data.loc[
            pair_data["x_variable"].eq(variable) & pair_data["y_variable"].eq(first_pair),
            "observed_x",
        ]
    else:
        first_pair = pair_data.loc[pair_data["y_variable"].eq(variable), "x_variable"].iloc[0]
        values = pair_data.loc[
            pair_data["y_variable"].eq(variable) & pair_data["x_variable"].eq(first_pair),
            "observed_y",
        ]
    if len(values) != 303:
        raise ValueError(f"Could not reconstruct 303 diagonal observations for {variable}")
    return values.to_numpy(float)


def render_main_figure_04(data_root: Path, output_dir: Path) -> Path:
    data_root = Path(data_root)
    pair_statistics = _read_csv(data_root / "figure04/questionnaire_pairwise_bh6.csv")
    pair_data = _read_csv(data_root / "figure04/questionnaire_pairplot_display.csv")
    item_statistics = _read_csv(data_root / "figure04/daily_item_trait_bh24.csv")
    if len(pair_statistics) != 6 or len(pair_data) != 1818 or len(item_statistics) != 24:
        raise ValueError("Figure 4 frozen source row counts changed")

    figure = new_figure(4.8 * 25.4, 6.25 * 25.4)
    _panel_heading(
        figure,
        "a",
        "Questionnaire distributions and pairwise relationships",
        "QC-consistent pooled Experiments 2-5, n = 303 · Spearman ρ; BH-6",
        0.985,
    )
    stats_index = {
        frozenset((str(row.x_variable), str(row.y_variable))): row
        for row in pair_statistics.itertuples(index=False)
    }
    grid = figure.add_gridspec(
        4,
        4,
        left=0.115,
        right=0.985,
        bottom=0.465,
        top=0.885,
        wspace=0.10,
        hspace=0.10,
    )
    for row_index, y_variable in enumerate(PAIR_VARIABLES):
        for column_index, x_variable in enumerate(PAIR_VARIABLES):
            if column_index > row_index:
                continue
            axis = figure.add_subplot(grid[row_index, column_index])
            if row_index == column_index:
                _draw_diagonal(axis, _diagonal_values(pair_data, x_variable), LABELS[x_variable])
            else:
                panel = pair_data.loc[
                    pair_data["x_variable"].eq(x_variable)
                    & pair_data["y_variable"].eq(y_variable)
                ]
                stat = stats_index[frozenset((x_variable, y_variable))]
                _draw_scatter_cell(
                    axis,
                    plot_x=panel["plot_x"].to_numpy(float),
                    plot_y=panel["plot_y"].to_numpy(float),
                    observed_x=panel["observed_x"].to_numpy(float),
                    observed_y=panel["observed_y"].to_numpy(float),
                    rho=float(stat.spearman_rho),
                    q_value=float(stat.q_value),
                    annotation_size=4.05,
                )
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.tick_params(axis="both", labelsize=3.65, pad=0.8)
            if row_index != len(PAIR_VARIABLES) - 1:
                axis.tick_params(axis="x", labelbottom=False)
            else:
                axis.set_xlabel(SHORT_LABELS[x_variable], fontsize=4.45, color=TEXT_COLOR, labelpad=1.8)
            if column_index != 0 or row_index == column_index:
                axis.tick_params(axis="y", labelleft=False)
            else:
                axis.set_ylabel(SHORT_LABELS[y_variable], fontsize=4.45, color=TEXT_COLOR, labelpad=2.0)

    _panel_heading(
        figure,
        "b",
        "Daily familiarity items and trait measures",
        "Same QC-consistent pooled cohort, n = 303 · Spearman ρ; BH-24",
        0.390,
    )
    rho_b = _matrix(
        item_statistics,
        row="item_variable",
        column="trait_variable",
        value="spearman_rho",
        row_order=DAILY_VARIABLES,
        column_order=TRAITS,
    )
    q_b = _matrix(
        item_statistics,
        row="item_variable",
        column="trait_variable",
        value="q_value",
        row_order=DAILY_VARIABLES,
        column_order=TRAITS,
    )
    _draw_heatmap(
        figure,
        rho_b,
        q_b,
        position=[0.190, 0.095, 0.660, 0.220],
        colorbar_position=[0.885, 0.095, 0.014, 0.220],
        x_labels=[LABELS[item] for item in TRAITS],
        y_labels=[LABELS[item] for item in DAILY_VARIABLES],
        rotation=18,
        annotation_size=4.15,
        tick_size=4.05,
    )
    return _save_pdf(figure, Path(output_dir) / "main_figure_04.pdf")


# ---------------------------------------------------------------------------
# Main Figure 5


def _draw_hlm_heatmap(
    axis: plt.Axes,
    frame: pd.DataFrame,
    limit: float,
) -> None:
    columns = ["AQ", "Daily familiarity", "Maemuki", "Flow"]
    values = frame.pivot(index="category", columns="moderator", values="estimate_per_sd").reindex(
        index=CATEGORY_ORDER, columns=columns
    )
    q_values = frame.pivot(index="category", columns="moderator", values="q_hlm_bh12").reindex(
        index=CATEGORY_ORDER, columns=columns
    )
    robust = frame.pivot(index="category", columns="moderator", values="survives_cr1_bh12").reindex(
        index=CATEGORY_ORDER, columns=columns
    )
    if values.isna().any().any() or q_values.isna().any().any() or robust.isna().any().any():
        raise ValueError("Figure 5 moderation heatmap contains missing cells")
    for row in range(len(CATEGORY_ORDER)):
        for column in range(len(columns)):
            value = float(values.iloc[row, column])
            axis.add_patch(
                Rectangle(
                    (column, row),
                    1,
                    1,
                    facecolor=DIVERGING_CMAP(np.clip((value / limit + 1) / 2, 0, 1)),
                    edgecolor="white",
                    linewidth=0.7,
                )
            )
            if bool(robust.iloc[row, column]):
                axis.add_patch(
                    Rectangle(
                        (column + 0.025, row + 0.025),
                        0.95,
                        0.95,
                        fill=False,
                        edgecolor=TEXT_COLOR,
                        linewidth=1.05,
                        zorder=5,
                    )
                )
            star = "*" if float(q_values.iloc[row, column]) < 0.05 else ""
            axis.text(
                column + 0.5,
                row + 0.5,
                f"{value:+.3f}{star}",
                ha="center",
                va="center",
                fontsize=5.7,
                fontweight="bold" if star else "normal",
                color="white" if abs(value) > 0.72 * limit else TEXT_COLOR,
                zorder=6,
            )
    axis.set_xlim(0, len(columns))
    axis.set_ylim(len(CATEGORY_ORDER), 0)
    axis.set_xticks(np.arange(len(columns)) + 0.5, columns)
    axis.set_yticks(np.arange(len(CATEGORY_ORDER)) + 0.5, [f"{item} trajectory" for item in CATEGORY_ORDER])
    axis.tick_params(axis="x", rotation=26, labelsize=5.2, pad=2)
    for tick in axis.get_xticklabels():
        tick.set_ha("right")
    axis.tick_params(axis="y", labelsize=5.5, pad=3)
    axis.tick_params(axis="both", length=0, colors=TEXT_COLOR)
    for spine in axis.spines.values():
        spine.set_color(HAIRLINE_COLOR)
        spine.set_linewidth(0.55)


def _draw_hlm_colorbar(axis: plt.Axes, limit: float) -> None:
    steps = 64
    edges = np.linspace(-limit, limit, steps + 1)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        axis.add_patch(
            Rectangle(
                (0, low),
                1,
                high - low,
                facecolor=DIVERGING_CMAP((index + 0.5) / steps),
                edgecolor="none",
            )
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(-limit, limit)
    axis.set_xticks([])
    axis.set_yticks([-limit, 0, limit])
    axis.set_yticklabels([f"{value:+.3f}" for value in [-limit, 0, limit]])
    axis.yaxis.tick_right()
    axis.yaxis.set_label_position("right")
    axis.tick_params(axis="y", labelsize=4.7, length=1.5, colors=MUTED_TEXT)
    for spine in axis.spines.values():
        spine.set_color(HAIRLINE_COLOR)
        spine.set_linewidth(0.55)


def _draw_hlm_trajectory(axis: plt.Axes, plot_data: pd.DataFrame) -> None:
    for category in CATEGORY_ORDER:
        subset = plot_data.loc[plot_data["broad_category"].eq(category)].sort_values(
            "trial_within_subcategory"
        )
        x = subset["trial_within_subcategory"].to_numpy(float)
        color = CATEGORY_COLORS[category]
        axis.fill_between(
            x,
            subset["ci_low"].to_numpy(float),
            subset["ci_high"].to_numpy(float),
            color=color,
            alpha=0.10,
            linewidth=0,
        )
        axis.scatter(
            x,
            subset["observed_mean"],
            marker=CATEGORY_MARKERS[category],
            s=12,
            facecolor="white",
            edgecolor=color,
            linewidth=0.65,
            zorder=3,
        )
        axis.plot(
            x,
            subset["fixed_effect_fitted"],
            color=color,
            linestyle=CATEGORY_LINESTYLES[category],
            linewidth=1.25,
        )
    style_axis(axis, grid_axis="y", zero_y=True)
    axis.set_xlim(0.7, 18.3)
    axis.set_ylim(-0.65, 0.65)
    axis.set_yticks(np.arange(-0.6, 0.61, 0.2))
    axis.set_xticks([1, 6, 12, 18])
    axis.set_xlabel("Comparison trial", labelpad=1)
    axis.set_ylabel("")
    arrow_x = -0.085
    text_x = -0.125
    arrow_style = {
        "arrowstyle": "-|>",
        "color": TEXT_COLOR,
        "linewidth": 0.8,
        "mutation_scale": 6.5,
    }
    axis.annotate("", xy=(arrow_x, 0.93), xytext=(arrow_x, 0.54), xycoords=axis.transAxes, arrowprops=arrow_style, annotation_clip=False)
    axis.annotate("", xy=(arrow_x, 0.07), xytext=(arrow_x, 0.46), xycoords=axis.transAxes, arrowprops=arrow_style, annotation_clip=False)
    axis.text(text_x, 0.735, "Familiarity", rotation=90, ha="center", va="center", fontsize=5.0, color=TEXT_COLOR, transform=axis.transAxes, clip_on=False)
    axis.text(text_x, 0.265, "Novelty", rotation=90, ha="center", va="center", fontsize=5.0, color=TEXT_COLOR, transform=axis.transAxes, clip_on=False)
    handles = [
        mlines.Line2D(
            [],
            [],
            color=CATEGORY_COLORS[category],
            linestyle=CATEGORY_LINESTYLES[category],
            marker=CATEGORY_MARKERS[category],
            markerfacecolor="white",
            markeredgecolor=CATEGORY_COLORS[category],
            linewidth=1.1,
            markersize=3.0,
            label=category,
        )
        for category in CATEGORY_ORDER
    ]
    axis.legend(
        handles=handles,
        loc="upper right",
        ncol=3,
        fontsize=3.9,
        handlelength=1.8,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    axis.tick_params(labelsize=4.1)


def render_main_figure_05(data_root: Path, output_dir: Path) -> Path:
    data_root = Path(data_root)
    trajectory = _read_csv(data_root / "figure05/trajectory_observed_and_fitted.csv")
    overall = _read_csv(data_root / "figure05/moderator_screen_bh12.csv")
    if len(trajectory) != 54 or len(overall) != 12:
        raise ValueError("Figure 5 frozen source row counts changed")
    figure = new_figure(4.8 * 25.4, 126.0)
    add_title_block(
        figure,
        "Visual familiarity trajectories and their psychological moderators",
        "Matched Experiments 3 + 5 · n = 153; 33,048 trials",
        left=0.07,
        title_size=8.0,
        subtitle_size=5.35,
        line_gap=0.044,
    )
    limit = 0.015
    trajectory_axis = figure.add_axes([0.17, 0.505, 0.75, 0.255])
    _draw_hlm_trajectory(trajectory_axis, trajectory)
    figure.text(0.025, 0.835, "a", fontsize=8.0, fontweight="bold", color=TEXT_COLOR)
    figure.text(0.075, 0.835, "Observed and fixed-effect fitted category trajectories", fontsize=6.0, fontweight="bold", color=TEXT_COLOR)
    figure.text(0.075, 0.808, "Participant-balanced means [95% CI] and frozen fallback fits", fontsize=4.55, color=MUTED_TEXT)

    overall_axis = figure.add_axes([0.205, 0.135, 0.675, 0.165])
    _draw_hlm_heatmap(overall_axis, overall, limit)
    figure.text(0.025, 0.405, "b", fontsize=8.0, fontweight="bold", color=TEXT_COLOR)
    figure.text(0.075, 0.405, "Complete overall-moderator screen · BH across 12 tests", fontsize=6.0, fontweight="bold", color=TEXT_COLOR)
    figure.text(0.075, 0.378, "Moderator × comparison-position coefficients; positive values indicate more familiarity-directed change", fontsize=4.55, color=MUTED_TEXT)
    colorbar_axis = figure.add_axes([0.900, 0.135, 0.014, 0.165])
    _draw_hlm_colorbar(colorbar_axis, limit)
    figure.text(0.885, 0.315, "b / SD / step", fontsize=4.6, color=TEXT_COLOR, ha="left", va="bottom")
    add_figure_note(
        figure,
        "Panel a: points are participant-balanced means, bands are 95% participant intervals and lines are frozen fallback-model fits.\n"
        "Panel b: * HLM BH q < .05; dark outline also CR1 BH q < .05. Coefficients are per 1-SD moderator increase.",
        left=0.07,
        bottom=0.018,
        size=4.65,
    )
    return _save_pdf(figure, Path(output_dir) / "main_figure_05.pdf")


def render_all(data_root: Path, output_dir: Path) -> list[Path]:
    """Render Main Figures 2--5 and return the four PDF paths in order."""

    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        render_main_figure_02(data_root, output_dir),
        render_main_figure_03(data_root, output_dir),
        render_main_figure_04(data_root, output_dir),
        render_main_figure_05(data_root, output_dir),
    ]
    return outputs


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]
    for pdf in render_all(repository_root / "data/main", repository_root / "outputs/figures/main"):
        print(pdf)
