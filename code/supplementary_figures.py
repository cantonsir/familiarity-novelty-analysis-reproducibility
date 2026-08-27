"""Reproduce the eleven active NHB supplementary figures from plot-ready data.

The inputs in ``data/supplement`` are frozen, privacy-reduced analysis tables.
No task/session export is required.  Geometry and styling below are the values
used by the active manuscript assets, including the two reader-facing crop/
relabel variants (Supplementary Figures 5 and 6).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter
from scipy.stats import pearsonr, spearmanr

from style import (
    CATEGORY_COLORS,
    CATEGORY_LINESTYLES,
    CATEGORY_MARKERS,
    GRID_COLOR,
    HAIRLINE_COLOR,
    MODE_COLORS,
    MUTED_TEXT,
    STATE_COLORS,
    SURFACE_COLOR,
    TEXT_COLOR,
    WHITE,
    ZERO_COLOR,
    add_figure_note,
    add_panel_heading,
    add_panel_label,
    add_title_block,
    set_publication_style,
    style_axis,
)


PAGE_POINTS = {
    1: (510.236, 459.213),
    2: (509.760, 297.600),
    3: (510.236, 297.638),
    4: (510.236, 566.929),
    5: (518.400, 146.880),
    6: (237.600, 189.360),
    7: (509.760, 260.640),
    8: (510.236, 328.819),
    9: (510.236, 232.441),
    10: (345.600, 547.200),
    11: (509.760, 334.080),
}


def _new_page(number: int) -> plt.Figure:
    """Create the exact active PDF artboard in points."""

    set_publication_style()
    width, height = PAGE_POINTS[number]
    figure = plt.figure(figsize=(width / 72.0, height / 72.0), facecolor="white")
    # Matplotlib 3.10 rounds the tuple passed to ``figure(figsize=...)`` to
    # 0.01 inch; setting each dimension afterwards preserves the source PDF's
    # exact point-size artboard.
    figure.set_figwidth(width / 72.0)
    figure.set_figheight(height / 72.0)
    return figure


def _save(figure: plt.Figure, output: Path, title: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="pdf",
        facecolor="white",
        metadata={"Title": title, "Creator": "NHB figure reproducibility package"},
    )
    plt.close(figure)
    return output


def render_figure_01(data_root: Path, output: Path) -> Path:
    """Participant demographics across Experiments 2-5."""

    demographics_root = data_root / "figure_01_demographics"
    age_points = pd.read_csv(demographics_root / "age_points.csv")
    gender_counts = pd.read_csv(demographics_root / "gender_counts.csv")
    race_counts = pd.read_csv(demographics_root / "race_counts.csv")
    age_points["experiment"] = age_points["experiment"].astype(int)
    gender_counts["experiment"] = gender_counts["experiment"].astype(int)
    experiments = [2, 3, 4, 5]
    expected = {2: 24, 3: 22, 4: 126, 5: 131}
    race_order = [
        "Asian",
        "Black / African American",
        "Hispanic / Latino",
        "White / Caucasian",
        "Multiracial / Mixed",
        "Other",
        "Prefer not to say",
    ]
    figure = _new_page(1)
    add_title_block(
        figure,
        "Participant demographics across Experiments 2–5",
        "Primary integrated analysis set (n = 303) · self-reported task fields · descriptive summaries",
        left=0.065,
        top=0.975,
        title_size=9.2,
        subtitle_size=6.2,
        line_gap=0.047,
    )
    grid = figure.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.985,
        bottom=0.095,
        top=0.845,
        height_ratios=[1.0, 1.22],
        width_ratios=[0.82, 1.52],
        hspace=0.50,
        wspace=0.34,
    )

    age_axis = figure.add_subplot(grid[0, :])
    arrays = [age_points.loc[age_points["experiment"].eq(exp), "age"].astype(float).to_numpy() for exp in experiments]
    age_axis.boxplot(
        arrays,
        positions=np.arange(4),
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": TEXT_COLOR, "linewidth": 1.2},
        boxprops={"facecolor": "#DCEAF8", "edgecolor": STATE_COLORS["Primary"], "linewidth": 0.8},
        whiskerprops={"color": STATE_COLORS["Primary"], "linewidth": 0.8},
        capprops={"color": STATE_COLORS["Primary"], "linewidth": 0.8},
    )
    for position, experiment in enumerate(experiments):
        subset = age_points.loc[age_points["experiment"].eq(experiment)].copy()
        jitter = subset["age_jitter"].to_numpy(float)
        point_color = MODE_COLORS["Laboratory"] if experiment in (2, 3) else MODE_COLORS["Online"]
        age_axis.scatter(
            position + jitter,
            subset["age"],
            s=7.5,
            facecolor=WHITE,
            edgecolor=point_color,
            linewidth=0.45,
            alpha=0.72,
            zorder=3,
        )
        ages = subset["age"].astype(float)
        age_axis.text(
            position,
            83.0,
            f"n={len(ages)}  ·  {ages.mean():.1f}±{ages.std(ddof=1):.1f}",
            ha="center",
            va="bottom",
            fontsize=5.25,
            color=MUTED_TEXT,
        )
    age_axis.set_xticks(np.arange(4), ["E2\nLaboratory", "E3\nLaboratory", "E4\nOnline", "E5\nOnline"])
    age_axis.set_xlim(-0.55, 3.55)
    age_axis.set_ylim(15, 88)
    age_axis.set_yticks([20, 40, 60, 80])
    age_axis.set_ylabel("Age (years)")
    add_panel_heading(age_axis, "Age distributions")
    add_panel_label(age_axis, "a", x=-0.060, y=1.03)
    style_axis(age_axis, grid_axis="y")

    gender_axis = figure.add_subplot(grid[1, 0])
    y_positions = np.arange(4)
    gender = gender_counts.set_index("experiment").loc[experiments]
    female_n = gender["female_n"].to_numpy(int)
    female_pct = 100.0 * female_n / np.array([expected[exp] for exp in experiments])
    male_n = gender["male_n"].to_numpy(int)
    male_pct = 100.0 - female_pct
    bars_f = gender_axis.barh(y_positions, female_pct, height=0.58, color=STATE_COLORS["Novelty"], edgecolor=WHITE, linewidth=0.6, label="Female")
    bars_m = gender_axis.barh(y_positions, male_pct, left=female_pct, height=0.58, color=STATE_COLORS["Familiarity"], edgecolor=WHITE, linewidth=0.6, hatch="////", label="Male")
    for index in range(4):
        for bars, count, percent in ((bars_f, female_n[index], female_pct[index]), (bars_m, male_n[index], male_pct[index])):
            bar = bars[index]
            gender_axis.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f"{count}\n{percent:.0f}%", ha="center", va="center", fontsize=5.1, color=WHITE, linespacing=0.92)
    gender_axis.set_yticks(y_positions, [f"E{exp}  n={expected[exp]}" for exp in experiments])
    gender_axis.invert_yaxis()
    gender_axis.set_xlim(0, 100)
    gender_axis.set_xticks([0, 50, 100])
    gender_axis.set_xlabel("Participants (%)")
    add_panel_heading(gender_axis, "Recorded gender at birth")
    add_panel_label(gender_axis, "b", x=-0.18, y=1.03)
    gender_axis.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, handlelength=1.5, columnspacing=1.0)
    style_axis(gender_axis, grid_axis="x")

    race_axis = figure.add_subplot(grid[1, 1])
    columns = ["2", "3", "4", "5", "Pooled"]
    counts = np.zeros((len(race_order), len(columns)), dtype=int)
    denominators = np.array([24, 22, 126, 131, 303])
    race_lookup = race_counts.set_index(["race_response", "experiment_label"])["count"]
    for row_index, race in enumerate(race_order):
        for col_index, column in enumerate(columns):
            counts[row_index, col_index] = int(race_lookup.loc[(race, column)])
    matrix = 100.0 * counts / denominators
    cmap = LinearSegmentedColormap.from_list("demographic_blue", [WHITE, SURFACE_COLOR, "#DCEAF8", "#76A9DC", STATE_COLORS["Primary"]], N=256)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            race_axis.add_patch(Rectangle((column_index - 0.5, row_index - 0.5), 1.0, 1.0, facecolor=cmap(np.clip(matrix[row_index, column_index] / 80.0, 0.0, 1.0)), edgecolor=WHITE, linewidth=1.0))
            percent = matrix[row_index, column_index]
            race_axis.text(column_index, row_index, f"{counts[row_index, column_index]}\n({percent:.0f}%)", ha="center", va="center", fontsize=4.6, color=WHITE if percent >= 44 else TEXT_COLOR, linespacing=0.92)
    race_axis.set_xlim(-0.5, 4.5)
    race_axis.set_ylim(6.5, -0.5)
    race_axis.set_xticks(np.arange(5), ["E2", "E3", "E4", "E5", "Pooled"])
    race_axis.set_yticks(np.arange(7), race_order)
    for spine in race_axis.spines.values():
        spine.set_color(HAIRLINE_COLOR)
        spine.set_linewidth(0.5)
    add_panel_heading(race_axis, "Recorded race response, n (within-experiment %)")
    add_panel_label(race_axis, "c", x=-0.08, y=1.03)
    add_figure_note(figure, "Percentages are descriptive. Race-response labels follow the task-code categories. Zeros are observed counts.", left=0.075, bottom=0.024, size=5.15)
    return _save(figure, output, "Participant demographics across Experiments 2-5")


def _trajectory(
    figure: plt.Figure,
    data: pd.DataFrame,
    *,
    categories: list[str],
    x_column: str,
    xlim: tuple[float, float],
    xticks: list[int],
    xlabel: str,
    ylabel: str,
    title: str,
    subtitle: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    colors = {**CATEGORY_COLORS, "Car": "#D55E00"}
    markers = {**CATEGORY_MARKERS, "Car": "D"}
    linestyles = {**CATEGORY_LINESTYLES, "Car": ":"}
    axis = figure.add_subplot(111)
    for category in categories:
        subset = data.loc[data.iloc[:, 0].eq(category)].sort_values(x_column)
        x = subset[x_column].to_numpy(float)
        axis.fill_between(x, subset["ci_low"].to_numpy(float), subset["ci_high"].to_numpy(float), color=colors[category], alpha=0.12, linewidth=0)
        axis.plot(x, subset["observed_mean"], color=colors[category], linestyle=linestyles[category], marker=markers[category], markerfacecolor="white", markeredgecolor=colors[category], markeredgewidth=0.8, markersize=4.3, linewidth=1.8, label=category)
    style_axis(axis, grid_axis="y", zero_y=True)
    axis.set_xlim(*xlim)
    axis.set_xticks(xticks)
    if y_limits is not None:
        axis.set_ylim(*y_limits)
        axis.set_yticks([-2, -1, 0, 1])
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(loc="upper right", ncol=2 if len(categories) == 4 else 3, frameon=False, title=None, handlelength=2.0, columnspacing=1.1)
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.15, top=0.78)
    add_title_block(figure, title, subtitle)


def render_figure_02(data_root: Path, output: Path) -> Path:
    data = pd.read_csv(data_root / "figure_02_experiment1_trajectory/plot_data.csv")
    figure = _new_page(2)
    _trajectory(
        figure,
        data,
        categories=["Face", "Scenery", "Geometry", "Car"],
        x_column="comparison_position",
        xlim=(0.7, 26.3),
        xticks=[1, 6, 12, 18, 24, 26],
        xlabel="Comparison position",
        ylabel="Familiar-target preference (−3 to +3)",
        title="Experiment 1 four-category preference trajectories",
        subtitle="Pilot laboratory cohort (n = 15) · participant-balanced mean and 95% t CI",
        y_limits=(-2, 1),
    )
    return _save(figure, output, "Experiment 1 four-category preference trajectories")


def render_figure_03(data_root: Path, output: Path) -> Path:
    data = pd.read_csv(data_root / "figure_03_experiment4_trajectory/plot_data.csv")
    figure = _new_page(3)
    _trajectory(
        figure,
        data,
        categories=["Face", "Scenery", "Geometry"],
        x_column="exposure_index",
        xlim=(0.7, 18.3),
        xticks=[1, 6, 12, 18],
        xlabel="Comparison trial",
        ylabel="Familiarity preference (−3 to +3)",
        title="Experiment 4 category trajectories",
        subtitle="Online cohort (n = 126) · 5th-ranked familiar target · participant-balanced mean and 95% t CI",
    )
    return _save(figure, output, "Experiment 4 category trajectories")


def _pattern_heading(figure: plt.Figure, label: str, title: str, subtitle: str, y: float) -> None:
    figure.text(0.014, y, label, ha="left", va="top", fontsize=9.25, fontweight="bold", color=TEXT_COLOR)
    figure.text(0.055, y, title, ha="left", va="top", fontsize=8.4, fontweight="bold", color=TEXT_COLOR)
    figure.text(0.055, y - 0.031, subtitle, ha="left", va="top", fontsize=6.75, color=MUTED_TEXT)


def _pattern_marginal(axis: plt.Axes, marginal: pd.DataFrame, include_neutral: bool) -> None:
    categories = ["Face", "Geometry", "Scenery"]
    left = np.zeros(3)
    for state in (["Novelty", "Neutral", "Familiarity"] if include_neutral else ["Novelty", "Familiarity"]):
        subset = marginal.loc[marginal.state.eq(state)].set_index("category")
        counts = np.array([int(subset.loc[c, "n"]) for c in categories])
        values = np.array([float(subset.loc[c, "percent"]) for c in categories])
        bars = axis.barh(categories, values, left=left, height=0.56, color=STATE_COLORS[state], edgecolor="white", linewidth=0.65)
        for bar, count, value in zip(bars, counts, values, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f"{count}\n{value:.0f}%", ha="center", va="center", fontsize=6.75, linespacing=0.92, color=TEXT_COLOR if state == "Neutral" else "white", fontweight="bold")
        left += values
    axis.set_xlim(0, 100)
    axis.set_xticks([0, 50, 100])
    axis.set_xlabel("Participants (%)", fontsize=7.0, labelpad=3)
    axis.set_title("Marginal category states", loc="left", fontsize=7.1, fontweight="bold", pad=4)
    axis.invert_yaxis()
    axis.tick_params(labelsize=6.75)
    style_axis(axis, grid_axis="x")


def _pattern_bars(axis: plt.Axes, patterns: pd.DataFrame, sign_only: bool) -> None:
    ordered = patterns.sort_values(["n", "profile_pattern"], ascending=[False, False]).reset_index(drop=True)
    y = np.arange(len(ordered), dtype=float)
    axis.barh(y, ordered["n"].to_numpy(float), color="#A1A1A6", edgecolor="#6E6E73", linewidth=0.40, height=0.68)
    axis.set_yticks(y, ordered["profile_pattern"].astype(str))
    axis.invert_yaxis()
    max_n = float(ordered["n"].max())
    axis.set_xlim(0, max_n * (1.34 if sign_only else 1.43))
    axis.set_xticks([0, 10, 20, 30, 40] if sign_only else [0, 5, 10, 15, 20])
    axis.set_xlabel("Participants", fontsize=7.0, labelpad=3)
    axis.set_title("Exact patterns (Face / Geometry / Scenery)", loc="left", fontsize=7.1, fontweight="bold", pad=4)
    axis.tick_params(labelsize=6.75)
    for index, row in ordered.iterrows():
        axis.text(float(row["n"]) + max_n * 0.025, index, f"{int(row['n'])} ({float(row['percent']):.1f}%)", va="center", ha="left", fontsize=6.75, color=TEXT_COLOR)
    style_axis(axis, grid_axis="x")


def render_figure_04(data_root: Path, output: Path) -> Path:
    marginal = pd.read_csv(data_root / "figure_04_fni_patterns/marginal_counts.csv")
    patterns = pd.read_csv(data_root / "figure_04_fni_patterns/pattern_counts.csv", dtype={"profile_pattern": "string"})
    figure = _new_page(4)
    _pattern_heading(figure, "a", "Sign-only FNI patterns", "E3/E5 all-completers (n = 155) | N: FNI < 0; F: FNI >= 0 | descriptive bins, not latent classes", 0.986)
    _pattern_marginal(figure.add_axes([0.080, 0.672, 0.305, 0.225]), marginal.loc[marginal.panel.eq("a")], False)
    _pattern_bars(figure.add_axes([0.480, 0.672, 0.490, 0.225]), patterns.loc[patterns.panel.eq("a")], True)
    _pattern_heading(figure, "b", "FNI patterns with a +/-0.20 neutral band", "Same participants (n = 155) | N: FNI < -0.20; 0: -0.20 <= FNI <= +0.20; F: FNI > +0.20 | descriptive bins", 0.607)
    _pattern_marginal(figure.add_axes([0.080, 0.112, 0.305, 0.402]), marginal.loc[marginal.panel.eq("b")], True)
    _pattern_bars(figure.add_axes([0.480, 0.112, 0.490, 0.402]), patterns.loc[patterns.panel.eq("b")], False)
    handles = [mpatches.Patch(facecolor=STATE_COLORS[state], edgecolor="none", label=f"{symbol}  {state}") for state, symbol in [("Novelty", "N"), ("Neutral", "0"), ("Familiarity", "F")]]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.014), ncol=3, frameon=False, fontsize=6.75, handlelength=1.6, columnspacing=2.2)
    figure.text(0.50, 0.063, "Gray exact-pattern bars encode frequency only; state meaning is carried by N / 0 / F.", ha="center", va="bottom", fontsize=6.75, color=MUTED_TEXT)
    return _save(figure, output, "Descriptive FNI pattern counts")


def render_figure_05(data_root: Path, output: Path) -> Path:
    """Native version of the active lower-profile crop (A=P092, B=P101)."""

    long = pd.read_csv(data_root / "figure_05_selected_profiles/selected_profiles_long.csv")
    variables = [
        "face_fni", "geometry_fni", "scenery_fni", "4_food/cuisine", "4_places/activities",
        "4_movies or videos", "4_restaurant", "4_clothes", "4_digital games",
        "4_books or magazines", "4_snacks or treats",
    ]
    labels = ["Face", "Geometry", "Scenery", "Food", "Places", "Movies", "Restaurants", "Clothes", "Games", "Books", "Snacks"]
    colors = {"A": "#FF9F0A", "B": "#0071E3"}
    # The active asset is the lower 2.04 inches of the original 7.2 x 4.8 inch
    # vector figure.  Recreate that source geometry, then crop the page; this
    # preserves the exact row spacing and typography without retaining hidden
    # off-artboard panels.
    set_publication_style()
    figure = plt.figure(figsize=(7.2, 4.8), facecolor="white")
    lower = figure.add_gridspec(3, 1, left=0.05, right=0.985, bottom=0.100, top=0.436, height_ratios=(0.28, 1.0, 1.0), hspace=0.39)
    header = figure.add_subplot(lower[0, 0])
    axes = [figure.add_subplot(lower[1, 0]), figure.add_subplot(lower[2, 0])]
    header.set_xlim(-0.5, 10.5); header.set_ylim(0, 1); header.axis("off")
    header.text(1.0, 0.48, "Category FNI", ha="center", va="center", fontsize=6.4, fontweight="bold", color="#9A5A16")
    header.text(6.5, 0.48, "Daily F/N items", ha="center", va="center", fontsize=6.4, fontweight="bold", color=STATE_COLORS["Novelty"])
    for axis, exemplar in zip(axes, ["A", "B"], strict=True):
        subset = long.loc[long.exemplar.eq(exemplar)].set_index("variable").loc[variables]
        values = subset["raw_value"].to_numpy(float)
        metadata = subset.iloc[0]
        descriptor = "author-selected familiarity-leaning case" if exemplar == "A" else "author-selected novelty-leaning case"
        axis.axvspan(-0.5, 2.5, color="#FFF4E8", alpha=0.58, zorder=0)
        axis.axvspan(2.5, 10.5, color="#EEF6FF", alpha=0.62, zorder=0)
        axis.axvline(2.5, color=GRID_COLOR, linewidth=0.65, zorder=1)
        axis.plot(np.arange(11), values, color=colors[exemplar], linewidth=0.90, marker="o", markersize=3.2, markeredgecolor="white", markeredgewidth=0.45, zorder=5)
        for x_value, value in enumerate(values):
            offset = -7.5 if value >= 2.55 else 6.0
            axis.annotate(f"{value:+.2f}", (x_value, value), xytext=(0, offset), textcoords="offset points", ha="center", va="top" if value >= 2.55 else "bottom", fontsize=4.25, color=TEXT_COLOR, zorder=6, clip_on=False)
        axis.set_xlim(-0.5, 10.5); axis.set_ylim(-3.35, 3.35); axis.set_yticks([-3, 0, 3]); axis.set_ylabel("F/N score")
        axis.set_title(f"{exemplar}  {metadata['manuscript_profile_id']} — {descriptor} · mean category {float(metadata['mean_category_fni']):+.2f} · average index {float(metadata['average_category_fni_daily_nf']):+.2f}", loc="left", pad=3, color=colors[exemplar], fontsize=6.0)
        style_axis(axis, grid_axis="y", zero_y=True)
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[1].set_xticks(np.arange(11), labels, rotation=25, ha="right")
    output.parent.mkdir(parents=True, exist_ok=True)
    full_page = output.with_name(output.stem + ".full-page.pdf")
    figure.savefig(full_page, format="pdf", facecolor="white", metadata={"Title": "Selected participant profile details", "Creator": "NHB figure reproducibility package"})
    plt.close(figure)
    source_page = PdfReader(full_page).pages[0]
    writer = PdfWriter()
    cropped = writer.add_blank_page(width=PAGE_POINTS[5][0], height=PAGE_POINTS[5][1])
    # The manuscript crop carries a seven-point downward translation relative
    # to the full artboard (confirmed against the active vector asset).
    cropped.merge_translated_page(source_page, 0.0, -7.0)
    writer.add_metadata({"/Title": "Selected participant profile details", "/Creator": "NHB figure reproducibility package"})
    with output.open("wb") as stream:
        writer.write(stream)
    full_page.unlink()
    return output


def render_figure_06(data_root: Path, output: Path) -> Path:
    data = pd.read_csv(data_root / "figure_06_equivalence/plot_data.csv").set_index("category").loc[["Face", "Scenery", "Geometry"]].reset_index()
    figure = _new_page(6)
    axis = figure.add_subplot(111)
    positions = np.arange(3)[::-1]
    axis.axvspan(-0.20, 0.20, color=SURFACE_COLOR, zorder=0)
    axis.axvline(-0.20, color=ZERO_COLOR, linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
    axis.axvline(0.20, color=ZERO_COLOR, linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
    for y, (_, row) in zip(positions, data.iterrows()):
        estimate = float(row["spearman_rho"]); low = float(row["bootstrap_90_ci_low"]); high = float(row["bootstrap_90_ci_high"])
        axis.errorbar(estimate, y, xerr=np.asarray([[estimate - low], [high - estimate]]), capsize=2, elinewidth=0.8, markeredgewidth=0.7, fmt=CATEGORY_MARKERS[row["category"]], color=CATEGORY_COLORS[row["category"]], markerfacecolor=CATEGORY_COLORS[row["category"]] if bool(row["equivalent_after_holm_0_05"]) else "white", markersize=4.0, zorder=3)
    axis.set_yticks(positions, data["category"])
    axis.set_xlim(-0.35, 0.35)
    axis.set_xlabel("")
    axis.set_title("Post hoc equivalence sensitivity", loc="left", pad=3, fontsize=6.5)
    style_axis(axis, grid_axis="x", zero_x=True)
    figure.subplots_adjust(left=0.25, right=0.96, bottom=0.20, top=0.88)
    figure.text(0.5, 19.69 / PAGE_POINTS[6][1], "Category FNI–Daily familiarity Spearman ρ (90% bootstrap CI)", ha="center", va="baseline", fontsize=6.0, color=TEXT_COLOR)
    return _save(figure, output, "Post hoc equivalence sensitivity")


def render_figure_07(data_root: Path, output: Path) -> Path:
    """Pooled questionnaire joint-association diagram."""

    paths = pd.read_csv(data_root / "figure_07_joint_model/model_paths.csv").set_index("path_id")
    fit = pd.read_csv(data_root / "figure_07_joint_model/model_fit.csv").iloc[0]
    blue, orange, text_color, muted = "#0878E8", "#FF9700", "#111111", "#646B73"
    figure = _new_page(7)
    axis = figure.add_axes([0, 0, 1, 1]); axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")
    axis.text(0.035, 0.935, "Pooled questionnaire joint-association model", fontsize=12.5, fontweight="bold", color=text_color, ha="left", va="top")
    axis.text(0.035, 0.825, "Same n = 303 cohort  ·  standardized coefficients [95% stratified-bootstrap CI]  ·  observational", fontsize=7.1, color=muted, ha="left", va="top")
    nodes = [((0.205, 0.600), "Daily\nfamiliarity", False), ((0.205, 0.245), "AQ", False), ((0.555, 0.570), "Flow\n(common 12)", True), ((0.865, 0.385), "Maemuki", True)]
    for center, label, highlighted in nodes:
        width, height = 0.23, 0.145
        axis.add_patch(FancyBboxPatch((center[0] - width / 2, center[1] - height / 2), width, height, boxstyle="round,pad=0.012,rounding_size=0.015", linewidth=1.3, edgecolor=blue if highlighted else "#6F7378", facecolor="#E7F2FF" if highlighted else "#F7F7F9", zorder=3))
        axis.text(*center, label, ha="center", va="center", fontsize=8.2, fontweight="bold", color=text_color, zorder=4, linespacing=1.0)
    def path(start, end, color, width, dashed=False, rad=0.0):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=width, linestyle="--" if dashed else "-", color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=2))
    path((0.325, 0.600), (0.430, 0.580), orange, 1.3)
    path((0.325, 0.285), (0.440, 0.520), blue, 2.6, True)
    path((0.675, 0.535), (0.745, 0.430), orange, 3.2)
    path((0.325, 0.245), (0.745, 0.370), blue, 2.9, True, -0.05)
    axis.add_patch(FancyArrowPatch((0.086, 0.575), (0.086, 0.270), arrowstyle="-", linewidth=2.0, color=orange, connectionstyle="arc3,rad=0.65", zorder=1))
    def label(x, y, line1, line2):
        axis.text(x, y, f"{line1}\n{line2}", ha="center", va="center", fontsize=5.8, color=text_color, linespacing=0.92, bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#CED5DC", linewidth=0.55), zorder=5)
    def path_stat(path_id: str) -> tuple[str, str]:
        row = paths.loc[path_id]
        return rf"$\beta={float(row['standardized_beta']):+.2f}$", f"[{float(row['ci_low']):+.2f}, {float(row['ci_high']):+.2f}]"
    label(0.390, 0.705, *path_stat("daily_to_flow"))
    label(0.397, 0.425, *path_stat("aq_to_flow"))
    label(0.725, 0.635, *path_stat("flow_to_maemuki"))
    label(0.565, 0.185, *path_stat("aq_to_maemuki"))
    correlation = paths.loc["daily_aq_correlation"]
    label(0.090, 0.420, f"Correlation  r = {float(correlation['standardized_beta']):+.2f}", f"[{float(correlation['ci_low']):+.2f}, {float(correlation['ci_high']):+.2f}]")
    axis.text(0.965, 0.055, rf"Fit: $\chi^2({int(fit['df'])})={float(fit['chi_square']):.2f}$, $p={float(fit['p_value']):.3f}$; CFI = {float(fit['cfi']):.3f}; TLI = {float(fit['tli']):.3f}; RMSEA = {float(fit['rmsea']):.3f}; SRMR = {float(fit['srmr']):.3f}".replace("0.", "."), fontsize=5.7, color=muted, ha="right", va="bottom")
    return _save(figure, output, "Pooled questionnaire joint-association model")


def _stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def _format_q(q_value: float) -> str:
    return "q < .001" if q_value < 0.001 else f"q = {q_value:.3f}".replace("0.", ".")


def render_figure_08(data_root: Path, output: Path) -> Path:
    points = pd.read_csv(data_root / "figure_08_negative_capability/plot_data.csv")
    statistics = pd.read_csv(data_root / "figure_08_negative_capability/statistics.csv").set_index("trait")
    specs = [
        ("a", "AQ", (3.0, 45.0)),
        ("b", "Maemuki", (-1.25, 2.65)),
        ("c", "Flow", (-1.30, 3.10)),
        ("d", "Daily familiarity", (-3.25, 3.25)),
    ]
    figure = _new_page(8)
    add_title_block(figure, "Negative Capability and questionnaire measures", "Pooled QC-clean Experiments 3-5, n = 279 · Spearman ρ; BH-4 · exploratory", left=0.065, top=0.965, title_size=8.5, subtitle_size=5.4, line_gap=0.050)
    grid = figure.add_gridspec(2, 2, left=0.085, right=0.985, bottom=0.135, top=0.810, wspace=0.27, hspace=0.48)
    for index, (letter, trait, ylim) in enumerate(specs):
        axis = figure.add_subplot(grid[index // 2, index % 2])
        panel = points.loc[points.trait.eq(trait)]
        x = panel["plot_negative_capability"].to_numpy(float); y = panel["plot_trait_value"].to_numpy(float)
        axis.scatter(x, y, s=7.0, marker="o", color="#547A9A", edgecolors="none", alpha=0.29, zorder=3)
        intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        line_x = np.linspace(-3.0, 3.0, 160)
        axis.plot(line_x, intercept + slope * line_x, color="#FF9F0A", linewidth=0.82, alpha=0.91, zorder=4)
        style_axis(axis, grid_axis="both", zero_x=True)
        axis.set_xlim(-3.25, 3.25); axis.set_ylim(*ylim); axis.set_xticks([-3, 0, 3]); axis.yaxis.set_major_locator(MaxNLocator(nbins=4)); axis.tick_params(axis="both", labelsize=5.4, pad=1.8)
        add_panel_heading(axis, trait, pad=3.0); add_panel_label(axis, letter, x=-0.14, y=1.045, size=8.0)
        row = statistics.loc[trait]; rho = float(row["spearman_rho"]); q_value = float(row["q_value_bh4"])
        axis.text(0.035, 0.955, f"ρ = {rho:+.2f}{_stars(q_value)}; {_format_q(q_value)}", transform=axis.transAxes, ha="left", va="top", fontsize=5.2, color=TEXT_COLOR, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": HAIRLINE_COLOR, "linewidth": 0.4, "alpha": 0.92}, zorder=7)
    figure.text(0.535, 0.055, "Negative Capability score", ha="center", va="bottom", fontsize=6.0, color=TEXT_COLOR)
    return _save(figure, output, "Negative Capability and questionnaire measures")


def render_figure_09(data_root: Path, output: Path) -> Path:
    data = pd.read_csv(data_root / "figure_09_aq_benefits/plot_data.csv")
    statistics = pd.read_csv(data_root / "figure_09_aq_benefits/statistics.csv").set_index("column")
    specs = [("a", "4_1_n_affect_daily", "Novelty-seeking benefit"), ("b", "4_2_f_affect_daily", "Familiarity-seeking benefit")]
    figure = _new_page(9)
    add_title_block(figure, "AQ and perceived benefits of daily novelty and familiarity seeking", "Pooled QC-clean Experiments 2-5, n = 303 · Spearman ρ; BH-3 · secondary", left=0.065, top=0.955, title_size=8.5, subtitle_size=5.4, line_gap=0.065)
    grid = figure.add_gridspec(1, 2, left=0.095, right=0.985, bottom=0.195, top=0.745, wspace=0.24)
    for index, (letter, column, title) in enumerate(specs):
        axis = figure.add_subplot(grid[0, index])
        plot_x = data[f"plot_{column}_x"].to_numpy(float); plot_y = data[f"plot_{column}_aq"].to_numpy(float)
        raw_x = data[column].to_numpy(float); raw_y = data["aq_score"].to_numpy(float)
        axis.scatter(plot_x, plot_y, s=7.0, marker="o", color="#547A9A", edgecolors="none", alpha=0.29, zorder=3)
        intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(raw_x)), raw_x]), raw_y, rcond=None)[0]
        line_x = np.linspace(-3.0, 3.0, 160)
        axis.plot(line_x, intercept + slope * line_x, color="#FF9F0A", linewidth=0.82, alpha=0.91, zorder=4)
        style_axis(axis, grid_axis="both", zero_x=True)
        axis.set_xlim(-3.25, 3.25); axis.set_ylim(2.5, 45.5); axis.set_xticks([-3, 0, 3]); axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True)); axis.tick_params(axis="both", labelsize=5.4, pad=1.8)
        add_panel_heading(axis, title, pad=3.0); add_panel_label(axis, letter, x=-0.14, y=1.045, size=8.0)
        if index == 0: axis.set_ylabel("AQ score", fontsize=6.0)
        else: axis.tick_params(labelleft=False)
        row = statistics.loc[column]; rho = float(row["spearman_rho"]); q_value = float(row["q_value_bh3"])
        axis.text(0.035, 0.955, f"ρ = {rho:+.2f}{_stars(q_value)}; {_format_q(q_value)}", transform=axis.transAxes, ha="left", va="top", fontsize=5.2, color=TEXT_COLOR, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": HAIRLINE_COLOR, "linewidth": 0.4, "alpha": 0.92}, zorder=7)
    figure.text(0.54, 0.070, "Perceived-benefit score (-3 to +3)", ha="center", va="bottom", fontsize=6.0, color=TEXT_COLOR)
    return _save(figure, output, "AQ and perceived benefits of daily novelty and familiarity seeking")


def _hlm_heading(figure: plt.Figure, label: str, title: str, subtitle: str, y: float) -> None:
    figure.text(0.012, y, label, ha="left", va="top", fontsize=7.4, fontweight="bold", color=TEXT_COLOR)
    figure.text(0.062, y, title, ha="left", va="top", fontsize=7.2, fontweight="bold", color=TEXT_COLOR)
    figure.text(0.062, y - 0.025, subtitle, ha="left", va="top", fontsize=4.35, color=MUTED_TEXT)


def _format_probability(value: float) -> str:
    return f"{value:.1e}" if value < 0.001 else f"{value:.3f}".replace("0.", ".")


def _hlm_scatter(axis: plt.Axes, points: pd.DataFrame, guide: pd.DataFrame, term: pd.Series, *, category: str, x_label: str, discrete_x: bool) -> None:
    color = CATEGORY_COLORS[category]
    guide_x = guide["predictor_value"].to_numpy(float)
    axis.fill_between(guide_x, guide["ci_low"].to_numpy(float), guide["ci_high"].to_numpy(float), color=color, alpha=0.10, linewidth=0)
    axis.plot(guide_x, guide["ols_fitted"], color=color, linewidth=1.25)
    for experiment, filled in [(3, False), (5, True)]:
        subset = points.loc[points["experiment"].astype(int).eq(experiment)]
        axis.scatter(subset["x_plot"], subset["empirical_slope"], s=10.5, marker=CATEGORY_MARKERS[category], facecolor=color if filled else "white", edgecolor=color, linewidth=0.55, alpha=0.52 if filled else 0.92, zorder=3)
    style_axis(axis, grid_axis="both", zero_y=True)
    axis.set_ylim(-0.18, 0.18)
    if discrete_x:
        axis.set_xlim(-3.35, 3.35); axis.set_xticks([-3, 0, 3])
    else:
        axis.set_xticks([-2, 0, 1])
    axis.set_xlabel(x_label, labelpad=2); axis.set_ylabel(f"{category} empirical slope", labelpad=2); axis.set_title("Participant slopes", loc="left", fontsize=4.7, fontweight="bold", pad=2)
    pearson = pearsonr(points["predictor_value"], points["empirical_slope"]); spearman = spearmanr(points["predictor_value"], points["empirical_slope"])
    family = "BH-24" if category == "Scenery" else "BH-12"
    axis.text(0.025, 0.975, f"r = {pearson.statistic:+.3f} (P = {_format_probability(pearson.pvalue)})\nρ = {spearman.statistic:+.3f} (P = {_format_probability(spearman.pvalue)})\nHLM b = {float(term['estimate']):+.4f}; q = {float(term['q_value']):.3f} ({family})", transform=axis.transAxes, ha="left", va="top", fontsize=3.45, linespacing=1.12, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": HAIRLINE_COLOR, "linewidth": 0.35, "alpha": 0.93}, zorder=5)
    axis.text(0.98, 0.025, "open: E3 lab  ·  filled: E5 online", transform=axis.transAxes, ha="right", va="bottom", fontsize=3.15, color=MUTED_TEXT)
    axis.tick_params(labelsize=3.7)


def _hlm_prediction(axis: plt.Axes, predictions: pd.DataFrame, observed: pd.DataFrame, *, category: str) -> None:
    observed_styles = {"low": {"color": "#AEAEB2", "linestyle": ":", "marker": "o"}, "high": {"color": "#636366", "linestyle": "--", "marker": "s"}}
    for group_id in ["low", "high"]:
        subset = observed.loc[observed.group_id.eq(group_id)].sort_values("trial_within_subcategory"); sty = observed_styles[group_id]
        axis.plot(subset["trial_within_subcategory"], subset["observed_mean"], color=sty["color"], linestyle=sty["linestyle"], marker=sty["marker"], markerfacecolor="white", markeredgecolor=sty["color"], markeredgewidth=0.45, markersize=2.1, linewidth=0.65, alpha=0.82)
    styles = {"Low (-1 SD)": ":", "Mean": "-", "High (+1 SD)": "--", "Low (−1 SD)": ":", "High (+1 SD)": "--"}
    for level in predictions["predictor_level"].drop_duplicates():
        subset = predictions.loc[predictions.predictor_level.eq(level)].sort_values("trial_within_subcategory")
        axis.plot(subset["trial_within_subcategory"], subset["predicted_familiar_preference"], color=CATEGORY_COLORS[category], linestyle=styles.get(level, "-"), linewidth=1.5 if level == "Mean" else 1.1)
    style_axis(axis, grid_axis="y", zero_y=True)
    axis.set_xlim(0.7, 18.3); axis.set_ylim(-0.70, 0.35); axis.set_xticks([1, 6, 12, 18]); axis.set_xlabel("Comparison trial", labelpad=2); axis.set_ylabel("Predicted preference", labelpad=2); axis.set_title("Fixed trajectories", loc="left", fontsize=4.7, fontweight="bold", pad=2)
    axis.text(0.025, 0.975, "model: dotted low  ·  solid mean  ·  dashed high\ngray marker-lines: observed low/high groups", transform=axis.transAxes, ha="left", va="top", fontsize=3.25, color=MUTED_TEXT, linespacing=1.1)
    axis.tick_params(labelsize=3.7)


def render_figure_10(data_root: Path, output: Path) -> Path:
    folder = data_root / "figure_10_hlm_moderators"
    scatter = pd.read_csv(folder / "scatter.csv"); guide = pd.read_csv(folder / "guide.csv"); observed = pd.read_csv(folder / "observed.csv"); prediction = pd.read_csv(folder / "prediction.csv"); term = pd.read_csv(folder / "term.csv")
    figure = _new_page(10)
    specs = [
        ("a", "Flow attitude and Geometry trajectory", "Historical Flow total · supported by original HLM and item-aware CR1 · exploratory", 0.988, 0.695, "Geometry", "Flow attitude (centered)", False),
        ("b", "Snacks/treats familiarity and Scenery trajectory", "Daily item (-3 always new to +3 always familiar) · original HLM only; not CR1-supported", 0.655, 0.370, "Scenery", "Snacks/treats familiarity", True),
        ("c", "Restaurant familiarity and Scenery trajectory", "Daily item (-3 always new to +3 always familiar) · original HLM only; not CR1-supported", 0.330, 0.045, "Scenery", "Restaurant familiarity", True),
    ]
    for panel, title, subtitle, heading_y, axes_y, category, x_label, discrete in specs:
        _hlm_heading(figure, panel, title, subtitle, heading_y)
        subset = lambda frame: frame.loc[frame.panel.eq(panel)]
        _hlm_scatter(figure.add_axes([0.105, axes_y, 0.395, 0.190]), subset(scatter), subset(guide), subset(term).iloc[0], category=category, x_label=x_label, discrete_x=discrete)
        _hlm_prediction(figure.add_axes([0.585, axes_y, 0.385, 0.190]), subset(prediction), subset(observed), category=category)
    return _save(figure, output, "Visualizations of trajectory moderation")


def _horizontal_errorbar(axis: plt.Axes, y: float, estimate: float, low: float, high: float, **kwargs) -> None:
    axis.errorbar(estimate, y, xerr=np.asarray([[estimate - low], [high - estimate]]), capsize=2, elinewidth=0.8, markeredgewidth=0.7, **kwargs)


def render_figure_11(data_root: Path, output: Path) -> Path:
    folder = data_root / "figure_11_reliability"
    fni = pd.read_csv(folder / "panel_a.csv"); slope = pd.read_csv(folder / "panel_b.csv"); reliability = pd.read_csv(folder / "panel_c.csv"); comparability = pd.read_csv(folder / "panel_d.csv")
    palette = {**CATEGORY_COLORS, "Experiment 3": "#0A84FF", "Experiment 5": "#30B0C7"}
    figure = _new_page(11)
    grid = figure.add_gridspec(2, 2, hspace=0.58, wspace=0.48)
    axes = [figure.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    categories = ["Face", "Scenery", "Geometry"]
    for axis, data, title, note in [(axes[0], fni, "Category-specific FNI stability", "Positions 2–18"), (axes[1], slope, "Category-specific trajectory-slope stability", "All 18 positions")]:
        ordered = data.set_index("category").loc[categories].reset_index(); positions = np.arange(3)[::-1]
        for y, (_, row) in zip(positions, ordered.iterrows()):
            _horizontal_errorbar(axis, y, row["spearman_brown_median"], row["spearman_brown_partition_p2_5"], row["spearman_brown_partition_p97_5"], fmt=CATEGORY_MARKERS[row["category"]], color=palette[row["category"]], markerfacecolor="white", markersize=4.0, zorder=3)
        axis.set_yticks(positions, ordered["category"]); axis.set_xlim(-0.75, 1.02); axis.set_xticks([-0.5, 0, 0.5, 1.0]); axis.set_xlabel("Spearman–Brown stability"); axis.set_title(""); style_axis(axis, grid_axis="x", zero_x=True); axis.text(0.02, 0.02, note, transform=axis.transAxes, color=MUTED_TEXT, fontsize=5.2)
    scales = ["AQ", "Daily NF", "Flow", "Maemuki"]; y_base = np.arange(4)[::-1]
    qc_rel = reliability.loc[reliability["sample"].eq("QC-clean")]
    for label, offset, marker in [("Experiment 3", 0.11, "o"), ("Experiment 5", -0.11, "s")]:
        subset = qc_rel.loc[qc_rel.experiment_label.eq(label)].set_index("scale").loc[scales]
        for y, (_, row) in zip(y_base + offset, subset.iterrows()):
            _horizontal_errorbar(axes[2], y, row["cronbach_alpha"], row["ci_low"], row["ci_high"], fmt=marker, color=palette[label], markerfacecolor="white" if label == "Experiment 3" else palette[label], markersize=3.6, zorder=3)
    axes[2].set_yticks(y_base, scales); axes[2].set_xlim(0, 1.02); axes[2].set_xticks([0, 0.5, 1.0]); axes[2].set_xlabel("Cronbach’s alpha (95% bootstrap interval)"); axes[2].set_title("Questionnaire reliability", loc="left", pad=3, fontsize=6.5); style_axis(axes[2], grid_axis="x"); axes[2].text(0.02, 0.02, "Flow: 12 items (E3), 13 items (E5)", transform=axes[2].transAxes, color=MUTED_TEXT, fontsize=5.0)
    subset = comparability.set_index("scale").loc[scales].reset_index(); positions = np.arange(4)[::-1]
    for y, (_, row) in zip(positions, subset.iterrows()):
        _horizontal_errorbar(axes[3], y, row["hedges_g_e5_minus_e3"], row["ci_low"], row["ci_high"], fmt="o", color=MUTED_TEXT, markerfacecolor="white", markersize=3.8, zorder=3)
    axes[3].set_yticks(positions, subset["scale"]); limit = max(1.0, float(np.nanmax(np.abs(subset[["ci_low", "ci_high"]].to_numpy()))) + 0.15); axes[3].set_xlim(-limit, limit); axes[3].set_xlabel("Hedges g (E5 − E3)"); axes[3].set_title("Scale-score comparability", loc="left", pad=3, fontsize=6.5); style_axis(axes[3], grid_axis="x", zero_x=True)
    for letter, axis in zip("abcd", axes): add_panel_label(axis, letter)
    # Match the reader-facing relabel positions in the active manuscript PDF.
    figure.text(60.0 / PAGE_POINTS[11][0], 307.5 / PAGE_POINTS[11][1], "Category-specific FNI stability", ha="left", va="baseline", fontsize=7.0, fontfamily="Helvetica", fontweight="bold", color="#141414")
    figure.text(317.5 / PAGE_POINTS[11][0], 307.5 / PAGE_POINTS[11][1], "Category-specific trajectory-slope stability", ha="left", va="baseline", fontsize=7.0, fontfamily="Helvetica", fontweight="bold", color="#141414")
    figure.legend(handles=[mlines.Line2D([], [], marker="o", color=palette["Experiment 3"], linestyle="", markerfacecolor="white", label="Experiment 3"), mlines.Line2D([], [], marker="s", color=palette["Experiment 5"], linestyle="", label="Experiment 5")], ncol=2, loc="upper right", bbox_to_anchor=(0.98, 0.995))
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.91)
    return _save(figure, output, "Measurement reliability and experiment comparability")


def render_all(data_root: Path, output_dir: Path) -> list[Path]:
    """Render all active supplementary figures in manuscript order."""

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    renderers = [
        render_figure_01, render_figure_02, render_figure_03, render_figure_04,
        render_figure_05, render_figure_06, render_figure_07, render_figure_08,
        render_figure_09, render_figure_10, render_figure_11,
    ]
    outputs = []
    for number, renderer in enumerate(renderers, start=1):
        outputs.append(renderer(data_root, output_dir / f"supplementary_figure_{number:02d}.pdf"))
    return outputs
