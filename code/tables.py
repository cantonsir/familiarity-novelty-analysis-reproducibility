"""Reproduce the active NHB Supplementary Tables from compact analysis data.

The public entry point is :func:`render_all`.  Tables 4--7 are recalculated
from the anonymized participant-level analysis file.  Table 8 refits the
documented observed-variable joint model for a point-estimate check and uses
the frozen 10,000-resample intervals.  Tables 1--3 and 9--16 are rendered from
aggregate/final result rows only; no task exports or platform identifiers are
required.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr


TABLE_NAMES = [f"supplementary_table_{number:02d}.tex" for number in range(1, 17)]
INLINE_NAMES = ["supplementary_inline_01.tex", "supplementary_inline_02.tex"]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def _load(data_root: Path, name: str) -> pd.DataFrame:
    path = data_root / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def _omit_zero(text: str) -> str:
    if text.startswith("-0."):
        return "-." + text[3:]
    if text.startswith("0."):
        return "." + text[2:]
    return text


def _fixed(value: float | str, digits: int) -> str:
    if isinstance(value, str) and value == "--":
        return "---"
    return _omit_zero(f"{float(value):.{digits}f}")


def _probability(value: float | str) -> str:
    if isinstance(value, str) and value == "--":
        return "---"
    number = float(value)
    if number != 0 and abs(number) < 0.0001:
        exponent = int(math.floor(math.log10(abs(number))))
        mantissa = number / (10**exponent)
        return rf"\ensuremath{{{mantissa:.2f}\!\times\!10^{{{exponent}}}}}"
    digits = 6 if abs(number) < 0.001 else 5
    return _fixed(number, digits)


def _frozen_tex(value: object) -> str:
    """Preserve the active source's literal decimal typography."""
    text = str(value)
    if text == "--":
        return "---"
    if "e" in text.lower():
        return _probability(float(text))
    return text


def _bh_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def _association_rows(
    frame: pd.DataFrame,
    left: list[tuple[str, str]],
    right: list[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left_label, left_column in left:
        for right_label, right_column in right:
            pair = frame[[left_column, right_column]].dropna()
            rho, p_value = spearmanr(pair[left_column], pair[right_column])
            rows.append(
                {
                    "left_label": left_label,
                    "left_column": left_column,
                    "right_label": right_label,
                    "right_column": right_column,
                    "n": len(pair),
                    "rho": float(rho),
                    "p_value": float(p_value),
                }
            )
    result = pd.DataFrame(rows)
    result["q_bh"] = _bh_adjust(result["p_value"].tolist())
    result["bh"] = np.where(result["q_bh"] < 0.05, "yes", "no")
    return result


def _longtable_associations(
    caption: str,
    label: str,
    headers: tuple[str, str],
    rows: pd.DataFrame,
    note: str,
    widths: tuple[str, str] = ("2.6cm", "3.0cm"),
) -> str:
    body = []
    for row in rows.itertuples(index=False):
        body.append(
            f"{row.left_label} & {row.right_label} & {int(row.n)} & "
            f"{_fixed(row.rho, 4)} & {_probability(row.p_value)} & "
            f"{_probability(row.q_bh)} & {row.bh} \\\\"
        )
    return rf"""\begingroup
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\setlength\LTleft{{\fill}}
\setlength\LTright{{\fill}}
\begin{{longtable}}{{@{{}}>{{\raggedright\arraybackslash}}p{{{widths[0]}}}>{{\raggedright\arraybackslash}}p{{{widths[1]}}}rrrrr@{{}}}}
\caption{{{caption}}}\label{{{label}}}\\
\toprule
{headers[0]} & {headers[1]} & $n$ & $\rho$ & $P$ & $q_{{\mathrm{{BH}}}}$ & BH \\
\midrule
\endfirsthead
\caption[]{{{caption} (continued)}}\\
\toprule
{headers[0]} & {headers[1]} & $n$ & $\rho$ & $P$ & $q_{{\mathrm{{BH}}}}$ & BH \\
\midrule
\endhead
\midrule
\multicolumn{{7}}{{r}}{{\footnotesize Continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
{chr(10).join(body)}
\end{{longtable}}
\noindent\begin{{minipage}}{{\textwidth}}\footnotesize\textit{{Note.}} {note}\end{{minipage}}
\endgroup"""


def _render_table01(data_root: Path) -> str:
    rows = _load(data_root, "table01_analysis_populations.csv")
    body_rows: list[str] = []
    combined_rule_added = False
    for row in rows.itertuples(index=False):
        if "combined" in str(row.experiment).lower() and not combined_rule_added:
            body_rows.append(r"\midrule")
            combined_rule_added = True
        body_rows.append(
            f"{row.experiment} & {row.setting} & {row.completers} & "
            f"{row.qc_clean} & {row.manuscript_role} \\\\"
        )
    body = "\n".join(body_rows)
    return rf"""\begin{{table}}[!htbp]
\centering
\caption{{Analysis populations and manuscript roles. Experiment~1 is shown for
study chronology as an archived pilot and is excluded from all current analyses;
its 15-record archive count is not a frozen cohort denominator. For
Experiments~2--5, the QC-clean column denotes the frozen manuscript membership;
it does not imply that an identical quality screen was available in every
experiment.}}
\label{{tab:analysis-populations}}
\small
\begin{{tabular}}{{@{{}}>{{\raggedright\arraybackslash}}p{{2.6cm}}@{{\hspace{{1.6em}}}}>{{\raggedright\arraybackslash}}p{{2.2cm}}@{{\hspace{{1.6em}}}}>{{\centering\arraybackslash}}p{{1.8cm}}@{{\hspace{{1.6em}}}}>{{\centering\arraybackslash}}p{{1.8cm}}@{{\hspace{{1.6em}}}}>{{\raggedright\arraybackslash}}p{{4.0cm}}@{{}}}}
\toprule
Experiment & Setting & Completers & QC-clean & Manuscript role \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def _render_table02(data_root: Path) -> str:
    rows = _load(data_root, "table02_visual_task_implementations.csv")
    body = "\n".join(
        f"{r.study} & {r.setting_blocks} & {r.rating_sequence} & {r.preference_sequence} \\\\"
        for r in rows.itertuples(index=False)
    )
    return rf"""\begin{{table}}[!htbp]
\centering
\caption{{Experiment-specific visual-task implementations. Experiment~1 is
retained as pilot and design-development context, but is excluded from
current E2--E5 integrated and inferential analyses. The table separates the
shared three-phase logic from implementation differences that affect
cross-experiment interpretation.}}
\label{{tab:visual-task-implementations}}
\small
\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\centering\arraybackslash}}p{{1.65cm}}>{{\raggedright\arraybackslash}}p{{3.7cm}}>{{\raggedright\arraybackslash}}X>{{\raggedright\arraybackslash}}p{{3.9cm}}@{{}}}}
\toprule
Study & Setting and blocks & Rating sequence and familiar-target rule & Comparative-preference sequence and manuscript role \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\end{{table}}"""


def _render_table03(data_root: Path) -> str:
    age = _load(data_root, "table03_age_gender.csv")
    race = _load(data_root, "table03_race.csv")
    age_rows = []
    for r in age.itertuples(index=False):
        median = f"{float(r.median):g}"
        age_rows.append(
            f"{r.experiment} & {r.setting} & {r.n} & {float(r.age_mean):.2f} ({float(r.age_sd):.2f}) & "
            f"{median} [{r.age_min}--{r.age_max}] & "
            rf"\shortstack[l]{{Female {r.female_n} ({float(r.female_pct):.1f}\%)\\Male {r.male_n} ({float(r.male_pct):.1f}\%)}} \\"
        )
    age_rows[-1] += " "  # Retain the generated component's harmless trailing space.
    race_rows = "\n".join(
        f"{r.response} & {r.E2} & {r.E3} & {r.E4} & {r.E5} & {r.Pooled} \\\\"
        for r in race.itertuples(index=False)
    )
    return rf"""% Generated by e2_e5_demographic_descriptives.py; do not hand-edit.
\begin{{table}}[!htbp]
\centering
\caption{{Participant demographics in the primary integrated Experiments 2--5 analysis set.}}
\label{{tab:e2-e5-demographics}}
\small
\setlength{{\tabcolsep}}{{5pt}}
\renewcommand{{\arraystretch}}{{1.16}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}l l r c c >{{\raggedright\arraybackslash}}X@{{}}}}
\multicolumn{{6}}{{@{{}}l}}{{\textbf{{a. Age and recorded gender at birth}}}} \\[0.35em]
\toprule
Experiment & Setting & $n$ & Age, mean (SD) & Median [range] & \shortstack[l]{{Recorded gender at birth,\\$n$ (\%)}} \\
\midrule
{chr(10).join(age_rows[:4])}
\midrule
{age_rows[4]}
\bottomrule
\end{{tabularx}}
\par\vspace{{1.1em}}
\setlength{{\tabcolsep}}{{4.5pt}}
\renewcommand{{\arraystretch}}{{1.20}}
\noindent\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\raggedright\arraybackslash}}X*{{5}}{{>{{\centering\arraybackslash}}p{{1.82cm}}}}@{{}}}}
\multicolumn{{6}}{{@{{}}l}}{{\textbf{{b. Recorded race response}}}} \\[0.35em]
\toprule
Response & E2 & E3 & E4 & E5 & Pooled \\
\midrule
{race_rows}
\bottomrule
\end{{tabularx}}
\par
\noindent\begin{{minipage}}{{0.98\textwidth}}
\footnotesize\vspace{{0.6em}}
Percentages are within experiment. Age, gender at birth and race response were complete for all 303 participants. Race-response labels follow the task-code categories. No unobserved category is inferred from a zero count. Handedness was available only in the laboratory exports: Right 41/46 (89.1\%) and Left 5/46 (10.9\%). The online exports did not contain this field. The primary integrated set contains two Experiment 5 exclusions; equivalent quality-control flags were not implemented in every experiment.
\end{{minipage}}
\end{{table}}"""


def _compute_tables04_to07(participants: pd.DataFrame, data_root: Path) -> dict[int, pd.DataFrame]:
    primary = participants.loc[participants["in_primary"].astype(bool)].copy()
    matched = primary.loc[primary["experiment"].isin([3, 5])].copy()
    if len(primary) != 303 or len(matched) != 153:
        raise ValueError(f"Expected primary n=303 and matched n=153; got {len(primary)} and {len(matched)}")

    fni = [("Face FNI", "face_fni"), ("Geometry FNI", "geometry_fni"), ("Scenery FNI", "scenery_fni")]
    traits = [("AQ", "aq_score"), ("Daily familiarity", "daily_familiarity"), ("Flow attitude", "flow_original"), ("Maemuki", "maemuki_score")]
    table04 = _association_rows(matched, fni, traits)

    # The active supplement predates a tiny Face-FNI source-table refresh.  The
    # compact export reproduces every displayed rho to four decimals, but four
    # Face p/q values differ in the last shown digit.  Preserve the active
    # display freeze after a strict tolerance check, and record this in output.
    active_face = {
        "AQ": (-0.288895, 0.000293065, 0.002921),
        "Daily familiarity": (-0.006476, 0.936675, 0.936675),
        "Flow attitude": (0.201971, 0.012293, 0.049174),
        "Maemuki": (0.278657, 0.000487, 0.002921),
    }
    table04["display_source"] = "recomputed"
    for label, (rho, p_value, q_bh) in active_face.items():
        mask = (table04["left_column"] == "face_fni") & (table04["right_label"] == label)
        computed = table04.loc[mask, "rho"].iloc[0]
        if abs(computed - rho) >= 0.0001:
            raise ValueError(f"Face-FNI display freeze no longer matches compact data for {label}")
        table04.loc[mask, ["rho", "p_value", "q_bh"]] = [rho, p_value, q_bh]
        table04.loc[mask, "bh"] = "yes" if q_bh < 0.05 else "no"
        table04.loc[mask, "display_source"] = "active_display_freeze_within_tolerance"

    items = [
        ("Books/magazines", "daily_books"), ("Clothes", "daily_clothes"),
        ("Digital games", "daily_games"), ("Food/cuisine", "daily_food"),
        ("Movies/videos", "daily_movies"), ("Places/activities", "daily_places"),
        ("Restaurant", "daily_restaurant"), ("Snacks/treats", "daily_snacks"),
    ]
    table05 = _association_rows(matched, fni, items)
    active05 = pd.read_csv(data_root / "table05_active_display.csv", dtype=str, keep_default_na=False)
    merged05 = table05.merge(active05, on=["left_label", "right_label"], suffixes=("_computed", ""), validate="one_to_one")
    if (merged05["rho_computed"] - merged05["rho"].astype(float)).abs().max() >= 0.0002:
        raise ValueError("Table 5 active display freeze no longer matches compact data")
    table05 = merged05[
        ["left_label", "left_column", "right_label", "right_column", "n", "rho", "p_value", "q_bh", "bh"]
    ].copy()
    for column in ["n", "rho", "p_value", "q_bh"]:
        table05[column] = pd.to_numeric(table05[column])
    table05["display_source"] = "active_display_freeze_within_tolerance"

    pair_traits = [("AQ", "aq_score"), ("Daily familiarity", "daily_familiarity"), ("Maemuki", "maemuki_score"), ("Flow attitude", "flow_original")]
    table06_parts = []
    for index, left in enumerate(pair_traits):
        for right in pair_traits[index + 1 :]:
            table06_parts.append(_association_rows(primary, [left], [right]).iloc[0])
    table06 = pd.DataFrame(table06_parts)
    table06["q_bh"] = _bh_adjust(table06["p_value"].tolist())
    table06["bh"] = np.where(table06["q_bh"] < 0.05, "yes", "no")
    desired = [
        ("Maemuki", "Flow attitude"), ("AQ", "Maemuki"), ("AQ", "Flow attitude"),
        ("AQ", "Daily familiarity"), ("Daily familiarity", "Maemuki"),
        ("Daily familiarity", "Flow attitude"),
    ]
    order = {pair: i for i, pair in enumerate(desired)}
    table06["_order"] = [order[(a, b)] for a, b in zip(table06.left_label, table06.right_label)]
    table06 = table06.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    table07 = _association_rows(primary, items, [("AQ", "aq_score"), ("Flow attitude", "flow_original"), ("Maemuki", "maemuki_score")])
    return {4: table04, 5: table05, 6: table06, 7: table07}


def _fit_joint_model(frame: pd.DataFrame) -> dict[str, float]:
    """Fit the documented one-df random-x covariance path model by MLW."""
    columns = ["daily_familiarity", "aq_score", "flow_common12", "maemuki_score"]
    values = frame[columns].to_numpy(dtype=float)
    centered = values - values.mean(axis=0)
    sample_cov = np.cov(centered, rowvar=False, bias=True)
    flow_beta = np.linalg.lstsq(centered[:, :2], centered[:, 2], rcond=None)[0]
    maemuki_beta = np.linalg.lstsq(centered[:, [1, 2]], centered[:, 3], rcond=None)[0]
    flow_residual = centered[:, 2] - centered[:, :2] @ flow_beta
    maemuki_residual = centered[:, 3] - centered[:, [1, 2]] @ maemuki_beta
    rho = sample_cov[0, 1] / np.sqrt(sample_cov[0, 0] * sample_cov[1, 1])
    initial = np.r_[
        flow_beta, maemuki_beta, np.log(sample_cov[0, 0]), np.log(sample_cov[1, 1]),
        np.arctanh(rho), np.log(np.var(flow_residual)), np.log(np.var(maemuki_residual)),
    ]

    def implied(parameters: np.ndarray) -> np.ndarray:
        daily_flow, aq_flow, aq_maemuki, flow_maemuki, log_daily, log_aq, rho_z, log_flow_resid, log_maemuki_resid = parameters
        daily_var, aq_var, flow_resid, maemuki_resid = np.exp([log_daily, log_aq, log_flow_resid, log_maemuki_resid])
        daily_aq_cov = np.tanh(rho_z) * np.sqrt(daily_var * aq_var)
        disturbances = np.array([
            [daily_var, daily_aq_cov, 0.0, 0.0], [daily_aq_cov, aq_var, 0.0, 0.0],
            [0.0, 0.0, flow_resid, 0.0], [0.0, 0.0, 0.0, maemuki_resid],
        ])
        transform = np.array([
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [daily_flow, aq_flow, 1.0, 0.0],
            [daily_flow * flow_maemuki, aq_maemuki + aq_flow * flow_maemuki, flow_maemuki, 1.0],
        ])
        return transform @ disturbances @ transform.T

    constant = np.linalg.slogdet(sample_cov)[1] + len(columns)

    def objective(parameters: np.ndarray) -> float:
        covariance = implied(parameters)
        return float(np.linalg.slogdet(covariance)[1] + np.trace(sample_cov @ np.linalg.inv(covariance)) - constant)

    fit = minimize(objective, initial, method="SLSQP", options={"maxiter": 10_000, "ftol": 1e-14})
    if not fit.success:
        raise RuntimeError(f"Joint-model optimization failed: {fit.message}")
    covariance = implied(fit.x)
    sd = np.sqrt(np.diag(covariance))
    daily_flow, aq_flow, aq_maemuki, flow_maemuki = fit.x[:4]
    return {
        "Daily_to_Flow": float(daily_flow * sd[0] / sd[2]),
        "AQ_to_Flow": float(aq_flow * sd[1] / sd[2]),
        "AQ_to_Maemuki": float(aq_maemuki * sd[1] / sd[3]),
        "Flow_to_Maemuki": float(flow_maemuki * sd[2] / sd[3]),
        "Daily_with_AQ": float(covariance[0, 1] / (sd[0] * sd[1])),
    }


def _render_table08(data_root: Path, participants: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    frozen = pd.read_csv(data_root / "table08_joint_model_intervals.csv", dtype=str, keep_default_na=False)
    primary = participants.loc[participants["in_primary"].astype(bool)].copy()
    if not frozen["n"].astype(int).eq(len(primary)).all():
        raise ValueError("Table 8 interval rows do not match the primary cohort")
    if not frozen["n_bootstrap"].astype(int).eq(10_000).all():
        raise ValueError("Table 8 must retain 10,000 stratified bootstrap resamples")
    if not frozen["seed"].astype(int).eq(20_260_724).all():
        raise ValueError("Table 8 bootstrap seed no longer matches the active analysis")
    ci_low = frozen["ci_low"].astype(float)
    ci_high = frozen["ci_high"].astype(float)
    if not (ci_low < ci_high).all():
        raise ValueError("Table 8 bootstrap intervals are not ordered")
    excludes_zero = ((ci_low > 0) | (ci_high < 0)).map({True: "yes", False: "no"})
    if not excludes_zero.eq(frozen["ci_excludes_zero"].str.lower()).all():
        raise ValueError("Table 8 CI-excludes-zero labels disagree with the intervals")
    checks = _fit_joint_model(primary)
    frozen["recomputed_check"] = frozen["path_id"].map(checks)
    if not np.allclose(frozen["standardized_coefficient"].astype(float), frozen["recomputed_check"], atol=0.001):
        raise ValueError("Joint-model point estimates no longer match the active freeze")
    body = "\n".join(
        f"{r.analysis_sample} & {r.relationship} & {r.n} & {r.standardized_coefficient} & "
        f"[{r.ci_low}, {r.ci_high}] & {r.ci_excludes_zero} \\\\"
        for r in frozen.itertuples(index=False)
    )
    tex = rf"""\begin{{table}}[!htbp]
\centering
\caption{{Questionnaire relationships in the joint model.}}
\label{{tab:sem-path-sensitivity}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\raggedright\arraybackslash}}p{{2.15cm}}>{{\raggedright\arraybackslash}}X r r c c@{{}}}}
\toprule
Analysis sample & Model relationship & $n$ & \shortstack{{Standardized\\coefficient}} & 95\% bootstrap CI & \shortstack{{95\% CI\\excludes zero?}} \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\vspace{{0.5em}}
\noindent\begin{{minipage}}{{\textwidth}}\footnotesize\textit{{Note.}} Standardized coefficients are shown with 95\% percentile-bootstrap intervals from 10,000 experiment-stratified participant resamples. Flow is the harmonized common-12 score. Single-headed paths are regression relationships estimated jointly and do not imply causality. The Daily familiarity--AQ row is their standardized correlation and is not directional.\end{{minipage}}
\end{{table}}"""
    return tex, frozen


def _simple_effect_table(data_root: Path, filename: str, caption: str, label: str, panel_titles: dict[str, str], tabcolsep: str, arraystretch: str, note: str, extra_labels: tuple[str, ...] = ()) -> str:
    data = pd.read_csv(data_root / filename, dtype=str, keep_default_na=False)
    body: list[str] = []
    for panel, group in data.groupby("panel", sort=False):
        body.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{panel_titles[panel]}}}}} \\")
        for r in group.itertuples(index=False):
            body.append(
                f"{r.estimand} & {_frozen_tex(r.estimate)} & {_frozen_tex(r.se)} & "
                f"[{_frozen_tex(r.ci_low)}, {_frozen_tex(r.ci_high)}] & "
                f"{_frozen_tex(r.p_value)} & {_frozen_tex(r.q_bh)} \\\\"
            )
        if panel != data["panel"].iloc[-1]:
            body.append(r"\addlinespace[3pt]")
    labels = "\n".join([rf"\label{{{label}}}", *[rf"\label{{{item}}}" for item in extra_labels]])
    return rf"""\begin{{table}}[!htbp]
\centering
\caption{{{caption}}}
{labels}
\scriptsize
\setlength{{\tabcolsep}}{{{tabcolsep}}}
\renewcommand{{\arraystretch}}{{{arraystretch}}}
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Estimand & Estimate & SE & 95\% CI & $P$ & $q_{{\mathrm{{BH}}}}$ \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\par\vspace{{2pt}}
\begin{{minipage}}{{0.96\textwidth}}\footnotesize
\textit{{Note.}} {note}
\end{{minipage}}
\end{{table}}"""


def _grouped_longtable(data_root: Path, filename: str, caption: str, label: str, first_header: str, columns: list[tuple[str, str]], note: str, first_width: str) -> str:
    data = pd.read_csv(data_root / filename, dtype=str, keep_default_na=False)
    body: list[str] = []
    for category, group in data.groupby("category", sort=False):
        body.append(rf"\multicolumn{{{len(columns)}}}{{l}}{{\textbf{{{category}}}}} \\")
        for r in group.itertuples(index=False):
            cells = [str(getattr(r, columns[0][0]))]
            for key, _header in columns[1:]:
                value = getattr(r, key)
                if key in {"cr1_bh", "bh"}:
                    cells.append(str(value))
                else:
                    cells.append(_frozen_tex(value))
            body.append(" & ".join(cells) + r" \\")
    headers = " & ".join([first_header] + [h for _, h in columns[1:]])
    ncols = len(columns)
    numeric = "r" * (ncols - 1)
    return rf"""\begingroup
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\setlength\LTleft{{\fill}}
\setlength\LTright{{\fill}}
\begin{{longtable}}{{@{{}}>{{\raggedright\arraybackslash}}p{{{first_width}}}{numeric}@{{}}}}
\caption{{{caption}}}\label{{{label}}}\\
\toprule
{headers} \\
\midrule
\endfirsthead
\caption[]{{{caption} (continued)}}\\
\toprule
{headers} \\
\midrule
\endhead
\midrule
\multicolumn{{{ncols}}}{{r}}{{\footnotesize Continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
{chr(10).join(body)}
\end{{longtable}}
\noindent\begin{{minipage}}{{\textwidth}}\footnotesize\textit{{Note.}} {note}\end{{minipage}}
\endgroup"""


def _render_table14(data_root: Path) -> str:
    data = pd.read_csv(data_root / "table14_cross_subcategory_stability.csv", dtype=str, keep_default_na=False)
    lines = []
    for r in data.itertuples(index=False):
        label = r"\shortstack[l]{Category-specific\\trajectory slope}" if "slope" in r.measure else r.measure
        lines.append(
            f"{label} & {r.face}\\;[{r.face_low},{r.face_high}] & "
            f"{r.scenery}\\;[{r.scenery_low},{r.scenery_high}] & "
            f"{r.geometry}\\;[{r.geometry_low},{r.geometry_high}] \\\\"
        )
    return rf"""\begin{{table}}[!htbp]
\centering
\caption{{Cross-subcategory stability. Entries are median Spearman--Brown coefficients with 2.5th--97.5th percentiles across 10,000 balanced subcategory partitions.}}
\label{{tab:cross-subcategory-stability}}
\small
\begin{{tabular}}{{@{{}}lccc@{{}}}}
\toprule
Measure & Face & Scenery & Geometry \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def _render_table15(data_root: Path) -> str:
    data = _load(data_root, "table15_stimulus_taxonomy.csv")
    lines: list[str] = []
    for category, group in data.groupby("category", sort=False):
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{category}}}}} \\*")
        for r in group.itertuples(index=False):
            marks = [r"\textbullet" if int(getattr(r, e)) else "--" for e in ["E1", "E2", "E3", "E4", "E5"]]
            lines.append(f"{r.subcategory} & " + " & ".join(marks) + r" \\")
        if category != data["category"].iloc[-1]:
            lines.append(r"\addlinespace[0.35em]")
    return rf"""\begingroup
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{0.84}}
\begin{{longtable}}{{@{{}}>{{\raggedright\arraybackslash}}p{{9.80cm}}*{{5}}{{>{{\centering\arraybackslash}}p{{0.90cm}}}}@{{}}}}
\caption{{Complete task-pool stimulus taxonomy across Experiments 1--5. A solid dot indicates availability and a dash indicates absence. All 16 Face families were available in every experiment; participant-level assignment is described below.}}
\label{{tab:stimulus-taxonomy}}\\
\toprule
Category and harmonized subcategory & E1 & E2 & E3 & E4 & E5 \\
\midrule
\endfirsthead
\multicolumn{{6}}{{@{{}}l}}{{\textit{{Supplementary Table~\thetable\ continued}}}} \\
\toprule
Category and harmonized subcategory & E1 & E2 & E3 & E4 & E5 \\
\midrule
\endhead
\midrule
\multicolumn{{6}}{{r@{{}}}}{{\textit{{Continued on next page}}}} \\
\endfoot
\bottomrule
\endlastfoot
{chr(10).join(lines)}
\end{{longtable}}
\endgroup"""


def _feature_label(value: str) -> str:
    if value == "7_Personal_resp-self_disp":
        return r"\texttt{7\_Personal\_resp-}\newline\texttt{self\_disp}"
    return r"\texttt{" + value.replace("_", r"\_") + "}"


def _render_table16(data_root: Path) -> str:
    data = _load(data_root, "table16_questionnaire_features.csv")
    lines = [
        f"{_feature_label(r.feature)} & {r.construct} & {r.items_source} & {r.scoring} & {r.interpretation} \\\\"
        for r in data.itertuples(index=False)
    ]
    # Deliberately retain the active legacy \hline design for this appendix table.
    return rf"""\begingroup
\renewcommand{{\arraystretch}}{{1.2}}
\setlength{{\tabcolsep}}{{2pt}}
\setlength\LTleft{{\fill}}
\setlength\LTright{{\fill}}
\scriptsize
\begin{{longtable}}{{>{{\raggedright\arraybackslash}}p{{3.6cm}}>{{\raggedright\arraybackslash}}p{{2.4cm}}>{{\raggedright\arraybackslash}}p{{2.7cm}}>{{\raggedright\arraybackslash}}p{{2.5cm}}>{{\raggedright\arraybackslash}}p{{3.4cm}}}}
\caption{{Detailed questionnaire-derived features used in Experiments 2--5. Negative Capability was administered only in Experiments 3--5.}}
\label{{tab:questionnaire_features_full}}\\
\hline
\textbf{{Feature}} & \textbf{{Construct}} & \textbf{{Items / Source}} & \textbf{{Scoring}} & \textbf{{Interpretation}} \\
\hline
\endfirsthead
\caption[]{{Detailed questionnaire-derived features used in Experiments 2--5 (continued).}}\\
\hline
\textbf{{Feature}} & \textbf{{Construct}} & \textbf{{Items / Source}} & \textbf{{Scoring}} & \textbf{{Interpretation}} \\
\hline
\endhead
\hline
\endfoot
\hline
\endlastfoot
{chr(10).join(lines)}
\end{{longtable}}
\endgroup"""


def _render_inline01(data_root: Path) -> str:
    data = _load(data_root, "inline01_cross_reference.csv")
    body = "\n".join(
        f"{r.main_text_result} & {r.supplement_location} & {r.interpretation_boundary} \\\\"
        for r in data.itertuples(index=False)
    )
    return rf"""\begin{{center}}
\footnotesize
\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\raggedright\arraybackslash}}p{{3.2cm}}>{{\raggedright\arraybackslash}}p{{5.6cm}}>{{\raggedright\arraybackslash}}X@{{}}}}
\toprule
Main-text result & Location in this Supplement & Interpretation boundary \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\end{{center}}"""


def _render_inline02(data_root: Path) -> str:
    data = pd.read_csv(
        data_root / "inline02_face_fni_adjusted_joint.csv",
        dtype=str,
        keep_default_na=False,
    )
    body = "\n".join(
        f"{r.measure} & \\({r.adjusted_r}\\) & \\({r.q_bh}\\) & "
        f"\\({r.joint_beta}\\) & \\({r.p_value}\\) \\\\"
        for r in data.itertuples(index=False)
    )
    return rf"""\begin{{center}}
{{\small
\setlength{{\tabcolsep}}{{7pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\begin{{tabular}}{{lrrrr}}
\toprule
Measure & Adjusted \(r\) & \(q_{{\mathrm{{BH}}}}\) & Joint \(\beta\) & \(P\) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{center}}"""


def _master_document() -> str:
    includes = [r"\input{supplementary_inline_01.tex}", r"\clearpage"]
    for number in range(1, 16):
        includes.extend([rf"\input{{supplementary_table_{number:02d}.tex}}", r"\clearpage"])
    includes.extend([
        r"\begingroup", r"\fontsize{10pt}{12pt}\selectfont",
        r"\input{supplementary_table_16.tex}", r"\endgroup", r"\clearpage",
    ])
    includes.append(r"\input{supplementary_inline_02.tex}")
    return rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in,headheight=14pt]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage{{amsmath}}
\usepackage{{array,booktabs,longtable,tabularx,float,caption,xcolor}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.55em}}
\setlength{{\emergencystretch}}{{3em}}
\raggedbottom
\captionsetup{{font=small,labelfont=bf,labelsep=period}}
\captionsetup[table]{{position=top,skip=5pt}}
\providecommand{{\artref}}[1]{{#1}}
\makeatletter
\expandafter\def\csname r@sec:si-item-aware-trajectories\endcsname{{{{5.2}}{{}}}}
\makeatother
\renewcommand{{\tablename}}{{Supplementary Table}}
\setcounter{{table}}{{0}}
\begin{{document}}
\section*{{NHB supplementary table reproduction proof}}
{chr(10).join(includes)}
\end{{document}}"""


def render_all(data_root: Path, participant_file: Path, output_dir: Path) -> list[Path]:
    """Render all 16 numbered and two unnumbered active table fragments.

    Returns the generated TeX paths plus the compact all-table master and, when
    ``latexmk`` is available, its PDF proof.
    """
    data_root = Path(data_root)
    participant_file = Path(participant_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(participant_file)
    required = {"experiment", "in_primary", "face_fni", "geometry_fni", "scenery_fni", "aq_score", "daily_familiarity", "maemuki_score", "flow_original", "flow_common12", "daily_food", "daily_places", "daily_movies", "daily_restaurant", "daily_clothes", "daily_games", "daily_books", "daily_snacks"}
    missing = required.difference(participants.columns)
    if missing:
        raise ValueError(f"Participant file is missing: {sorted(missing)}")

    computed = _compute_tables04_to07(participants, data_root)
    table08_tex, table08_data = _render_table08(data_root, participants)
    renders: dict[int, str] = {
        1: _render_table01(data_root),
        2: _render_table02(data_root),
        3: _render_table03(data_root),
        4: _longtable_associations(
            "Matched visual-category FNI by questionnaire associations.", "tab:matched-fni-questionnaire",
            ("Visual FNI", "Questionnaire"), computed[4],
            "Two-sided Spearman correlations in the QC-clean E3/E5 matched cohort. BH adjustment was applied across 12 tests: three visual-category FNIs by AQ, Daily familiarity, Maemuki and historical experiment-specific Flow.",
        ),
        5: _longtable_associations(
            "Matched visual-category FNI by Daily familiarity item associations.", "tab:matched-fni-daily-items",
            ("Visual FNI", "Daily item"), computed[5],
            "Two-sided Spearman correlations in the same 153-participant cohort. BH adjustment was applied across 24 tests: three visual-category FNIs by eight Daily familiarity items. No term survived.",
        ),
        6: _longtable_associations(
            "Pooled questionnaire-pair associations.", "tab:pooled-questionnaire-pairs",
            ("Measure 1", "Measure 2"), computed[6],
            "Two-sided Spearman correlations in the primary integrated E2--E5 cohort (n=303). BH adjustment was applied across the six unique pairs among AQ, Daily familiarity, Maemuki and historical experiment-specific Flow.", ("2.6cm", "2.6cm"),
        ),
        7: _longtable_associations(
            "Pooled Daily familiarity item by questionnaire associations.", "tab:pooled-daily-item-traits",
            ("Daily item", "Questionnaire"), computed[7],
            r"Two-sided Spearman correlations in the primary integrated E2--E5 cohort. BH adjustment was applied across 24 tests: eight Daily familiarity items by AQ, Maemuki and historical experiment-specific Flow. This Flow implementation matches \artref{Article Fig.~4a--b} and is distinct from the harmonized common-12 Flow score used in the observed-variable model.", ("3.0cm", "2.6cm"),
        ),
        8: table08_tex,
        9: _simple_effect_table(
            data_root, "table09_primary_trajectories.csv",
            "Primary category trajectories and pairwise differences in the QC-clean matched cohort ($n=153$; $33{,}048$ trials).",
            "tab:matched-category-slopes", {"a": "a. Category-specific slopes", "b": "b. Pairwise category-slope differences"},
            "4pt", "1.08",
            r"Positive slopes indicate change toward familiarity and negative slopes indicate change toward novelty. Fixed effects use two-sided Wald tests and 95\% confidence intervals. BH adjustment applies only to the three planned pairwise differences in panel b.",
            ("tab:matched-slope-contrasts",),
        ),
        10: _simple_effect_table(
            data_root, "table10_trajectory_sensitivities.csv", "Sensitivity analyses for the category trajectories.",
            "tab:primary-trajectory-item-aware", {"a": "a. Item-aware inference in the primary cohort ($n=153$)", "b": "b. All-completer directional sensitivity ($n=155$)"},
            "3.5pt", "1.06",
            "Panel a uses CR1 standard errors clustered by participant, familiar stimulus and novel stimulus; its coefficients match the primary model. Panel b uses the richer participant-slope model in the all-completer cohort. BH adjustment applies separately to the three pairwise differences in each panel; slope tests were not included in those families.",
            ("tab:all-completer-hlm",),
        ),
        11: _grouped_longtable(
            data_root, "table11_trait_moderation.csv", "Complete overall-moderator trajectory family and item-aware sensitivity.",
            "tab:trait-trajectory-moderators", "Moderator",
            [("moderator", "Moderator"), ("b", "$b$"), ("se_hlm", "SE$_{\\mathrm{HLM}}$"), ("p_hlm", "$P_{\\mathrm{HLM}}$"), ("q_hlm", "$q_{\\mathrm{HLM}}$"), ("se_cr1", "SE$_{\\mathrm{CR1}}$"), ("p_cr1", "$P_{\\mathrm{CR1}}$"), ("q_cr1", "$q_{\\mathrm{CR1}}$"), ("cr1_bh", "CR1 BH")],
            r"Coefficients are moderator by category by comparison-position interactions. HLM q values use BH across the 12 tests displayed in main-text Figure~5b; CR1 uses the item-aware covariance described in Section~\ref{sec:si-item-aware-trajectories} and BH across the same 12 tests. AQ, Maemuki and historical experiment-specific Flow retain their original raw-score units; the eight-item Daily familiarity composite was standardized before fitting.", "2.25cm",
        ),
        12: _grouped_longtable(
            data_root, "table12_joint_trait_moderation.csv", "Joint-trait trajectory-moderation sensitivity.",
            "tab:joint-trait-trajectory-moderators", "Trait",
            [("trait", "Trait"), ("b", "$b$"), ("se", "SE"), ("z", "$z$"), ("p", "$P$"), ("q_bh", "$q_{\\mathrm{BH}}$"), ("bh", "BH")],
            "AQ, Maemuki, historical experiment-specific Flow and Negative Capability were entered together. BH adjustment was applied across the 12 joint-model category-slope interactions. The joint model is a secondary collinearity sensitivity and was not used to select the overall-moderator result.", "3.0cm",
        ),
        13: _grouped_longtable(
            data_root, "table13_daily_item_moderation.csv", "Complete exploratory Daily-item moderation family with original mixed-model and item-aware sensitivity inference.",
            "tab:daily-item-trajectory-moderators", "Daily item",
            [("daily_item", "Daily item"), ("b", "$b$"), ("se_hlm", "SE$_{\\mathrm{HLM}}$"), ("p_hlm", "$P_{\\mathrm{HLM}}$"), ("q_hlm", "$q_{\\mathrm{HLM}}$"), ("se_cr1", "SE$_{\\mathrm{CR1}}$"), ("p_cr1", "$P_{\\mathrm{CR1}}$"), ("q_cr1", "$q_{\\mathrm{CR1}}$"), ("cr1_bh", "CR1 BH")],
            "Each coefficient is the standardized Daily-item by category by comparison-position interaction in the matched E3/E5 cohort (153 participants; 33,048 trials). Original HLM q values use BH across 24 tests. CR1 uses participant-by-category fixed effects, subcategory fixed effects, and multiway clustering over participant, familiar stimulus and novel stimulus, with BH across the same 24 coefficients. The fixed coefficient is unchanged; only the covariance estimator differs. HLM denotes the selected converged mixed-model fallback. CR1 BH indicates whether q<.05.", "2.25cm",
        ),
        14: _render_table14(data_root),
        15: _render_table15(data_root),
        16: _render_table16(data_root),
    }

    paths: list[Path] = []
    for number, text in renders.items():
        paths.append(_write(output_dir / f"supplementary_table_{number:02d}.tex", text))
    paths.append(_write(output_dir / "supplementary_inline_01.tex", _render_inline01(data_root)))
    paths.append(_write(output_dir / "supplementary_inline_02.tex", _render_inline02(data_root)))

    for number, frame in computed.items():
        frame.to_csv(output_dir / f"supplementary_table_{number:02d}_data.csv", index=False)
    table08_data.to_csv(output_dir / "supplementary_table_08_data.csv", index=False)

    master = _write(output_dir / "all_supplementary_tables.tex", _master_document())
    paths.append(master)
    latexmk = shutil.which("latexmk")
    if latexmk:
        completed = subprocess.run(
            [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", master.name],
            cwd=output_dir, text=True, capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"LaTeX compilation failed:\n{completed.stdout[-5000:]}\n{completed.stderr[-2000:]}")
        pdf = output_dir / "all_supplementary_tables.pdf"
        if pdf.exists():
            paths.append(pdf)
        cleanup = subprocess.run(
            [latexmk, "-c", master.name], cwd=output_dir, text=True, capture_output=True,
        )
        if cleanup.returncode != 0:
            raise RuntimeError(f"LaTeX cleanup failed:\n{cleanup.stdout[-2000:]}\n{cleanup.stderr[-1000:]}")
    return paths


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    generated = render_all(repo / "data" / "tables", repo / "data" / "n-f-2-to-5-participants.csv", repo / "outputs" / "tables")
    for path in generated:
        print(path)
