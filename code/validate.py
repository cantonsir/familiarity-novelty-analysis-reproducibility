"""Lightweight checks for the reviewer-facing figure and table package."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import pandas as pd
from pypdf import PdfReader


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _page_size(path: Path) -> tuple[int, float, float]:
    reader = PdfReader(path)
    if not reader.pages:
        raise ValueError(f"PDF has no pages: {path}")
    page = reader.pages[0]
    return len(reader.pages), float(page.mediabox.width), float(page.mediabox.height)


def require_arial() -> Path:
    """Fail before rendering if the manuscript's required Arial font is absent."""

    try:
        resolved = Path(
            font_manager.findfont(
                FontProperties(family="Arial"), fallback_to_default=False
            )
        )
    except ValueError as exc:
        raise RuntimeError(
            "Arial is required for manuscript-identical figure typography. "
            "Install Arial before running this repository."
        ) from exc
    if not resolved.is_file() or "arial" not in resolved.name.lower():
        raise RuntimeError(f"Arial preflight resolved an unexpected font: {resolved}")
    return resolved


def _count_raster_images(resources: object, visited: set[int] | None = None) -> int:
    """Count image XObjects, including those nested in Form XObjects."""

    if resources is None:
        return 0
    visited = visited if visited is not None else set()
    resources = resources.get_object() if hasattr(resources, "get_object") else resources
    xobjects = resources.get("/XObject") if hasattr(resources, "get") else None
    if xobjects is None:
        return 0
    xobjects = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
    count = 0
    for reference in xobjects.values():
        marker = id(reference)
        if marker in visited:
            continue
        visited.add(marker)
        item = reference.get_object() if hasattr(reference, "get_object") else reference
        subtype = str(item.get("/Subtype", ""))
        if subtype == "/Image":
            count += 1
        elif subtype == "/Form":
            count += _count_raster_images(item.get("/Resources"), visited)
    return count


def validate_figures(
    repository_root: str | Path,
    *,
    tolerance_pt: float = 0.06,
) -> pd.DataFrame:
    """Check that every active figure exists on the declared artboard.

    Binary hashes are reported for provenance but are not required to match.
    The required-text and vector-only checks prevent a blank, unrelated, or
    rasterized page from passing solely because its artboard is correct.
    """

    root = Path(repository_root).resolve()
    manifest = pd.read_csv(root / "data" / "figure_manifest.csv")
    rows: list[dict[str, object]] = []
    for record in manifest.to_dict("records"):
        output = root / str(record["output_file"])
        row = dict(record)
        row["exists"] = output.is_file()
        if output.is_file():
            pages, width, height = _page_size(output)
            row.update(
                {
                    "pages": pages,
                    "observed_width_pt": width,
                    "observed_height_pt": height,
                    "width_error_pt": abs(width - float(record["page_width_pt"])),
                    "height_error_pt": abs(height - float(record["page_height_pt"])),
                    "generated_pdf_sha256": sha256(output),
                }
            )
            page = PdfReader(output).pages[0]
            extracted_text = " ".join((page.extract_text() or "").split())
            required_text = str(record["required_text"])
            row["required_text_found"] = required_text in extracted_text
            row["extracted_text_characters"] = len(extracted_text)
            row["raster_image_xobjects"] = _count_raster_images(
                page.get("/Resources")
            )
            row["vector_only"] = row["raster_image_xobjects"] == 0
            row["artboard_ok"] = (
                pages == 1
                and row["width_error_pt"] <= tolerance_pt
                and row["height_error_pt"] <= tolerance_pt
            )
            row["binary_reference_match"] = (
                row["generated_pdf_sha256"] == record["reference_pdf_sha256"]
            )
            row["content_ok"] = (
                row["required_text_found"]
                and row["extracted_text_characters"] >= 40
                and row["vector_only"]
            )
        else:
            row["artboard_ok"] = False
            row["binary_reference_match"] = False
            row["required_text_found"] = False
            row["vector_only"] = False
            row["content_ok"] = False
        rows.append(row)
    report = pd.DataFrame(rows)
    if not report["artboard_ok"].all():
        bad = report.loc[~report["artboard_ok"], ["figure", "output_file"]]
        raise AssertionError(f"Missing or incorrectly sized figure PDFs:\n{bad.to_string(index=False)}")
    if not report["content_ok"].all():
        bad = report.loc[
            ~report["content_ok"],
            ["figure", "output_file", "required_text_found", "vector_only"],
        ]
        raise AssertionError(
            "Figure content validation failed:\n" + bad.to_string(index=False)
        )
    return report


def validate_analysis_crosschecks(repository_root: str | Path) -> pd.DataFrame:
    """Bind Main Figure 3/4 frozen display rows to recomputed Tables 4--7."""

    root = Path(repository_root).resolve()
    specifications = [
        (
            "Main Figure 3 BH-12 / Supplementary Table 4",
            root / "data/main/figure03/category_questionnaire_bh12.csv",
            root / "outputs/tables/supplementary_table_04_data.csv",
            ("behavior_label", "questionnaire_label"),
            ("left_label", "right_label"),
        ),
        (
            "Main Figure 3 BH-24 / Supplementary Table 5",
            root / "data/main/figure03/category_daily_bh24.csv",
            root / "outputs/tables/supplementary_table_05_data.csv",
            ("behavior_label", "daily_item_label"),
            ("left_label", "right_label"),
        ),
        (
            "Main Figure 4 BH-6 / Supplementary Table 6",
            root / "data/main/figure04/questionnaire_pairwise_bh6.csv",
            root / "outputs/tables/supplementary_table_06_data.csv",
            ("x_label", "y_label"),
            ("left_label", "right_label"),
        ),
        (
            "Main Figure 4 BH-24 / Supplementary Table 7",
            root / "data/main/figure04/daily_item_trait_bh24.csv",
            root / "outputs/tables/supplementary_table_07_data.csv",
            ("item_label", "trait_label"),
            ("left_label", "right_label"),
        ),
    ]
    rows: list[dict[str, object]] = []
    tolerance = 5e-4
    for family, figure_path, table_path, figure_keys, table_keys in specifications:
        figure = pd.read_csv(figure_path)
        table = pd.read_csv(table_path)
        symmetric = "BH-6" in family

        def key(frame_row: pd.Series, columns: tuple[str, str]) -> tuple[str, str]:
            values = (str(frame_row[columns[0]]), str(frame_row[columns[1]]))
            return tuple(sorted(values)) if symmetric else values

        table_lookup = {key(row, table_keys): row for _, row in table.iterrows()}
        rho_errors: list[float] = []
        q_errors: list[float] = []
        n_matches: list[bool] = []
        missing: list[str] = []
        for _, figure_row in figure.iterrows():
            row_key = key(figure_row, figure_keys)
            table_row = table_lookup.get(row_key)
            if table_row is None:
                missing.append(" / ".join(row_key))
                continue
            rho_errors.append(
                abs(float(figure_row["spearman_rho"]) - float(table_row["rho"]))
            )
            q_errors.append(
                abs(float(figure_row["q_value"]) - float(table_row["q_bh"]))
            )
            n_matches.append(int(figure_row["n"]) == int(table_row["n"]))
        max_rho = max(rho_errors, default=float("inf"))
        max_q = max(q_errors, default=float("inf"))
        passed = (
            not missing
            and len(rho_errors) == len(figure)
            and all(n_matches)
            and max_rho <= tolerance
            and max_q <= tolerance
        )
        rows.append(
            {
                "analysis_family": family,
                "matched_rows": len(rho_errors),
                "expected_rows": len(figure),
                "missing_keys": "; ".join(missing),
                "all_n_match": all(n_matches) if n_matches else False,
                "max_rho_absolute_error": max_rho,
                "max_q_absolute_error": max_q,
                "tolerance": tolerance,
                "within_tolerance": passed,
            }
        )
    report = pd.DataFrame(rows)
    if not report["within_tolerance"].all():
        raise AssertionError(
            "Frozen figure rows disagree with recomputed table analyses:\n"
            + report.to_string(index=False)
        )
    return report


def validate_tables(repository_root: str | Path) -> pd.DataFrame:
    """Check that all 16 numbered and two inline table fragments were made."""

    root = Path(repository_root).resolve()
    expected = [f"supplementary_table_{number:02d}.tex" for number in range(1, 17)]
    expected += ["supplementary_inline_01.tex", "supplementary_inline_02.tex"]
    rows = []
    for name in expected:
        path = root / "outputs" / "tables" / name
        rows.append(
            {
                "output_file": str(path.relative_to(root)),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": sha256(path) if path.is_file() else "",
            }
        )
    report = pd.DataFrame(rows)
    if not report["exists"].all() or (report["bytes"] == 0).any():
        bad = report.loc[(~report["exists"]) | (report["bytes"] == 0), "output_file"]
        raise AssertionError("Missing or empty table fragments: " + ", ".join(bad))
    return report


def validate_public_data(repository_root: str | Path) -> pd.DataFrame:
    """Reject common raw platform identifiers from public CSV inputs."""

    root = Path(repository_root).resolve()
    forbidden_columns = re.compile(
        r"(^|_)(filename|file_name|submission|timestamp|participant_id|participant_display_id|participant_key|"
        r"prolific|mturk|sona|worker_id|assignment_id)(_|$)",
        flags=re.IGNORECASE,
    )
    forbidden_values = re.compile(
        r"(?:\.json\b|file_[0-9a-f]{16,}|[0-9a-f]{24}__[0-9a-f]{24}|"
        r"\d{4}-\d{2}-\d{2}T\d{2}[:\-]\d{2})",
        flags=re.IGNORECASE,
    )
    rows = []
    violations = []
    for path in sorted((root / "data").rglob("*.csv")):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        bad_columns = [column for column in frame if forbidden_columns.search(column)]
        value_hits = 0
        for column in frame.select_dtypes(include="object"):
            value_hits += int(frame[column].str.contains(forbidden_values, na=False).sum())
        row = {
            "data_file": str(path.relative_to(root)),
            "rows": len(frame),
            "forbidden_columns": ";".join(bad_columns),
            "forbidden_value_hits": value_hits,
            "public_identifier_check": not bad_columns and value_hits == 0,
        }
        relative = str(path.relative_to(root))
        namespace_ok = True
        if "analysis_record_id" in frame:
            namespace_ok = bool(
                frame["analysis_record_id"].str.fullmatch(r"A\d{3}").all()
            )
        if "manuscript_profile_id" in frame:
            namespace_ok = namespace_ok and bool(
                frame["manuscript_profile_id"].str.fullmatch(r"P\d{3}").all()
            )
        if relative.endswith("figure_01_demographics/age_points.csv"):
            namespace_ok = namespace_ok and not {
                "analysis_record_id",
                "gender_birth",
                "race_response",
            }.intersection(frame.columns)
        if relative.endswith("figure_01_demographics/participants.csv"):
            namespace_ok = False
        linked_demographics = "age" in frame.columns and bool(
            {"gender", "gender_birth", "race", "race_response"}.intersection(
                frame.columns
            )
        )
        namespace_ok = namespace_ok and not linked_demographics
        row["namespace_and_minimization_check"] = namespace_ok
        row["public_identifier_check"] = (
            row["public_identifier_check"] and namespace_ok
        )
        rows.append(row)
        if not row["public_identifier_check"]:
            violations.append(row)
    report = pd.DataFrame(rows)
    if violations:
        details = pd.DataFrame(violations)[
            [
                "data_file",
                "forbidden_columns",
                "forbidden_value_hits",
                "namespace_and_minimization_check",
            ]
        ]
        raise AssertionError(
            "Potential raw identifiers remain in public data:\n"
            + details.to_string(index=False)
        )

    pooled = pd.read_csv(root / "data/n-f-2-to-5-participants.csv")
    profiles = pd.read_csv(root / "data/main/figure02/participant_profiles.csv")
    selected = pd.read_csv(
        root
        / "data/supplement/figure_05_selected_profiles/selected_profiles_long.csv"
    )
    if profiles["manuscript_profile_id"].duplicated().any():
        raise AssertionError("Manuscript profile IDs must be unique in Figure 2 data")
    if not set(profiles["analysis_record_id"]).issubset(
        set(pooled["analysis_record_id"])
    ):
        raise AssertionError("Figure 2 analysis-record IDs do not join to the pooled file")
    joined = profiles.merge(
        pooled,
        on="analysis_record_id",
        how="left",
        suffixes=("_profile", "_pooled"),
        validate="one_to_one",
    )
    value_pairs = {
        "face_fni": "face_fni",
        "geometry_fni": "geometry_fni",
        "scenery_fni": "scenery_fni",
        "4_nf_daily": "daily_familiarity",
        "4_food/cuisine": "daily_food",
        "4_places/activities": "daily_places",
        "4_movies or videos": "daily_movies",
        "4_restaurant": "daily_restaurant",
        "4_clothes": "daily_clothes",
        "4_digital games": "daily_games",
        "4_books or magazines": "daily_books",
        "4_snacks or treats": "daily_snacks",
    }
    for profile_column, pooled_column in value_pairs.items():
        left_name = (
            f"{profile_column}_profile"
            if profile_column in pooled.columns
            else profile_column
        )
        right_name = (
            f"{pooled_column}_pooled"
            if pooled_column in profiles.columns
            else pooled_column
        )
        error = (
            pd.to_numeric(joined[left_name]) - pd.to_numeric(joined[right_name])
        ).abs().max()
        if float(error) > 1e-12:
            raise AssertionError(
                f"Profile crosswalk disagrees with pooled {pooled_column}: {error}"
            )
    selected_map = selected[
        ["manuscript_profile_id", "analysis_record_id"]
    ].drop_duplicates()
    profile_map = profiles[
        ["manuscript_profile_id", "analysis_record_id"]
    ]
    merged = selected_map.merge(
        profile_map,
        on=["manuscript_profile_id", "analysis_record_id"],
        how="left",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise AssertionError(
            "Selected manuscript profiles do not match the documented ID crosswalk"
        )
    return report


def write_validation_reports(repository_root: str | Path) -> tuple[Path, Path, Path, Path]:
    root = Path(repository_root).resolve()
    output = root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    figure_report = validate_figures(root)
    table_report = validate_tables(root)
    privacy_report = validate_public_data(root)
    analysis_report = validate_analysis_crosschecks(root)
    figure_path = output / "figure_validation.csv"
    table_path = output / "table_validation.csv"
    privacy_path = output / "data_privacy_validation.csv"
    analysis_path = output / "analysis_crosscheck.csv"
    figure_report.to_csv(figure_path, index=False)
    table_report.to_csv(table_path, index=False)
    privacy_report.to_csv(privacy_path, index=False)
    analysis_report.to_csv(analysis_path, index=False)
    return figure_path, table_path, privacy_path, analysis_path
