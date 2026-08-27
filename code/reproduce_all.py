"""Single command-line entry point for every released figure and table."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Set the backend before Matplotlib is imported. The active assets were made
# with Matplotlib's macOS backend; Jupyter commonly exports an inline backend
# that changes a handful of antialiased panel-letter pixels in Figure 2. Other
# platforms use the portable noninteractive Agg backend.
os.environ["MPLBACKEND"] = "MacOSX" if sys.platform == "darwin" else "Agg"

import matplotlib
import numpy
import pandas
import scipy

matplotlib.use(os.environ["MPLBACKEND"], force=True)

import main_figures
import supplementary_figures
import tables
import validate


def _input_manifest(data_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(data_root.rglob("*")):
        if path.is_file() and path.name != ".DS_Store":
            rows.append(
                {
                    "path": str(path.relative_to(data_root.parent)),
                    "bytes": path.stat().st_size,
                    "sha256": validate.sha256(path),
                }
            )
    return rows


def reproduce(repository_root: str | Path) -> dict[str, object]:
    root = Path(repository_root).resolve()
    data_root = root / "data"
    output_root = root / "outputs"
    main_dir = output_root / "figures" / "main"
    supplement_dir = output_root / "figures" / "supplement"
    table_dir = output_root / "tables"

    arial_path = validate.require_arial()
    table_outputs = tables.render_all(
        data_root / "tables",
        data_root / "n-f-2-to-5-participants.csv",
        table_dir,
    )
    # Tables 4--7 recompute the correlation/BH families from the compact
    # participant file. Bind those results to the frozen display rows before
    # any Main Figure 3/4 rendering occurs.
    validate.validate_analysis_crosschecks(root)
    main_figure_01 = main_dir / "main_figure_01.pdf"
    main_figure_01_source = (
        root / "source" / "figure_01" / "main_figure_01_editable.pptx"
    )
    if not main_figure_01.is_file() or not main_figure_01_source.is_file():
        raise FileNotFoundError(
            "Main Figure 1 requires both its canonical PDF and editable PowerPoint source."
        )
    main_outputs = [
        main_figure_01,
        *main_figures.render_all(data_root / "main", main_dir),
    ]
    supplement_outputs = supplementary_figures.render_all(
        data_root / "supplement", supplement_dir
    )
    figure_report, table_report, privacy_report, analysis_report = (
        validate.write_validation_reports(root)
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "matplotlib": matplotlib.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
        },
        "arial_font": {
            "filename": arial_path.name,
            "sha256": validate.sha256(arial_path),
        },
        "inputs": _input_manifest(data_root),
        "editable_sources": [
            {
                "path": str(main_figure_01_source.relative_to(root)),
                "bytes": main_figure_01_source.stat().st_size,
                "sha256": validate.sha256(main_figure_01_source),
            }
        ],
        "outputs": {
            "main_figures": [str(Path(p).relative_to(root)) for p in main_outputs],
            "supplementary_figures": [
                str(Path(p).relative_to(root)) for p in supplement_outputs
            ],
            "tables": [str(Path(p).relative_to(root)) for p in table_outputs],
            "figure_validation": str(figure_report.relative_to(root)),
            "table_validation": str(table_report.relative_to(root)),
            "data_privacy_validation": str(privacy_report.relative_to(root)),
            "analysis_crosscheck": str(analysis_report.relative_to(root)),
            "author_side_visual_parity": "outputs/manuscript_visual_parity.csv",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Main Figure 1 and regenerate Main Figures 2–5 plus all "
            "supplementary figures/tables."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root of fn_analysis_reproducibility (default: inferred).",
    )
    args = parser.parse_args()
    result = reproduce(args.repo_root)
    print(json.dumps(result["outputs"], indent=2))


if __name__ == "__main__":
    main()
