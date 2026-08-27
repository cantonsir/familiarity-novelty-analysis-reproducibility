# Released analysis data

This directory contains only the compact preprocessed, aggregate, or frozen
result tables needed by the public reproduction code. Raw task exports, task
code, image-level identifiers, platform identifiers, and submission metadata
are intentionally excluded.

## Participant-level files

- `n-f-1-participant-trajectories.csv`: 1,560 preprocessed rows from 15
  release-pseudonymized Experiment 1 participants, four categories, and 26
  comparison positions. Positive values indicate greater familiar-target
  preference. Supplementary Figure 2 recomputes its means and pointwise 95% t
  intervals from these rows.
- `n-f-2-to-5-participants.csv`: 305 release-pseudonymized participant records
  used for the integrated questionnaire analyses and their declared primary or
  sensitivity cohorts.
- `participant_data_dictionary.csv`: definitions, ranges, and non-missing
  coverage for the N-F-2--5 participant columns.

The Experiment 1 namespace (`E1A001`--`E1A015`) is independent of the
integrated-analysis namespace (`A001`--`A305`) and the manuscript display
namespace (`P001`--`P155`). Numeric suffixes must not be used to join these
namespaces.

## Output-specific inputs

- `main/`: compact inputs for Main Figures 2--5.
- `supplement/`: compact inputs for Supplementary Figures 1--11.
- `tables/`: frozen result rows and metadata for the supplementary tables.
- `output_input_manifest.csv`: exact output-to-input routing and reproduction
  level for every released component.
- `figure_manifest.csv`: expected artboards, text sentinels, active-asset names,
  and reference hashes.

Some files support full recomputation, while others are plot-ready or frozen
result tables. The `reproduction_level` column in
`output_input_manifest.csv` makes that boundary explicit for every output.

## Public-release boundary

`code/validate.py` scans all released CSV files for common raw identifiers and
checks the declared pseudonym formats and demographic minimization rules. These
checks support technical data minimization; they do not establish legal or
ethical permission to share participant-level derived data.
