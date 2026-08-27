# F/N manuscript figure and table reproduction

This compact repository regenerates the displayed analyses requested for the
Nature Human Behaviour manuscript:

- Main Figures 2–5 (Figure 1 is intentionally outside this repository);
- Supplementary Figures 1–11;
- Supplementary Tables 1–16; and
- the two unnumbered tabular displays in the Supplementary Information.

The figures use the same fixed artboards, Arial typography, colors, markers,
panel labels, annotations, and panel geometry as the active manuscript and
Supplementary Information. This is not a generic restyling of the results.

## Repository contents

```text
code/
  main_figures.py             Main Figures 2–5
  supplementary_figures.py    Supplementary Figures 1–11
  tables.py                   Supplementary table analyses and TeX rendering
  style.py                    Exact shared manuscript visual system
  reproduce_all.py            One-command entry point
  validate.py                 Font, analysis, vector-PDF, and privacy checks
data/
  n-f-1-participant-trajectories.csv
                              Pseudonymized E1 participant-level curves
  n-f-2-to-5-participants.csv Anonymized participant-level analysis file
  participant_data_dictionary.csv
  output_input_manifest.csv   Exact output-to-input routing
  figure_manifest.csv         Active asset/artboard provenance
  main/                       Final plot data for Main Figures 2–5
  supplement/                 Final plot data for Supplementary Figures 1–11
  tables/                     Frozen result tables and study metadata
notebooks/
  01_figures_and_tables_from_preprocessed_data.ipynb
outputs/
  figures/main/
  figures/supplement/
  tables/
```

No task code, raw task exports, platform identifiers, submission identifiers,
or the large historical figure-building framework is included.

## Reproduction level

The package deliberately distinguishes three reproducibility levels:

1. **Recomputed analysis + rendering.** Supplementary Figure 2 recomputes all
   participant-balanced means and pointwise 95% t intervals from the compact
   N-F-1 participant trajectory file, then cross-checks them against the frozen
   display summary. Supplementary Tables 4--7 recompute the correlation/BH
   families from the compact N-F-2--5 participant file. Their 66 rows are
   cross-checked against the frozen Main Figure 3/4 display inputs before those
   figures render. Table 8 refits the one-df random-x covariance path model for
   its point estimates and checks the frozen 10,000-resample intervals.
2. **Plot-data rendering.** Trajectory summaries, participant display
   coordinates, and other plot-ready values are rerendered with the active
   manuscript geometry.
3. **Frozen-result rendering.** Previously fitted HLM, bootstrap, reliability,
   and sensitivity results are supplied as small result-level CSV files. They
   are not refitted here because the upstream task/session data and large
   preprocessing framework are intentionally outside this release.

This boundary keeps the repository small while making every displayed number
and graphical element inspectable. `data/output_input_manifest.csv` links all
33 figure/table components to their exact inputs; the notebook displays this
map. `outputs/run_manifest.json` records input hashes and software versions.

## Run everything

Python 3.11 or newer is recommended. Arial must be installed to reproduce the
manuscript typography; the command fails before rendering if Arial is absent.
`latexmk` is optional and is used only to compile the 19-page all-table PDF
proof. The 18 TeX table fragments are generated without it.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/reproduce_all.py
```

Or open and run the single notebook:

```bash
jupyter lab notebooks/01_figures_and_tables_from_preprocessed_data.ipynb
```

The command and notebook write the same outputs. Existing output files are
replaced deterministically.

## Cohorts retained in the data

- E1 pilot laboratory cohort: 15 participants, represented by 1,560
  participant/category/comparison-position trajectory rows.
- E3/E5 matched primary behavioral cohort: 153 participants, 33,048 trials in
  the upstream analysis; final trial/model summaries are released here.
- E3/E5 all-completer profile cohort: 155 participants.
- E2–E5 primary questionnaire cohort: 303 participants.
- E2–E5 all-completer sensitivity cohort: 305 participants.

Experiment 4 remains a design-shift sensitivity and is not mixed into the
matched E3/E5 trajectory analysis. The two Figure 2 examples are descriptive
cases rather than participant types or clusters.

### Participant-ID namespaces

Three deliberately different pseudonym namespaces are used:

- `experiment1_record_id` uses `E1A001`--`E1A015` only for the released
  participant-level N-F-1 trajectory curves. It is not linked to the archived
  task filenames or to the N-F-2--5 participant file.
- `analysis_record_id` uses `A001`--`A305` and joins the compact participant
  file to released analysis tables.
- `manuscript_profile_id` uses `P001`--`P155` only for the within-profile order
  displayed in Main Figure 2 and Supplementary Figure 5.

Thus manuscript profile `P092` maps to analysis record `A242`, and manuscript
profile `P101` maps to `A251`. They are not pooled records `A092` or `A101`.
The explicit crosswalk is stored in
`data/main/figure02/participant_profiles.csv` and checked on every run.

## Output validation

`data/figure_manifest.csv` records the active manuscript filename, artboard,
required text sentinel, and SHA-256 reference hash for each of the 15 figure
PDFs. Validation requires one correctly sized, nonblank, vector-only page with
the expected figure text, all 18 table fragments, the exact cohort sizes, and
agreement between recomputed and displayed correlation families. Binary PDF
identity is reported only as provenance: metadata and PDF object ordering can
differ without changing the rendered page.

`outputs/manuscript_visual_parity.csv` is the author-side raster comparison
against the active LaTeX assets. Main Figures 2--5 are pixel-identical at 300
dpi. Supplementary figures are exact/near-exact except the three legacy
crop/relabel assets (5, 6, and 11), whose reader-facing pages are recreated
natively; their residual raster differences are recorded rather than hidden.

## Public-release gate

The N-F-1 trajectory data use release-only pseudonyms and contain only the
within-participant preference curves needed for Supplementary Figure 2. The
demographic microdata used to draw Supplementary Figure 1 were minimized: age
points are not linked to gender or race, and gender/race are released only as
aggregate counts. The automated scan rejects common platform identifiers, old
participant-ID namespaces, and a linked demographic table. This scan does
**not** establish legal or ethical de-identification.

Do not publish this repository until the corresponding author or data
controller confirms that the participant-level derived file and age points may
be shared under the study consent, IRB/ethics terms, and institutional data-
sharing policy. If approval does not cover those rows, keep them in a controlled
archive and publish only the aggregate/result-level inputs.

Also add author-approved code/data licenses and final citation/DOI metadata
before GitHub or archival release; these cannot be inferred from the analysis
files.
