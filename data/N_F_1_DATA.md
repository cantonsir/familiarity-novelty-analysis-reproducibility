# N-F-1 preprocessed data

`n-f-1-participant-trajectories.csv` is the compact participant-level input for
Supplementary Figure 2. It contains 1,560 rows: 15 release-pseudonymized
participants x 4 categories x 26 comparison positions.

The values are preprocessed familiar-target preference curves. For each
archived participant, responses were first recoded so positive values indicate
preference for the familiar target. Repeated observations were then averaged
within stimulus subcategory, participant, category and comparison position.
The public table begins at that preprocessed stage; it does not include raw
task exports, image names, filenames, platform identifiers or questionnaire
responses.

| Column | Meaning |
|---|---|
| `experiment1_record_id` | Release-only pseudonym (`E1A001`--`E1A015`); not linked to archived filenames or later experiments. |
| `category` | Face, Scenery, Geometry or Car. |
| `comparison_position` | Ordered comparison position, 1--26. |
| `familiar_target_preference` | Participant-balanced input value on the task's -3 to +3 scale; higher values indicate greater familiar-target preference. |

The Figure 2 renderer groups these rows by category and comparison position,
then recomputes the mean, participant SD, SEM and pointwise 95% t interval. It
requires exact agreement (absolute tolerance `1e-12`) with
`supplement/figure_02_experiment1_trajectory/plot_data.csv`, which is retained
as a frozen audit target.
