# Paper Figure Workflow

This directory contains only small, versioned reference material for recreating
paper figures. Generated figures, local archives, saved sessions and analysis
caches are intentionally ignored by Git.

Use [the handoff guide](../docs/handoff.md) to configure another machine and the
[main README](../README.md) for the complete application workflow.

## Reference material kept in Git

`paper/reference/` contains compact BO configuration snapshots and the channel
ranking table used during candidate review. Treat these as provenance aids, not
as substitutes for the measurement data or saved BO sessions.

## Dataset roles

| Dataset | Electrode / role | Primary workflows |
|---|---|---|
| NP | Nanoporous | Parameter sweep, BO validation and titration |
| Planar/Kana | Planar | Parameter sweep, BO validation and titration |
| Amp0 | Planar comparison | BO validation and titration |
| Vancomycin | Vancomycin experiment | BO validation and mainly signal-off titration |

Always confirm the electrode and experiment identity from the source metadata.
Do not infer them only from a shorthand folder or saved-session name.

## Figure families

### Parameter sweep and BO validation

- Type 1/1A/1B/1C: measured parameter-sweep landscapes, slices, examples and
  measured cube views.
- Type 2/2A: real BO history and validation views.

For every BO figure, record the exact source session, optimizer direction,
physical channel/group and score shown. Original recorded Q, rescored Q and
pairwise SNR answer different questions.

### Titration figures

- Type 3: SWVs plus titration response.
- Type 3A: cross-channel signal-on/off comparison.
- Type 3B: signal-on/off with shared Langmuir treatment.
- Type 3C: paired signal-on/off Langmuirs.
- Type 3D: manual SWVs, optimized SWVs and response only.
- Type 4: concentration by measurement and predicted versus known.
- Type 4A: Langmuir plus both Type 4 panels.

Type 3D is the compact waveform/response layout. Type 4A is the full calibration
layout. Formatting presets are stored per figure type so the finalized Type 3D
layout does not overwrite Type 4A, and vice versa.

## Titration setup

1. Open the intended saved SWV session or analyze the intended titration folder.
2. Confirm the matched BO snapshot and processing settings.
3. Enable titration vlines and verify every interval label and concentration.
4. Select the concentration steps included in statistics and fitting.
5. Set the manual display start/end only when the automatic range is not the
   intended scientific range.
6. Inspect failures and the quality audit before generating paper panels.

The known-concentration and Langmuir panels update from the same current
selection. If the concentration range, accepted plateaus, methods or channel are
changed, generate the preview again.

Known working concentration spans include:

- NP: 40-5120 uM in `np_titration_aug13_all_channels` (all ten channels, with
  Channels 5 and 3 preselected) or the smaller
  `np_titration_aug13_high_quality`; for an 80-5120 uM
  presentation use grouped measurements **21-160**, retaining the immediately
  preceding buffer at measurements 21-30;
- Planar/Kana: 20-5120 uM;
- Amp0: 1000-64000 uM;
- Vancomycin: 0.2-51.2 uM.

The preferred high-range NP rows are Channel 5 optimized (signal-off; strongest
5120 uM recovery) and Channel 3 optimized (signal-on; cleanest single-peak SWVs).
Use the labels stored in the NP session rather than copying a range from another
experiment.

The response plateaus/error bars can be displayed as unconnected dashed black
summaries. Titration vline labels can be centered in their steps with the final
endpoint label omitted while preserving the vline positions.

## Layout and export

1. Choose the figure type and load its saved formatting preset.
2. Select the row channels and optimized/manual methods. For mixed signal-on and
   signal-off figures, place signal-on rows first unless the figure's scientific
   story requires another order.
3. Adjust the shared editable canvas. It is applied per repeated row.
4. Use the actual plot-area guides when aligning panels; unequal label margins
   can otherwise make equal outer boxes produce unequal-looking axes.
5. Click **Generate preview** and inspect labels, legends, plot-area size and row
   spacing.
6. Download PNG, PDF, SVG or ZIP from the separate buttons shown after preview.

Preview DPI defaults to 100 and publication raster export defaults to 600. Keep
the JSON metadata/recipe with each final figure. Store manuscript exports in the
chosen external results folder, not inside this repository.

## Interpretation and reporting

- A parameter-sweep optimum and a real-BO optimum need not be identical. Report
  both observed parameter triplets and scores; do not imply agreement unless the
  data demonstrate it.
- Compare optimization efficiency at a matched measurement budget. A best-of-200
  sweep value is not a fair direct benchmark for a 50-observation BO run; retain
  the full-sweep extrema in the audit data, but use equal-budget values for the
  main performance claim.
- Q values are comparable only when both the score definition and SWV peak
  extraction settings match. Rescoring saved peak metrics alone does not
  harmonize different peak finders.
- Cherry-picked channels should still represent the observed distribution. Keep
  enough channels to show repeatability, electrode differences and meaningful
  spread, while documenting any excluded corner/extreme candidates.
- Visual axis cropping, trace spacing and legend placement are presentation
  choices. They must not silently alter scoring, fitting or accepted scans.
- Type 4/4A normally reuses its calibration data. Describe it as
  predicted-versus-known within calibration, not independent validation, unless
  a genuinely separate validation dataset was used.
- Preserve raw inputs and final metadata so every table entry and plot can be
  traced back to its session/folder.

## What is deliberately not here

The following stay local and are covered by `.gitignore`:

- `paper/generated/` and `paper/outputs/`;
- `paper/.local_archive/`;
- raw or extracted titration folders;
- analysis-session recipes and `*.results.pkl.gz` caches;
- generated PNG, PDF, SVG and ZIP bundles.

Older tracked sample renders and superseded helper scripts were removed because
they were stale, large, and recoverable from Git history if ever needed.
