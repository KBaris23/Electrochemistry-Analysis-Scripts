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

For chronological stacks, **Auto-polish** restores the original outlier-exclusion
action (Conservative by default). It runs only when clicked and selects iterations
in the existing exclusion controls. It switches display repair/smoothing off.

**Auto-polish (keep traces)** clears those exclusions and enables trace-local
baseline repair: isolated step/spike repair, light smoothing with a window based
on the peak bracket, baseline anchoring at the trace's own correction minima,
and levelling of supported straight outer baselines. If an outer baseline still
fails, only that waveform is displayed within its own correction bracket. Other
waveforms retain the selected shared/custom window. Crops never expand a custom
range. Missing voltage segments are drawn with gaps rather than straight connectors.
There is no averaging across iterations or fitting to another waveform's shape.

Under **Baseline polishing**, choose **Repair then exclude unrecoverable** for
last-resort *trace* exclusions, or **Off** to restore the original display.
Keep traces retains unresolved waveforms and flags them for review. Recovery
retries saved analysis without BO acceptance gates or unsafe jump correction;
the original failure is reported. It does not change recorded acceptance or Q.
Missing/unreadable or uncorrectable measurements remain in the failure report.
**SWV polishing report** below the preview and exported panel audits list every
repair, crop, recovery and exclusion. Source files and analysis results are never
written by polishing. **Clear exclusions** also changes the last-resort mode back
to Keep traces.

The earlier Light/Moderate/Strong display-only filter remains under **Advanced
smoothing** and is used only when baseline polishing is off.

In the stack spacing controls, **Keep original iteration axis (leave gaps)**
preserves numeric positions and defaults to an axis from 1 to 50. Excluded
iterations leave empty positions, including at either end. Set a different
start/end for other runs; limits expand to include any retained trace outside
them. This option uses the shared-origin coordinate frame and is saved with
the panel for export. Turn it off to pack accepted iterations together.

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

Figure settings survive navigation away from Paper Figures. Switching types
keeps each type's latest formatting in the current workspace. **Save settings
to saved session** persists those formats and updates the visible controls
without erasing saved settings for hidden controls or changing the analysis
cache. Use **Overwrite formatting preset** only to change a type's global default.

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

## Titration templates and saved formatting

In **Saved Sessions**, select a session and click **Open**, then go to **Paper
Figures**. Choose a template and click **Generate preview**:

- **Type 3E - Titration and Langmuir** keeps the physical response/fit panel sizes
  derived from Type 3D.
- **Type 3F - SWVs only** shows the control and optimized SWV overlays.
- **Type 3G - Titration, Langmuir and accuracy** combines titration response,
  Langmuir fit, and a larger predicted-versus-known panel.
- **Type 3H - Vertical titration, Langmuir and accuracy** stacks response,
  Langmuir fit, and predicted-versus-known concentration from top to bottom.
  The plotting areas share the same width; every titration panel has its own
  concentration labels. Each selected channel gets its own vertical stack,
  arranged beside the other channels. Stack width controls each column's width.
- **Type 4B - Concentration prediction and accuracy** shows concentration by
  measurement and predicted versus known.

The **Titration figure preset** menu loads the approved export styles from the
tracked `paper_titration` section of `figure_composer_presets.json`. Click
**Apply titration preset**, then **Generate preview**. Six presets are included:
horizontal and vertical larger-text and double-text layouts, plus vertical
balanced text and balanced text with a slightly taller prediction panel.
The latter keeps every plot width and the upper two plot heights unchanged,
and increases only the prediction plotting area's height by 25%.
All presets retain concentration labels on every titration panel.

Presets store appearance; the open session supplies channels, methods, ranges,
and fitting selections. They work with the planar kanamycin, nanoporous
kanamycin and vancomycin sessions. For the exact exported dataset selection,
import its PNG metadata or `figure-metadata.json`, including the vancomycin
final buffer/target exclusion, then apply a style. Use **Save settings to saved
session** to retain it. The existing per-type formatting overwrite/reload
controls save customized defaults locally; the six shared presets remain
available. Rendered panels can also be captured in **Figure Composer**.

Each session retains its separate template layouts. Use **Save settings to
saved session** after editing; reopening the session restores these settings
without changing its cached analysis. Titles, legend labels, font sizes,
panel positions, and aspect ratios remain editable. Method colors apply to
response/fitting curves; SWV measurement colormaps are configured separately.

Under **Panel layout, axes, and curve style**, **Use plain Arial throughout**
applies regular Arial to panel text. **Preserve panel canvas and vector PDF/SVG**
keeps physical font sizes and exports source curves and text as vectors.
Colorbar gradients and raster source panels remain images. Download the complete
PNG/PDF/SVG/ZIP figure or choose an individual plot for PNG/PDF/SVG export.

Type 3G's **Equal spacing between plotting areas** matches gaps between visible
plot edges, including in PDF/SVG exports. Disable it to place panels manually.
Concentration labels default to 5.5 pt and stay centered within their dose
intervals. Overlapping text can move vertically within an interval; it never
expands axis limits. Buffer shading, one-time units, and top-row-only labels are
optional. Prediction panels reuse the calibration data and therefore show
concentration reconstruction, not independent validation.

Concentration labels now default to every titration panel. The top-row-only
option remains available for matching schedules. Type 3H supports a taller
comparison canvas through the row-height control. Import a generated figure's
metadata to restore its exact channel selection, display range and excluded
concentration steps.

Dataset-specific batch scripts, figure packages, and research notes remain
local and Git-ignored. Shared plotting helpers are available through
`core.paper_replay.app_functions()` for reproducible batch rendering without
starting Streamlit. Keep saved sessions and source data separately when moving
the workflow to another machine.

## What is deliberately not here

The following stay local and are covered by `.gitignore`:

- `paper/generated/` and `paper/outputs/`;
- `paper/.local_archive/`;
- raw or extracted titration folders;
- analysis-session recipes and `*.results.pkl.gz` caches;
- generated PNG, PDF, SVG and ZIP bundles.

Older tracked sample renders and superseded helper scripts were removed because
they were stale, large, and recoverable from Git history if ever needed.
