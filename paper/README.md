# Paper figures: real-data runbook

Updated 16 October 2026. Order: **sweep, BO validation, titration, concentration diagnostics**. These are preliminary figures, not publication approval. Raw data were not modified.

## Nanoporous NP: native application workflow

Final review folder:
`%USERPROFILE%\Downloads\Figures materials for the paper\Results\journal_plots\np`.
See [Recreating figures](../docs/recreating-figures.md) for exact session and
metadata instructions. Every required template is available in the application.

| Figure | Saved session | Selection |
|---|---|---|
| Type 1C sweep | `np_sweep_type1c` | Channel 2; SWVs 87/80; 4/7 mV slices; one cube with points and planes |
| Type 2 validation | `np_validation` | Group/channel 5; recorded optimization data |
| Type 3 titration | `np_titration` | Optimized/manual pairs on channels 2–5 |
| Type 3A ON/OFF | `np_titration_type3a` | Channel 5 ON and channel 3 OFF, each with its own manual reference |
| Type 4 concentration | `np_titration_type4` | Same four-channel titration calibration |

For sweep/validation open **Figure Composer** and render/export. For titration
open **Paper Figures** and click **Generate paper figure**. Fits and responses use
source measurements **41–160**, displayed as **1–120**: buffer alternating with
**80, 160, 320, 640, 1280, 2560 µM**. SWV overlays retain the full analyzed range.
Extreme filtering and **Ignore the first target response** stay off. Type 4 is a
calibration diagnostic, not held-out validation.

The fresh native analysis rejects 20 optimized channel-2 traces whose left
minimum is below the saved -0.44 V acceptance limit. The legacy batch cache had
marked these as accepted. Current caches retain those failures, and accepted-only
figures exclude them; no raw measurements or acceptance limits were changed.

Type 3A's shared panel contains both baseline-referenced Langmuir fits and
matched-dose ON-minus-OFF values. The difference is descriptive, not a third
independently fitted curve. Negative signal-off values remain negative.

Use **150-DPI PNG plus metadata** for this review; leave PDF/SVG off. Type 1C now
renders its SWVs from numeric source traces through the application. The raster
fallback and incorrect copies have been retired. Scoring profiles, selected
points, slices, methods, and layout are recorded in metadata and saved sessions.

Session copies are in `paper/generated/np_sessions/`; active sessions are in
`%LOCALAPPDATA%\Electrochemistry Analysis Scripts\analysis_sessions`.
Keep the source volume connected for reanalysis. Metadata restores settings; a
session and its trusted cache restore the analyzed workspace. Neither replaces
the original inputs. Raw data, caches, and local recipes are excluded from Git.

The following sections document earlier releases and historical figure names.

## 1. Open the generated figures

Use the consolidated [journal plots](generated/journal_plots/) or [download the ZIP](generated/journal_plots.zip). Validation and titration now live together under each dataset, with only maximize/minimize subfolders. Each newly rendered figure has PNG/PDF/SVG and one metadata JSON. Kana alone has the two-row Type 3A exports (app equivalent: Type 3C). Amp0 stacks use per-trace correction brackets to avoid extreme off-bracket tails; recorded Q is unchanged. See [the short guide](generated/journal_plots/READ_ME.txt). PDF/SVG composite panels contain rasterized content, not fully editable vector curves.

The older selection table and detailed investigation notes below are historical, not the current export manifest. Superseded folders are recoverable under ignored `paper/.local_archive/superseded_20261011/`, outside the active outputs. Raw data and local sessions were not moved. Use the consolidated folder above for current figures; do not mix older Amp0 shared-window stacks into it.

**Reference style:** Type 3/3B SWVs are **overlaid, not offset stacks**, using the standard SWV plotter and its purple-to-yellow (`plasma`) measurement colorbar. Only analysis-accepted **smoothed + corrected** traces are shown, clipped between final correction minima, with residual endpoint baseline removed using the existing anchor-offset method. Dashed zero line and anchor dots match the standard SWV view. These display operations do not recompute peak metrics or fit parameters.

Response plots use **dark blue = Optimized Method**, **light blue = Manual Method**, with an in-panel legend. They show **Change in Peak Current**: each target measurement minus its immediately preceding buffer plateau; buffer measurements minus their own plateau (noise is retained). Langmuir curves/points have their anchor-buffer value subtracted for display only. Signal-off remains negative. In Type 3B, the shared fit distinguishes optimized ON (blue) and OFF (red); ON-minus-OFF now subtracts the two baseline-referenced changes, not unrelated absolute baselines.

Type 3C keeps the two Type 3B directional rows but replaces the shared optimized-only final column with two row-specific Langmuir panels. Each compares that row's optimized method against the same unoptimized/manual reference.

To make these again without starting over:

1. Restart the app, then **Saved analysis sessions > Saved session > Open**: `kana_try_titration`, `amp0_titration` or `vanco_titration`.
2. Choose **View > Paper Figures** and the channel/type listed below. Click **Use reference titration style** (important when reopening older saved settings). This keeps your methods/range but selects **Overlaid**, stride **1**, **Zero correction anchors**, **Accepted SWVs only**, and **Change from preceding buffer**. Keep extreme filtering OFF for the main figures.
3. Use **Generate paper figure > Download Package**. The ZIP already uses `dataset/maximize/` or `dataset/minimize/` when one direction applies, and otherwise the dataset root. Detailed per-figure settings are in sections 6–7.

| Figure | Open PNG | Assessment |
|---|---|---|
| Kana sweep, ch3, Type 1A | [Sweep](.local_archive/superseded_20261011/preliminary_journal_plots/kana_try/sweep_ch3_bo_111904_a0d8ad_multipanel.png) | Main landscape candidate |
| Kana BO, ch3 maximize, Type 2 | [Validation](.local_archive/superseded_20261011/preliminary_journal_plots/kana_try/maximize/ch3_bo_113013_8dd581_multipanel.png) | Matches the recovered titration channel |
| Kana ch3 ON/manual, Type 3 | [Titration](.local_archive/superseded_20261011/preliminary_journal_plots/kana_try/maximize/ch3_type3.png) | Matches your earlier strong channel/segment |
| Kana ch5 ON/OFF, Type 3B | [Two-direction comparison](.local_archive/superseded_20261011/preliminary_journal_plots/kana_try/ch5_type3B.png) | Cleaner two-direction example |
| Kana ch3 ON/OFF, Type 3B | [Alternative](.local_archive/superseded_20261011/preliminary_journal_plots/kana_try/ch3_type3B.png) | OFF saturation less convincing than ch5 |
| Kana ch5 ON, Type 4 | [Calibration diagnostic](.local_archive/superseded_20261011/preliminary_journal_plots/kana_try/maximize/ch5_type4_calibration_diagnostic.png) | Not independent prediction validation |
| Amp0 BO, ch6 minimize | [Validation](.local_archive/superseded_20261011/preliminary_journal_plots/amp0/minimize/ch6_bo_145633_9f5760_multipanel.png) | Alternative to the earlier ch10 choice |
| Amp0 ch6 OFF/manual, Type 3 | [Titration](.local_archive/superseded_20261011/preliminary_journal_plots/amp0/minimize/ch6_type3.png) | Detectability example; weak calibration |
| Amp0 ch10 OFF/manual, Type 3 | [Original candidate](.local_archive/superseded_20261011/preliminary_journal_plots/amp0/minimize/ch10_type3.png) | Retained, not a strong Langmuir result |
| Amp0 ch6, Type 4 | [Diagnostic only](.local_archive/superseded_20261011/preliminary_journal_plots/amp0/minimize/ch6_type4_calibration_diagnostic.png) | Not recommended as a paper prediction result |
| Vanco BO, ch6 minimize | [Validation](.local_archive/superseded_20261011/preliminary_journal_plots/vanco/minimize/ch6_bo_142117_84662a_multipanel.png) | Physical ch6 is BO **Group 4** |
| Vanco ch6 OFF/manual, Type 3 | [Titration](.local_archive/superseded_20261011/preliminary_journal_plots/vanco/minimize/ch6_type3.png) | Modest response, usable fit |
| Vanco ch2 OFF/manual, Type 3 | [Alternative](.local_archive/superseded_20261011/preliminary_journal_plots/vanco/minimize/ch2_type3.png) | Higher full-series R² |
| Vanco ch6, Type 4 | [Calibration diagnostic](.local_archive/superseded_20261011/preliminary_journal_plots/vanco/minimize/ch6_type4_calibration_diagnostic.png) | Same-calibration inversion only |
| Kana station 2 BO, ch10 maximize | [Replicate/SI](.local_archive/superseded_20261011/preliminary_journal_plots/kana_st2/maximize/ch10_bo_171120_dc5dbc_multipanel.png) | Separate run, not pooled with setup 5 |

Amp0 ch10 BO is also retained under `amp0/minimize/`.

Compact regeneration recipes and marker files are under `audit/recipes/`; current PNGs remain in the dataset/direction folders. BO figure folders retain their `ui_settings.json`. Superseded PNG/PDF/SVG copies and stale UI exports were removed.

Filtering, simply:

- **Main PNGs:** extreme filtering OFF; no interpolation, missing-dose patching or moving-average replacement.
- **Separate sensitivity check:** `kana_try/maximize/filtered_sensitivity_ch3_type3.png`, extreme filtering ON. The response plot and fits now honor the same exclusions. Original measurements remain intact.
- **Smoothing:** applies to each voltage/current SWV, using the recorded analysis settings (15-point, order 2), not across concentration steps. Cropping never invents a missing waveform or baseline bracket.
- **Accepted-only waveform previews:** failed scans are omitted just as in the standard SWV overlay, not repaired or counted as successful responses. The full failure counts remain in section 8 and the audit. Waveform colorbars follow the standard plotter's within-panel accepted-measurement sequence; sparse methods can have fewer displayed measurements.

The review folder is intentionally compact: current PNGs and Composer `.figure.json` sidecars live with each dataset, while `audit/recipes/` retains the small settings/marker inputs needed for regeneration. Large stale image archives were removed; use the figure links above.

### How these were made

The original figures were generated through **Streamlit AppTest widgets**. Later local releases replayed saved sessions and called the application's analysis/plotting helpers directly, including optional jump reanalysis; they were not all browser/widget runs. Regeneration scripts and private paths stay local. Use each plot's metadata for its actual processing and fit scope, and inspect final exports at publication size. Concentration checks reuse their fitted titration data and are not independent validation.

## 2. Launch and save

Replace `<YOUR_CHECKOUT>`, `<DATA_ROOT>` and `<PAPER_FIGURES>` in this guide with your local paths; these are placeholders, not bundled datasets. Generated exports may have been moved outside the checkout. Their sidecar metadata remain the authoritative recipes.

```powershell
cd "<YOUR_CHECKOUT>"
.\Open_SWV_App.cmd
```

Restart the app process once after these code updates so the new QC functions load.

Save with sidebar **Saved analysis sessions > Save as > Save**. New recipes and recovery live outside the checkout in local app data (Windows: `%LOCALAPPDATA%\Electrochemistry Analysis Scripts\analysis_sessions`). **Include derived analysis cache for fast reopen** is optional: it adds tens of MB, not a copy of raw data. Older app-relative `analysis_sessions/` recipes remain available and untouched; those older copies may still sync through OneDrive.

The normal list now also includes the three nanoporous workspaces: `np sweep`, `np validation`, and `np titration`, alongside the seven Kana/Amp0/Vanco sessions. Select one and click **Open**. Validation here means the BO/chronological-stack workspace; concentration diagnostics remain available by switching to Type 4 within a titration workspace. **Show older/extra sessions** reveals previous per-figure variants, Kana ST2, legacy recipes, and recovery. Originals and shared caches were retained, not deleted; new saves remain visible normally.

In BO Figure Composer, choose **Shared channel** for Type 1/1A/1B or **Validation channel** for Type 2/2A. Composer uses the full loaded experiment even if a restored sidebar **Channel groups** filter says Group 3. Choosing a new channel in a linked panel's **Channels** box also updates the shared selection; later metric/formatting edits must not restore the preset's original channel. Paper Figures has its own **Physical channel** selector for Types 3/3B/3C/4. Raw imported images and legacy captured plots require replacing/regenerating the source image rather than changing a measurement channel.

Titration and its concentration diagnostic can use **the same SWV analysis session**: switch **Paper Figures → Type 3 / Type 4**. Separate Type 4 recipe names are conveniences, not separate acquisitions. Type 4 currently reuses the calibration fit; it is not independent validation. A genuinely separate validation experiment needs its own source-data session.

For the compact concentration layout, choose **Type 4 → Concentration validation layout**: right-panel width **45%**, gap **0.04**, **RMS fold error position → Above plot**, and **Square predicted-vs-known axes OFF**. The right panel uses its width, error summaries sit outside the data, and the canvas remains 3.1 inches per row. Above/Below/Inside, width and gap are editable and recorded in figure metadata. This changes formatting only, not fits, error bars or predictions. Click **Generate paper figure → Download PNG**, then save your local session. See the main README's **Reopen the prepared paper figures without entering paths** for the complete click sequence.

Raw paths must remain accessible. Moving a recipe alone does not make its data portable. Use the app's Open control; do not assume a Windows double-click association or save prompt when closing the browser. Keep a named save, not only autosaved recovery.

## 3. Figure A: planar-kana sweep

1. Select **Analysis mode > BO Session**. Paste this in **Experiment or session folder**:

   ```text
   <DATA_ROOT>\500um_planar_BO_try_again_20260918_112055\parameter_sweep_20260921_110003\bo_sessions\bo_111904_a0d8ad
   ```

2. Select `bo_111904_a0d8ad`, not the empty `bo_111849_1099a5`. Keep **Original recorded Q**. Set **Channel groups: Group 3**.
3. Open **Figure Composer > Load a preset or saved figure**. Select **Type 1A - Sweep comparison (cube left)**, wait for the selection rerun, then **Load preset**.
4. Keep shared channel **3**, automatic ON/OFF examples and **two step slices: 4 and 7 mV**. This export uses ON iteration **174** (500 Hz, 40 mV amplitude, 2 mV step) and OFF **184** (108 Hz, 60 mV amplitude, 6 mV step). These are sweep examples, not the later titration methods.
5. Keep preset layout/camera/fonts. Traces are **Raw / unsmoothed (corrected)**, with snapshot correction and the selected peak-bracket crop. No Q clipping or exclusions were applied.
6. Optionally click **Render figure (fast preview)** to inspect, then **Export PNG** or **Export all (ZIP)**. No preview prerequisite or separate final-render button remains. Each format includes embedded metadata; Export all also includes the JSON recipe. For this preliminary round, save at the dataset root as `kana_try/sweep_ch3_…`; do not create a channel/type subfolder.

For exact formatting, upload the saved `.figure.json` or `.figure.zip` after loading this session. **Edit panel** changes content, camera, markers, text and layout; interactive camera changes must be cached/applied. Type 1B mirrors left/right. Type 1 is the larger four-slice layout. Sweep presets require a survey session, not an optimization session.

## 4. Figure B: BO validation for each run

**Kana setup 5**

```text
<DATA_ROOT>\500um_planar_BO_try_again_20260918_112055\planar_BO_kana_20260918_112056\bo_sessions\bo_113013_8dd581
```

**Amp0**

```text
<DATA_ROOT>\100um_amp0_high_conc_20260915_130213\amp0_100um_20260915_130234\bo_sessions\bo_145633_9f5760
```

**Vanco**

```text
<DATA_ROOT>\vanco_first_try_20260826_132944\vanco_try_1_20260826_132945\bo_sessions\bo_142117_84662a
```

**Kana station 2**

```text
<DATA_ROOT>\500um_planar_kana_20260917_170330\500um_planar_kana_20260917_170349\bo_sessions\bo_171120_dc5dbc
```

| Run | Sidebar group | Composer direction | Linked channel |
|---|---:|---|---|
| Kana setup 5 | 3 | maximize | `3_max` |
| Amp0 preferred preliminary | 6 | minimize | `6_min` |
| Amp0 earlier candidate | 10 | minimize | `10_min` |
| Vanco | 4 | minimize | `6_min` |
| Kana station 2 | 10 | maximize | `10_max` |

For each row:

1. Paste its folder in **BO Session**, select its group, keep **Original recorded Q**.
2. In **Figure Composer**, select **Type 2 - BO validation**, then **Load preset**.
3. Set **Optimizer shown in BO panels** to the table's direction. Check linked group/channel across Q, paired metrics, cube, stack and parallel coordinates; do not leave Q on all groups.
4. Keep the preset: all successful iterations, dashed **5-point trailing mean**, best-so-far line and best-observed marker in the metric panels; buffer/target **Peak prominence**; snapshot-corrected unsmoothed chronological traces, no normalization. The chronological panel does **not** mark the best iteration. **Chronological voltage range** offers **Corrected peak bracket** (default) or **Full analyzed voltage range** (all points retained by the saved BO analysis crop, not uncropped acquisition data). This display choice does not recompute Q. Buffer/target replicates share one iteration plane with actual accepted IDs; newer rear traces fade. The white-backed, upper-right Buffer/Target legend matches the metric legend style without the best-observed entry. Current ticks use detected peak height or an editable maximum. See the main README's Composer steps for all controls. Regenerate after changing settings; old PNGs do not update themselves.
5. Click **Export PNG/PDF/SVG** or **Export all (ZIP)**; preview is optional. Final rendering happens on demand and unchanged downloads reuse it. Save under `dataset/maximize/` or `dataset/minimize/`; keep channel and figure type in the filename, for example `amp0/minimize/ch6_bo_….png`. Every format embeds the recipe, including camera and formatting; Export all adds the JSON sidecar. On older Streamlit, use the Download button shown after format preparation.

Type **2A** is the focused alternative, omitting stack/parallel coordinates. Optimization inputs offer 2/2A; survey inputs offer 1/1A/1B. Types 3/3B/4 belong to SWV mode.

Peak prominence is noise-normalized, not current in microamps. Recorded `Q_run` can differ from the cube's paired-Q measure because of objective aggregation/penalties. Larger raw peaks do not necessarily mean better paired SNR. Best-so-far necessarily improves by construction; it does not establish superiority to random search without a matched benchmark.

## 5. Figure C: prepare titration in SWV mode

| Dataset | Folders input | Scans analyzed | Per-method display interval |
|---|---|---:|---|
| Kana setup 5 | `<DATA_ROOT>\titration_only\kana_try` | 6,000 | 21-200 |
| Amp0 | `<DATA_ROOT>\titration_only\amp0` | 4,800 | 21-160 |
| Vanco | `<DATA_ROOT>\titration_only\vanco` | 4,800 | 21-200 |

1. Select **Analysis mode > SWV** and replace **Folders** with one input above, not all three. These `titration_only` folders are deliberately narrowed copies that omit warm-up/BO scans; they are not a required software layout. The app can instead receive an ordinary experiment root or nested measurement folder, recursively discovers SWV CSVs, and looks for a matching BO snapshot beside/above it. For an extracted folder, retain its copied `session_log.txt` so provenance matching remains available.
2. Leave automatic BO-settings loading enabled. Verify the snapshot path belongs to the corresponding BO session in section 4. For extracted inputs the app resolves the original run; normal experiment folders resolve their associated session. If ambiguous, explicitly select that session's `bo_config_snapshot.json` and load it. Use the settings-check control before overriding anything.
3. Choose **Grouping > SWV settings**, **SWV peak height source > Corrected + smoothed**, all physical channels. Click **Run Analysis** once and wait.
4. Enable **Treat vline intervals as titration steps** and **Fit Langmuir**. Check the queue-derived vlines. If absent, paste the matching app-exported [kana markers](.local_archive/superseded_20261011/preliminary_journal_plots/audit/recipes/kana_try/vlines.txt), [amp0 markers](.local_archive/superseded_20261011/preliminary_journal_plots/audit/recipes/amp0/vlines.txt) or [vanco markers](.local_archive/superseded_20261011/preliminary_journal_plots/audit/recipes/vanco/vlines.txt).
5. Set **Titration baseline mode > Immediately preceding buffer**, units **uM**, **Plateau edge trim fraction = 0.15**.
6. Under **Concentrations included in titration statistics**, deselect only `buffer_1` and the first duplicate low-dose block (`20 uM_1`, `1000 uM_1` or `0.2 uM_1`). Keep the second low-dose block and all later buffers/doses. This treats the first pair as equilibration per the supplied design; confirm the rationale in the notebook.
7. Leave **Remove extreme titration outliers OFF**. LOD/ULOQ display annotations were OFF; numerical estimates still export. Click **Apply Display Controls**.
8. Compact final-state recipes and focused Amp0 crop evidence are under `audit/`. For new analysis settings, use **Export** for updated CSVs and **Quality audit** for waveform flags/exclusion candidates; keep them outside the PNG review folder.

### Exact analysis settings

Kana setup 5 and the non-manual Amp0 methods share these settings; vanco differs in minima requirements:

| Control | Value |
|---|---|
| Crop | -0.55 to 0.00 V |
| Smoothing | window 15, polynomial order 2 |
| Minima search window | 0.30 V |
| Minimum start voltage | -0.60 V |
| Minimum peak height | 0.001 uA |
| Double correction | ON |
| Prominent minima / require local minima both sides | ON / ON for kana and amp0; OFF / OFF for vanco |
| Accepted peak voltage | -0.45 to -0.10 V |
| Left minimum acceptance | -0.54 to -0.10 V |
| Right minimum acceptance | -0.45 to -0.01 V |
| Apply BO acceptance windows | ON |
| Wavelet denoising / wavelet correction | OFF / OFF |

Station 2 BO uses its own snapshot: crop -0.45 to +0.10 V, minimum start -0.50 V and different acceptance windows. Do not overwrite it with this table.

Amp0 Channel 6 manual (200 Hz) is the audited exception: crop -0.48 to -0.12 V, peak -0.41 to -0.16 V, left minimum -0.48 to -0.25 V and right minimum -0.30 to -0.12 V. This accepts 135/160 raw manual traces versus 20/160 with the global crop. Applying it globally reduced the optimized 347 Hz method to 3/160 accepted, so the saved analysis session records a method-specific override and retains the original crop for every other method. No values were imputed; see `audit/amp0_method_specific_crop.json`.

Vlines use **physical-channel acquisition scans**: `1,buffer; 31,lowest dose; 61,buffer; 91,lowest dose again; 121,buffer; 151,next dose; ...`. End markers: 601 kana/vanco, 481 amp0. Grouping maps these to method-local 1,11,21,31,...; **do not divide pasted vlines yourself**. Doses double: kana 20-5,120 uM, amp0 1,000-64,000 uM, vanco 0.2-51.2 uM.

Display start 21 removes the first 20 **method** scans (initial buffer/target pair), not 20 raw files. A display crop does not alter fit inclusion. Selected doses feed the existing hybrid fit: Langmuir through the largest absolute response, with later points retained and potentially a polynomial guide. Export now distinguishes R² on that fitted branch from R² across all selected accepted targets. Do not report one as the other.

## 6. Figure C: generate Type 3 or 3B

Open **SWV > Paper Figures**. Common settings: **14 in width, 10 pt font, panel raster DPI 300, Overlaid SWVs, stride 1, plasma colorbar, Overlaid response curves, no extra columns, panel letters OFF**. Click **Use reference titration style**; keep **Corrected peak region (between minima)**, **Zero correction anchors ON**, **Accepted SWVs only ON**, and **Change from preceding buffer**. All waveform panels use smoothed + corrected current; bounds come from the final correction pass. PNG exports at 600 DPI; constituent panels retain their selected raster DPI. Use **Download PNG** only for this preliminary round. Save Type 3/4 under `dataset/maximize/` or `dataset/minimize/`; save mixed-direction Type 3B at `dataset/`. Never create a separate `titrations/` folder or a channel-named subfolder.

### Kana ch3: recovered earlier channel, Type 3

Select **Type 3 - SWV and titration response**, one row, physical **ch3**, optimized **500 Hz / 50 mV / 1 mV (maximize)**, manual **200 Hz / 36 mV / 2 mV**, display **21-200**. Click **Generate paper figure > Download Package**. The primary file belongs at `kana_try/maximize/ch3_type3.png`.

Your older SVG under `<PAPER_FIGURES>\08_Q_scoring_scheme` matches this method/segment. Its 179 values match current app-export values essentially exactly (squared correlation approximately 1). It omits method scan **189**, physical scan **565**, which is currently accepted but flagged by the existing extreme filter. Primary figures retain it. A separate filtered Type 3 is in `kana_try/maximize/filtered_sensitivity_ch3_type3.png`; no replacement point was synthesized. The original exclusion rationale cannot be proved from the SVG alone.

The adjacent `resources/conceptual_titration.py` creates synthetic data and was **not** used as experimental evidence.

### Kana ON/OFF: Type 3B

Select **Type 3B - Signal-on/off shared Langmuir**, physical **ch5**, ON **500/40/2**, OFF **138/90/3**, manual **200/36/2** (Hz/mV/mV), display **21-200**. Generate/download the package as `kana_try/ch5_type3B.png`.

Ch3 alternative: ON **500/50/1**, OFF **125/70/8**, same manual. Type 3B has two rows and one shared final column containing optimized ON, OFF and matched-dose ON-minus-OFF. The difference is descriptive, **not a third independently fitted binding curve**. Manual SWVs/time courses remain in the first three columns, not the shared Langmuir panel.

### Amp0: Type 3 detectability comparison

Use **ch6**, optimized **347 Hz / 70 mV / 10 mV (minimize)** versus manual **200/36/2**, display **21-160**. The saved session applies the documented focused crop only to this manual method. Generate/download the package as `amp0/minimize/ch6_type3.png`.

Earlier ch10 candidate: optimized **362/80/9 (minimize)** versus manual, same interval; retained as `amp0/minimize/ch10_type3.png`. Neither establishes robust full-range calibration. Many manual scans and some high-dose optimized blocks fail detection; these were not filled. Do not force an ON/OFF layout when both directions are not supported by the data.

### Vanco: Type 3

Use **ch6**, optimized **401/70/5 (minimize)** versus manual **200/36/2**, display **21-200**. Save the package under `vanco/minimize/ch6_type3.png`.

Alternative **ch2**: optimized **322/60/4 (minimize)** versus manual, same interval; `vanco/minimize/ch2_type3.png`. Chip `350_r_np_45_15` is recorded; confirm what `np` means before labeling it planar/nanoporous.

## 7. Figure D: Type 4 diagnostics

Keep the same channel/method/step selection, choose **Type 4 - Concentration validation**, Generate, download. Examples use kana **ch5 ON**, amp0 **ch6 OFF**, vanco **ch6 OFF**, each against same-channel manual with the intervals above. Save under `<dataset>/<maximize|minimize>/ch<channel>_type4_calibration_diagnostic.png`.

These invert a calibration fitted to these data: **not held-out validation**. Inversion is unstable near saturation; unavailable predictions remain missing. Amp0 Type 4 is an audit output, not a recommended paper result. Do not hide failed predictions or insert averaged concentrations into accuracy statistics.

## 8. Findings and limitations

Current app results below: unfiltered, preceding-buffer correction, initial equilibration pair excluded. R² evaluates the Langmuir model across **all selected accepted target doses**, not only its fitted branch. SNR is the largest dose-level value in the exported table, not BO Q or performance at every dose.

| Method | Full-series R² | App Kd, uM | App LOD, uM | Max titration SNR |
|---|---:|---:|---:|---:|
| Kana ch3 ON | 0.9846 | 265.0 | 9.98 | 78.8 |
| Kana ch3 manual | 0.5640 | 136.6 | 9.67 | 33.2 |
| Kana ch5 ON | 0.9858 | 422.7 | 25.52 | 48.0 |
| Kana ch5 OFF | 0.9742 | 655.7 | 78.85 | 23.0 |
| Kana ch5 manual | 0.9573 | 166.8 | 8.61 | 53.7 |
| Amp0 ch6 OFF | 0.6808 | Not reliable | Not recommended | 10.6 |
| Amp0 ch10 OFF | -367.7 | Not reliable | Not recommended | 3.3 |
| Vanco ch2 OFF | 0.9719 | 0.384 | 0.187 | 6.25 |
| Vanco ch2 manual | 0.8815 | 0.719 | 0.459 | 5.51 |
| Vanco ch6 OFF | 0.9319 | 0.871 | 0.350 | 8.06 |
| Vanco ch6 manual | 0.7857 | 0.440 | 0.406 | 4.03 |

Kd/LOD are model outputs, not independently established constants/limits. Some manual fits stop before later selected doses. Amp0 ch10's two-dose fitted branch has R² 0.7213 but extrapolates badly to later selected doses; its very negative full-series R² is not a typo. Ch6's fitted-branch R² is 0.9148, not its full-series value. The earlier blanket amp0 R² claim and vanco's alleged 1 uM Kd floor were not reproduced.

Defensible preliminary storyline: strong kana concentration dependence and selected-method advantages; amp0 detectability rescue and modest vanco improvements as qualified secondary results. There is **no universal LOD/SNR advantage**: kana ch5 manual has lower fitted LOD and higher maximum SNR than optimized ON. Preserve paired comparisons and all-channel tables. These figures alone do not prove universal aptamer performance, specificity, reversibility or BO superiority to random search.

### Waveform audit

Entire inputs, all methods, before display crops/outlier filtering:

| Dataset | Total | Analysis failures | Jump-screen candidates |
|---|---:|---:|---:|
| Kana setup 5 | 6,000 | 1,029 | 857 |
| Amp0 | 4,800 | 1,363 | 455 |
| Vanco | 4,800 | 214 | 1 |

Jump candidates are screening hints, not confirmed artifacts or exclusions: internal adjacent-current change above 10 robust derivative SD and 15% of waveform range. Sharp peaks, crop edges and gaps can trigger them. Use **Quality audit > Inspect analysed waveform > Preview waveform QC** to compare stored raw/corrected/smoothed arrays and download. Screening covers the analysis crop, not voltages outside it.

- Kana ch3: ON/manual each 200/200 accepted, OFF 199/200; its one jump candidate is near the crop edge, physical scan 421. Ch5 ON 194/200, OFF/manual each 200/200. Ch9 failed 553/600, **not 100%** as previously summarized. Other strong ON candidates remain ch1,4,7,8,10; full-channel exports preserve the selection context.
- Amp0 ch6: OFF 131/160 accepted with the original method settings; manual 135/160 with the audited focused crop (25 remain rejected). Ch10 retains its original method processing. These counts do not establish binding specificity. Raw changes near -0.43 to -0.44 V remain visible in the raw-trace audit, and some high-dose blocks lack accepted plateaus.
- Vanco ch6: all 600 accepted; ch2 OFF/manual each 200/200. The sole jump candidate is ch10 physical scan 155. Its raw CSV jumps from -0.470662 to -0.343809 V (0.126853 V versus typical 0.001186 V increments). A connecting line spans missing voltage samples; no intermediate measurements were created. Cause unknown. This is not on chosen ch2/ch6 channels.
- `<DATA_ROOT>\vanco_500um_planar_20260908_113415` currently contains metadata/logs but no local raw scans. Its earlier claimed poor response was not re-established, nor was it substituted for the available run.

### Exclusions and sensitivity analysis

Primary figures: initial buffer/duplicate-dose pair omitted from titration statistics; original acceptance rules retained except for the explicit Amp0 Channel 6 manual crop above. **No new jump-based exclusions, Q clipping, imputation or moving-average gap filling.** BO's five-point mean is a trend line, not replacement observations. SWV stride 10 is display sampling.

For a sensitivity branch: **Remove extreme titration outliers ON > Apply Display Controls**, regenerate and export separately. The retained settings are under `audit/recipes/<dataset>/filtered_sensitivity/`; create fresh CSV/audit exports if numerical sensitivity results are needed. Do not silently switch primary analysis to that branch. Exclusions need not be printed inside the image, but rules/counts must remain available in Methods/SI and the audit.

## 9. Templates, checks and remaining work

- All Composer presets are editable, including the supplied Type presets. They live in the visible, tracked `figure_composer_presets.json`; the historical `builtins` and `presets` sections are both writable from the app. SWV Types 3/3B/3C/4 are guided generators in `app.py` Paper Figures.
- Composer **Save / update preset** saves a template; `.figure.json` saves a composition; analysis sessions restore inputs/settings/optional derived results. They are different objects.
- Collaboration: send a Composer PNG plus its `.figure.json` or `.figure.zip` to reproduce a Composer figure on another machine; the app can load those directly in **Load a preset or saved figure**. A Paper Figure PNG, metadata JSON, or package can now be loaded through **Paper Figures → Load figure recipe**; it restores source folders, SWV/titration processing, selected steps, methods, and layout controls. The recipient still needs the recorded raw folders or an equivalent compatible copy. Never send/load an untrusted derived-results cache.
- To save or update any preset: apply pending panel/layout edits, keep its name under **Reusable preset**, then click **Save / update preset**. The same name updates it; a new name creates a copy. No Enter key, rendering, or first-save copy is required. Loading any preset fills in its exact name automatically. Alternatively use **Saved preset to overwrite > Overwrite selected preset**, including for Types 1/1A/1B/2/2A. Updates affect future preset loads, not existing session files or PNGs.
- Layout boxes include labels/legends; grid defaults to 1%, overlap is optional, export crops unused outer workspace. Workspace zoom does not alter export scale. Use the generated JSON for exact formatting.
- This run fixed session restoration of button events/JSON identity keys, final-export invalidation, extracted-folder direction lookup, stale Paper Figures downloads, mislabeled response axes and inconsistent method colors. Added app-visible raw/corrected QC and fitted-branch/full-series R² exports. Correction/fitting algorithms were not replaced.
- Refresh fixes: smoothed/corrected waveform clipping to final correction minima; one clearly labeled filter status; consistent extreme filtering in response plots and fits; selected-preset overwrite without name retyping. Full-analysis-crop display remains available explicitly.
- Automated tests (2026-10-11): **448 passed, 4 skipped** (one existing Matplotlib open-figure warning); the final legend/session changes also passed a **23-test targeted recheck**. Coverage includes baseline handling, jump correction, calibrated chronological ticks, cropped/full range, depth fading, PNG transparency, SWV thickness, local-session compatibility and concentration-summary placement. Five Type 2 PNGs/sidecars and three concentration-diagnostic PNGs were regenerated. A prepared titration workspace for each of the four datasets reopened through the actual app's Open button with cached results and figure settings restored. All 19 prepared recipes have accessible source paths and, where applicable, shared local caches. Browser/print-size inspection and scientific validation remain separate checks.
- Generated outputs/local caches remain Git-ignored; source changes and this guide are version-controlled. Committing the code does not upload raw data or these local PNGs.

Still needed: notebook immobilization/cleaning details, chip morphology confirmation and equilibration rationale; independent concentration validation, specificity controls, reversibility assessment and the planned BO-versus-random simulations for stronger claims.
# Updated BO Validation Export

The current release is [journal_plots.zip](generated/journal_plots.zip), combining
titration and Type 2/2A BO validation. Maximize channels: Kana 7, Vanco physical 10
(group 8), Amp0 9. Minimize channels: Kana 5, Vanco physical 2 (group 1), Amp0 6.
Selected minimum-Q optima are at 138, 322 and 347 Hz, respectively.
Amp0 stacks now use per-trace correction brackets to avoid off-bracket extremes;
recorded Q and peak currents are unchanged. Each newly rendered figure has
PNG/PDF/SVG and one `.metadata.json`; selection/processing audits are per figure.
Kana channels 3 and 5 also have two-row Type 3A titrations (app Type 3C).
This is a presentation selection, not independent assay validation. Amp0
titrations remain exploratory. Raw files and local saved sessions were unchanged.
