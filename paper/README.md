# BO paper: working guide

Updated 9 October 2026. This replaces the old handoff, STEP_BY_STEP, and storage/data-audit instructions. Raw data remain in `C:\TEMP\BO`. Start with the exact runbook below; the remaining sections explain the assessment and later work.

## 0. Start here: planar-kana sweep first

### Final software audit (9 October)

The real-input smoke checks loaded the kana sweep and all four BO experiments,
validated preset compatibility, and regenerated their selected BO validation
figures. The Streamlit widget path was exercised for sweep source editing and
BO stack edits (including changing and applying the displayed trace count).
Removing whole-experiment hashing from automatic OFF-example selection reduced
the measured warm sweep rerun from about 10.6 to 2.5 seconds in the local test
harness; timings are not a guarantee for every computer or final export.

Selected-channel titration processing used each experiment's saved analysis
settings and acceptance windows, with outlier removal **off**. Counts include
all methods on that physical channel, not just the optimized method:

| Dataset / physical channel | Analyzed scans | Accepted | Rejected |
|---|---:|---:|---:|
| Kana setup 5 / 5 | 600 | 594 | 6 |
| Kana station 2 / 10 | 600 | 571 | 29 |
| Amp0 / 10 | 480 | 241 | 239 |
| Vanco / 6 | 600 | 600 | 0 |

These are processing checks, **not new Kd/LOD results or a claim that every
channel responds**. Amp0's rejected scans must remain reported. Vanco channel 6
belongs to **BO Group 4**, not Group 6; selecting mismatched group/channel data
produces empty panels. The raw-data directories were not modified. Before the
paper handoff, inspect one final export at intended size and save/reopen one
named workspace in the actual browser. Automated tests do not replace that
browser/print-size check or independent validation of fitted concentrations.

### Latest reference-figure fixes

- **Shared custom templates:** save with **Figure Composer → Save preset**. The visible `figure_composer_presets.json` in the app root is tracked by Git; commit/push it to share subsequent edits with Max. Built-ins remain in code and generated exports remain ignored. Old hidden preset files are still read for compatibility.

- **Missing presets:** Types 1/1A/1B appear for survey sessions; Types 2/2A appear for optimization sessions. For kana validation, replace the `parameter_sweep...` path with the `planar_BO_kana...` BO path in Figure B below. Types 3/3B/4 are under **SWV → Paper Figures**.
- **Edit panel:** one editor contains content, formatting, progress highlights, layout, and optional interactive 3D. Use **Preview panel**, **Apply changes**, or **Cancel panel edits**. For cubes: enable **Interactive 3D view → Cache view → Use cached camera → Apply changes**. No tab switching. In a stack, **Maximum displayed traces** controls display sampling and rendering cost, not scientific analysis.
- See the [main README capability guide](../README.md#current-workflow-and-capabilities) for save/reopen limitations, outlier filtering versus display clipping, and missing-data handling. No missing measurements are synthesized.

- **Canvas boxes now describe the whole panel**, including titles, labels, legends and colourbars. Rendering fits those elements inside the box with a small safety margin. Ordinary 2D charts use the available width/height; cubes, square maps and embedded images retain their proportions. Very small boxes can reduce text size, so enlarge the box if needed.
- **Allow overlap**, below the manual layout canvas, is off by default. Touching edges are fine; overlapping boxes cannot be applied/rendered unless you enable it. Existing rectangles are not moved automatically. Export still crops unused outside canvas without deleting the deliberate gaps between panels. Explicit cross-panel zoom connectors remain cross-panel annotations.

- The Composer chronological stack now follows the paper's direction: early scans are faint at the upper-left/back; later scans progress toward the lower-right/front in darker blue/orange. Current is not inverted: only the display offsets change. Its title retains the displayed iteration range. Offset spacing indicates order, not elapsed time; no additional smoothing, normalization or scan selection is applied by this style. Refresh the app and click **Render figure (fast preview)** to update an existing composition. Only reload a preset if you also want to reset its layout.

- Type 1/1A/1B now default to **Raw / unsmoothed (corrected)**. In **Edit panel → B/C → Trace smoothing (after correction)** choose raw or smoothed; either retains saved BO baseline correction and the selected peak-bracket crop. Linked controls no longer reset this choice. Reload a preset for the new default; existing saved selections are preserved.
- **Edit panel → Show SWV example markers** independently hides/shows the coloured symbol in an SWV panel or the highlighted points and iteration callouts on a cube. Q colours, slice planes, waveform titles and selected data are unchanged.
- Type 2 has a larger chronological-stack panel with less internal whitespace and clearer early traces. Rejected/missing scans leave no display gaps or error-count footer; diagnostics remain attached to the generated source figure. Spacing shows chronological order, not elapsed time. Measured current/voltage arrays are unchanged. Reload the preset to adopt its larger rectangle; existing manual layouts stay untouched. Cube frequency labels sit slightly closer to the cube.

- Reload **Type 2** or **Type 2A**, then render again to apply the new preset geometry. Existing saved layouts are not overwritten. Type 2 uses a large 4:3 workspace: two trends upper-left, progression cube upper-right, chronological SWVs lower-left, parallel coordinates lower-right. Type 2A omits the bottom panels and uses a wide workspace.
- Q has a dashed five-point trailing mean; buffer/target remain blue/orange. Trend titles, black axes frames and upper-left boxed legends remain visible. The cube has separate Q and red-to-black iteration scales. Parallel coordinates highlight the **selected observation**, not an assumed optimum; select the desired observation before rendering. Step size is shown in mV.
- Select the same channel/direction throughout (for example `5_max` for kana channel 5 ON). Archived physical-channel trace names are reconciled with directional analysis names. The chronological stack uses saved BO analysis settings by default and shows the displayed iteration range without “subset” in the title. Its arrow reads **Iteration number**. Rejected scans are omitted rather than plotted uncorrected.
- SWV **Paper Figures** Types 3/3B/4 now export transparent, tightly cropped figures. **Show panel letters** is optional and off by default. These composites still contain rasterized panels: set **Panel raster DPI** before generating; PDF does not make those panels vector graphics.
- Type 3B includes ON, OFF and **ON minus OFF at matched positive concentrations**. The latter subtracts the baseline-processed plateau values, averaging repeated doses within each method. It is a descriptive difference curve, **not a third Langmuir fit**. No unmatched dose is extrapolated. Choose the baseline mode deliberately before generation.
- For Types 3/3B/4, the measurement range is a display crop. Fits use all included titration steps; change the included doses to change the calibration fit. Type 4 remains within-calibration prediction, not independent validation.

### Paper figure scope

| Dataset | BO validation | BO sweep | Titration response / concentration validation |
|---|---:|---:|---:|
| Planar kana, setup 5 | Yes | Yes | Yes |
| Amp0 | Yes | No | Yes |
| Vanco | Yes | No | Yes |
| Kana station 2 | Yes (replicate/SI) | No | No |

In other words: make the parameter-sweep figure only for planar kana; make titration Figures C/D for planar kana, amp0, and vanco; and make a BO-validation figure for every dataset, including kana station 2.

### 0.1 Launch

1. In PowerShell, run exactly:

   ```powershell
   cd "C:\Users\Asus\OneDrive\Desktop\Jun-Chau Lab\Chien Lab Scripts\Analysis Scripts\swv_app"
   .\Open_SWV_App.cmd
   ```

   Alternatively, double-click `Open_SWV_App.cmd`. Wait for the browser tab at `http://localhost:8501`.
2. To save work later, open **Saved analysis sessions** in the sidebar, enter **Save as**, optionally include the derived cache, and click **Save**. Raw CSVs are not copied; reopen through the same expander with **Saved session -> Open**.

### 0.2 Figure A — Parameter-sweep landscape or comparison (Type 1, 1A, or 1B)

Use this only for a **survey/parameter-sweep** session, not an optimization session.

1. Choose **Analysis mode -> BO Session** and paste:
   `C:\TEMP\BO\500um_planar_BO_try_again_20260918_112055\parameter_sweep_20260921_110003\bo_sessions\bo_111904_a0d8ad`
2. Open **Figure Composer -> Load a preset or saved figure**.
3. Pick one template:
   - **Type 1:** full 8-panel, 7 × 7 in standalone landscape.
   - **Type 1A:** spacious 9 × 10 in workspace with the cubes on the left.
   - **Type 1B:** mirrored Type 1A with support panels on the left.
4. For Type 1A/1B, choose one shared channel and two measured step-size planes. The ON example defaults to maximum paired Q. **Signal-off example selection** offers the recorded minimum or a **Peak-aligned example**: lowest negative Q among the 12 leading candidates with six accepted scans and median buffer/target peak positions within 50 mV. This changes the illustrated record, never the underlying Q data. For setup-5 sweep channel 10, the recorded minimum is iteration 167; the aligned example is iteration 42. Cube highlights and SWVs stay linked. Use **Edit panel → Marker size / Camera X/Y/Z / Camera distance** for each cube.

   SWVs default to **Use saved BO analysis settings**, including acceptance windows, and **Show only corrected peak bracket** (between the detected minima). This is a display crop after analysis. Uncheck it to see the full saved voltage crop. **Black SWV axes box** toggles the black frame; **Coloured panel border** controls an optional extra frame. Slice coloured borders follow the data axes, not the entire panel slot. Titles show iteration/parameters or step/channel, without ON/OFF wording.

   Exports use tight content bounds with a small margin and transparent background. Empty space outside the composition is removed; your spacing between panels is retained. Cube faces are transparent and step ticks are labelled in mV.
5. **Loading a preset does not render an image yet.** Confirm that the canvas and panel count changed (Type 1 = 8 panels; Type 1A/1B = 6 panels), scroll below the panel settings, then click **Render figure (fast preview)**. When the preview is correct, click **Create final PNG/PDF/SVG**. The compact templates have no A–F letters, keep a tight 2:1 cube/support width ratio, and permit exactly two linked slice planes.
6. Use Type 1A plus Type 1B only when you have two genuinely comparable sweep datasets (for example planar and nanoporous). Use Type 1 only with a loaded survey/parameter-sweep session: BO-only folders do not contain the required sweep landscape.

**To reproduce the latest checked sweep:** load Type 1A, choose **Shared channel = 10**, **Signal-off example selection = Peak-aligned example**, and planes **0.004 / 0.007 V**. Verify ON iteration **120** and OFF **42**. These sweep examples are not the channel-5 titration methods below.

1. In **Edit panel → A**, set marker size **6**, opacity **0.75**, Camera X/Y/Z **1.65 / 1.15 / 1.10**, Camera distance **1.0**, and Plot text **9**. Repeat for cube **D**.
2. In **B**, select **Raw / unsmoothed (corrected)** under **Trace smoothing (after correction)**. Keep **Corrected**, **Use saved BO analysis settings**, **Show only corrected peak bracket**, and **Black SWV axes box** on. Repeat for **C**. Raw means unsmoothed display, not uncorrected current; saved analysis smoothing still helps determine the baseline/peak.
3. To hide coloured example symbols, uncheck **Show SWV example markers** separately in **A, B and C**. Leave it on where you want the correspondence shown. Slice-plane colours and Q scales are unaffected.
4. Set **Plot text = 9** on the remaining panels; keep panel letters off and the **9 × 10 in** workspace. Render a fast preview. If satisfied, create the final exports and save the analysis session as `kana_sweep_ch10`.

```text
Type 1A                         Type 1B
┌──────────────┬────┐           ┌────┬──────────────┐
│ A cube       │ B ON│           │B ON│       A cube │
│              ├────┤           ├────┤              │
│              │ C OFF│          │C OFF│             │
├──────────────┼────┤           ├────┼──────────────┤
│ D cube/slices│ E/F│           │ E/F│D cube/slices │
└──────────────┴────┘           └────┴──────────────┘
```

### 0.3 Figure B — BO validation (Type 2 or Type 2A)

Use this figure to show that BO explored parameter space and improved/selected waveform quality. It uses the saved BO session directly—**no snapshot is needed**.

1. Choose **Analysis mode -> BO Session** and paste the appropriate session path below.
2. Choose the listed channel group and optimization direction.
3. Open **Figure Composer -> Load a preset or saved figure**.
4. Choose **Type 2 - BO validation** for the five-panel version, or **Type 2A - BO validation (focused)** for cube + two trends only. Click **Load preset**.
5. Set the cube, buffer/target, chronological SWV and parallel-coordinate panels to the same channel/direction (e.g. `5_max`). Use **Observation iteration** to choose the red dashed selection in parallel coordinates; it is not automatically the optimum.
6. Keep the dashed **5-point running mean** in Q_run vs iteration and leave display clipping off. Type 2 uses a large **4:3** canvas with no panel letters by default. Its chronological stack uses saved BO settings and packs accepted displayed scans together; retain a caption explaining that it is a subset.
7. Click **Render figure (fast preview)**. Inspect the stack and both Q/iteration colourbars, then **Create final PNG/PDF/SVG**. Save the session as `kana_BO_ch5_on` (or the matching dataset/channel). Loading a new preset replaces the working layout, so save before switching.

| Dataset | Session path | Group / direction |
|---|---|---|
| Kana setup 5 | `C:\TEMP\BO\500um_planar_BO_try_again_20260918_112055\planar_BO_kana_20260918_112056\bo_sessions\bo_113013_8dd581` | Channel 5; **maximize** |
| Amp0 | `C:\TEMP\BO\100um_amp0_high_conc_20260915_130213\amp0_100um_20260915_130234\bo_sessions\bo_145633_9f5760` | Channel 10; **minimize** |
| Vanco | `C:\TEMP\BO\vanco_first_try_20260826_132944\vanco_try_1_20260826_132945\bo_sessions\bo_142117_84662a` | **Group 4**, channel `6_min`; **minimize** (group IDs are not channel numbers) |
| Kana station 2 | `C:\TEMP\BO\500um_planar_kana_20260917_170330\500um_planar_kana_20260917_170349\bo_sessions\bo_171120_dc5dbc` | Channel 10; **maximize** |

```text
Type 2                         Type 2A (focused)
┌──────────────┬────────────┐   ┌───────────┬────────────┐
│ Q trend      │ cube/path  │   │ Q trend   │ cube/path  │
│ buffer/target│            │   │ phases    │            │
├──────────────┼────────────┤   └───────────┴────────────┘
│ SWV stack    │ parallel   │
└──────────────┴────────────┘
```

### 0.4 Figure C — Titration response (Type 3 or Type 3B)

This figure needs an SWV analysis first. Choose **Analysis mode -> SWV**, select the correct `titration_only` folder, and leave **Automatically load one uniquely matched snapshot when folders change** enabled. The app finds the BO snapshot, applies its processing settings, and still lets you edit any control. In **Peak / Baseline -> BO analysis match**, use **Check current settings** before analysis; use **Reload matched settings** only if you want to discard edits.

1. Click **Run Analysis**. Enable **Treat vline intervals as titration steps**, **Fit Langmuir-style curve to step plateaus**, and **Immediately preceding buffer**; then click **Apply Display Controls**.
2. Open **Paper Figures**.
3. Choose **Type 3** for manual versus one optimized method, or **Type 3B** for the fixed ON/manual and OFF/manual comparison with one shared optimized-only Langmuir panel.
4. For Type 3B, select one physical channel, one optimized signal-on method, one optimized signal-off method, and one manual/reference method; use the same display range for both rows. Generate and download PNG/PDF.

| Dataset | SWV folder | First physical channel / methods |
|---|---|---|
| Kana setup 5 | `C:\TEMP\BO\titration_only\kana_try` | Channel 5; ON 500 Hz/40 mV/2 mV, OFF 138 Hz/90 mV/3 mV, manual 200 Hz/36 mV/2 mV |
| Amp0 | `C:\TEMP\BO\titration_only\amp0` | Channel 10; minimize method and its manual method |
| Vanco | `C:\TEMP\BO\titration_only\vanco` | Channel 6; minimize method and its manual method |
| Kana station 2 | `C:\TEMP\BO\titration_only\kana_st2` | Channel 10; maximize method and its manual method |

```text
Type 3B
┌──────┬──────┬─────────┬─────────────────┐
│manual│  ON  │ON/manual│                 │
├──────┼──────┼─────────┤ shared ON/OFF   │
│manual│ OFF  │OFF/manual│   Langmuir     │
└──────┴──────┴─────────┴─────────────────┘
```

### 0.5 Figure D — Concentration validation (Type 4)

Use the same SWV analysis, physical channel, methods, and display range selected for Figure C. In **Paper Figures**, choose **Type 4 - Concentration validation**. For a Type 3B comparison, set **Comparison rows = 2**: optimized-ON versus manual in row 1 and optimized-OFF versus manual in row 2. Click **Generate paper figure** and download it. Its predicted-versus-known panel is fit self-consistency, not external validation.

## 1. Detailed SWV/titration reference for Figures C and D

After the BO figures, the next deliverable is a channel-5 titration results bundle plus two readable individual plots.

1. In the running app choose **Analysis mode: SWV**. Select only:
   `C:\TEMP\BO\titration_only\kana_try`
2. Set **Group plotted traces by: SWV settings**. Choose the folder before loading settings, because folder changes can reset crop controls.
3. Open **Peak / Baseline -> BO analysis match**. The matched setup-5 snapshot should already be shown and loaded automatically. Click **Check current settings** before running. If automatic matching is unavailable, paste this fallback path and click **Load analysis settings from BO config**:
   `C:\TEMP\BO\500um_planar_BO_try_again_20260918_112055\planar_BO_kana_20260918_112056\bo_sessions\bo_113013_8dd581\bo_config_snapshot.json`.
4. Verify crop **-0.55 to 0.00 V**, smoothing **15 / 2**, minima window **0.30 V**, minimum start voltage **-0.6 V**, double correction **on**, minimum peak **0.001 uA**, prominent minima **on**, and **Apply BO acceptance windows on**. Peak window: **-0.45 to -0.10 V**; left minimum: **-0.54 to -0.10 V**; right minimum: **-0.45 to -0.01 V**; require local minima on both sides **on**. In Experimental, leave wavelet correction and background recentering **off**. The loader does not reset every experimental option.
5. Click **Run Analysis** once and wait for completion. Use **Channels to plot: 5** for inspection; avoid restricting the analysis to selected doses or favorable scans.
6. In Scan Annotations verify **Autotitration detected** and alternating buffer/target intervals. All four titration-only folders now contain `session_log.txt`; manual vline files are absent from the current analysis-plan folder. Check that boundaries land between blocks before fitting. Do not substitute guessed vlines if detection fails.
7. Enable **Treat vline intervals as titration steps**, **Fit Langmuir-style curve to step plateaus**, and choose **Immediately preceding buffer**. Use **uM**. Exclude only the first duplicate `20 uM_1` block for the proposed equilibration analysis; retain the second 20 uM block and the needed preceding buffer. Record this exclusion and its experimental justification. Keep the full time course visible. Click **Apply Display Controls**.
8. In **Metrics**, select peak current and combine the three channel-5 methods. Identify methods by settings, not their display order:

   | Role from existing records | Frequency | Amplitude | Step |
   |---|---:|---:|---:|
   | Optimized ON | 500 Hz | 40 mV | 2 mV |
   | Optimized OFF | 138 Hz | 90 mV | 3 mV |
   | Manual | 200 Hz | 36 mV | 2 mV |

9. Record the peak-height source, plateau edge trim, and outlier setting. The new example uses **corrected + smoothed** peak current. For the first verification leave extreme-outlier removal off and retain failures in the exported results; if a sensitivity comparison is needed, save it separately. Confirm each method has its own approximately 200-measurement axis, rather than fitting the 600 interleaved scans as a single method.
10. Inspect the full current time course and Langmuir plot. Check buffer returns, early equilibration, failed points, and whether the high doses actually flatten. A fit alone does not establish reversibility or saturation.
11. **Before switching views**, use Export to save the data bundle and download the individual time-course and Langmuir plots. Keep the results, titration-step table, fit summary, and processing inputs together. Save under a new run name; do not overwrite an earlier result. The export metadata records BO acceptance-window values and the loaded config path, but also retain the exact BO config: a path can become unavailable after moving data. The copied config in `paper/reference/` is supplementary provenance, not proof that the widgets were unchanged.
12. Compare the new fit table with the old example's printed LODs: ON **22.7 uM**, OFF **75.2 uM**, manual **8.96 uM**. These are values visible in an existing PNG, **not independently reproduced from raw scans in this assessment**. If they differ, compare peak source, interval selection, trim, outlier setting, and buffer selection before changing anything to force agreement.

Stop point: one saved bundle, one full time course, one three-method fit plot, and documented settings. This is a useful completed unit even if Composer needs further work.

## 2. Assessment of the new software and outputs

Reviewed the uncommitted diff against HEAD `951c5b7`, the current notes, render helper/logs, actual config, ranking CSV, and BO/titration/landscape PNGs. The following post-review fixes were applied: linked Type-1 sweep controls, an autosave-recovery naming correction, non-clipping BO-validation defaults, slice-plane colour propagation, cache source-file validation, and BO-window/config provenance in export metadata.

| Change / finding | Assessment and consequence |
|---|---|
| BO settings loader and acceptance windows | Useful. Reuses the BO rejection function and includes windows in the analysis cache key. However, experiment-folder selection picks the largest state file, not a scientifically selected session. Use an exact config path. |
| Loader says it reproduces analysis exactly | Too strong: it does not reset all experimental controls, and missing config window values can leave defaults or previous values. For these runs check the populated controls and keep wavelet/recentering off. |
| Export provenance | The manifest now records whether BO acceptance windows were applied, their exact values, and the loaded snapshot path. Preserve the snapshot itself alongside each bundle too: paths can become unavailable after moving data. |
| Add to Composer and axis limits | Useful additions. Captures are session objects, so save individual plots before switching modes and inspect panel assignments. The log prints only the newest captures; it does **not** prove older captures were lost. The actual titration PNG contains all seven panels. |
| BO preset metric fallback | The preset asks for peak height, but the UI filters options to objective-related metrics and silently replaces an unavailable selection. The exported BO panel B is **Repeat-scan SNR**, not peak height. The previous guide acknowledges this in one place but contradicts itself elsewhere. Use the actual metric label; do not describe this panel as current. |
| SI preset direction | Both new BO presets default to **maximize**. For amp0/vanco signal-off examples explicitly choose **minimize** after loading. A preset name does not establish the right channel or direction. |
| Percentile clipping | The new Type-2 BO-validation preset now defaults to clipping off. Other templates may still expose clipping as an explicit display control. It changes what the reader sees and is not equivalent to justified outlier handling. |
| Landscape example | Trace panels are labeled **Ch 10**, with the same-looking data shown once in current and once normalized. Maps/colorbars show **Repeat-scan SNR**. It is not the promised channel-3/channel-5, best/worst-Q comparison. Choose channels, metric, iterations, slice values, and color limits explicitly. |
| Titration figure | A useful exploratory result, but seven panels, long method labels, redundant diagnostics, and separate overlay colorbars make it too busy. The ON colorbar ends at 194 while the other two end at 200; shared color semantics have not been established. |
| Print-size and font changes | Potentially useful; high DPI does not solve crowded layout. Judge at final physical size. The canvas names are software presets, not verified requirements of the eventual ACS journal. |
| Chronological stack | Better bounded and limited to 60 traces by default, but therefore a subsample. State the thinning/offset rule if retained; use simple selected overlays if the stack does not communicate an additional result. |
| Old helper | `make_titration_only.py` depends on an external Claude scratchpad CSV path and does not itself recreate all the added log provenance. Preserve as history; do not rerun to rebuild the datasets. |

Code anchors: `app.py` (`_load_bo_analysis_into_sidebar`, `build_export_metadata`); `bo_session_viewer.py` (`_composer_bo_optimization_preset`, `_composer_bo_compact_preset`, `_composer_hyperparameter_sweep_preset`, `_q_relevant_metrics`, `_composer_apply_config`).

Verification: **299 tests passed, 1 skipped** using `.venv310` and `MPLBACKEND=Agg` after the post-review fixes. The normal launcher uses `.venv`; that environment has no pytest. `.venv310` has pytest, but its default Tk plotting backend fails because Tcl is missing. The headless backend is required for non-GUI test execution. Passing tests do not validate scientific choices or reproduce titration fits.

## 3. What is actually established versus still provisional

Verified locally today: titration-only CSV counts are **kana_try 6,000; kana_st2 5,600; amp0 4,800; vanco 4,800**, and each has a session log. This is a file count, not a fresh queue-by-queue completeness audit. The setup-5 snapshot confirms the analysis controls above and group exploration **0.7**, initial budget **0**, candidate pools **1000/100**, GP falloff entries **0.2**, and seed **42**.

The old ranking CSV and the new plot do not use identical reported numbers. For channel 5, the CSV lists ON/OFF/manual maximum responses **0.062 / -0.012 / 0.023 uA**; the new narrative reports approximately **0.068 / -0.014 / 0.022 uA**. The CSV lists ON Kd **412 uM**, R2 **0.99**, SNR **65.4**. These are historical table entries, not regenerated results. Do not mix that table with the new plots until the saved settings/definitions reconcile them.

The existing titration PNG shows a larger ON current response but a **higher estimated LOD than manual**. Its predicted-versus-known panel uses the fitted calibration data; the printed RMS fold errors do not demonstrate independent predictive performance. Larger response, higher SNR, and lower LOD are separate claims.

Kana setup 5 remains the sensible first dataset based on the available plots and historical ranking. Treat channel 5 as an illustrative example, then export a table for **all ten channels and all available methods**, including failure rates and failed fits. Responsive-channel SI panels can supplement that table; they should not hide channels 2, 6, and 9. Do not call a failed channel physically dead without additional evidence.

Other dataset claims (amp0 detectability rescue, vanco response strength, station-2 noise cause), exact Kd/LOD/SNR ranges, queue completeness, and the cross-group claim about iteration 47 were not independently recomputed in this assessment. The meaning of `np`, surface chemistry, and mechanism claims still need notebook evidence. Do not attribute poor response to burn-off or aptamer retention from these plots alone.

## 4. Assemble figures only after section 1 passes

### Current figure templates and the right workflow

The app deliberately has two figure tools:

**Editing content without losing layout:** select a panel (click its layout
rectangle or use **Edit panel**). Content and formatting live in that same panel
editor; there is no separate Edit source button. **Preview panel** renders just
the selected panel; **Apply changes** refreshes the composition. **Cancel panel
edits** restores settings to the last Apply, or when the panel was selected.
Switching panels keeps current settings; use Cancel first if you want to discard them.
For cubes, enable **Interactive 3D view**, drag/scroll, then **Cache view → Use
cached camera → Apply changes**. Orbit and zoom transfer to the publication cube;
arbitrary Plotly pan/roll do not. The interactive view is opt-in to avoid unnecessary
rendering latency. In linked sweeps, channel,
trace iteration and slice edits also update their cube links. **Reset source
iteration choices** returns to automatic example selection.

For Type 2/2A, choose **Validation group / channel (all panels)** and
**Validation channel** in Composer, plus the maximize/minimize optimizer.
All validation panels follow that selection; Q_run is the saved group objective,
not a concatenation of every channel's iterations.

New Type 2 presets use unsmoothed, corrected SWVs with the saved BO analysis
settings, thicker stack lines, and an iteration arrow that remains visible
outside the trace bounds. Type 2/2A cubes start at camera X=1.65, Y=1.15,
Z=1.10. Existing saved figures keep their processing/camera choices: use the
stack's **Trace processing** control to choose `corrected_current`, or reload
the preset after saving your current layout if you want the new defaults.

**Validation progress controls:** select any supported panel and use its inline
**Content and appearance · progress highlights** controls. These work in custom
compositions as well as presets. Toggle the best-observed-Q marker and change
its color; Q panels can add a best-so-far curve; cubes can fade early connections,
change path width, and mark start/final observations. Stack controls include line
width, oldest-trace opacity, and left/down offsets for the iteration arrow.
Existing camera, marker-size, fonts, axes, border and layout controls remain available.
Imported raster images cannot expose original data-generation settings.

Validation presets default to **Peak prominence**. Peak height (µA) remains
selectable, with optional recorded replicate SD (not SEM) for one channel.
Saved prominence scores can be zero when peak detection fails; the connected
line follows those recorded scores, not measured zero-current peaks. Peak height
has gaps where detection produced no value; these are not filled or interpolated.
Start and Final use distinct boxed callouts and marker shapes on the cube;
their horizontal/vertical offsets are editable in the panel's progress controls.
The magenta diamond identifies the same best
recorded Q iteration throughout the figure, with minimum Q for signal-off and
maximum Q for signal-on. Ambiguous multi-group/direction data are not assigned
a common best iteration. Display sampling retains available traces of that best
iteration in the chronological stack. No missing traces or responses are invented.

The kana setup-5 snapshot uses pairwise repeat-scan SNR for paired Q, with a
0.001 µA noise floor. Peak prominence, by contrast, is peak current divided by
background RMS. The largest absolute peak or buffer–target difference need not
maximize Q because repeat variability also matters. Neither a BO walk nor a
best-so-far curve alone proves superiority over random search or performance on
every aptamer; use the independent titration comparison and appropriate controls.

**Kana example units:** the local `mini_bo_g7_i47` (BO channel 7, iteration 47)
has raw target maxima about 0.239 µA (239 nA), and corrected target peaks
about 0.078 µA using its saved analysis settings. The sweep channel 10,
iteration 120 example has corrected target peaks about 0.036 µA. These are
different channels/runs and raw versus corrected values must not be conflated.
Do not substitute a BO trace into a sweep figure whose cube contains survey points.
A larger sweep-only candidate is **channel 5, iteration 61**: rerunning its six
CSVs with the saved settings gives corrected target peaks 0.1239–0.1245 µA
and buffer peaks 0.0939–0.0970 µA. This is a candidate, not an automatic
replacement: select examples by paired response and repeatability as well as
peak height, and retain the matching channel/iteration on the cube.

For plots made in other plotting tabs, **Add to Composer** retains its existing
behavior. **Replace selected Composer panel A/B/...** replaces only the selected
panel's content while retaining its layout; replacing a linked sweep panel turns
the composition into an independent custom layout. Captured plots do not yet
support reopening every original tab control automatically.

The editor defaults to **Workspace zoom → Fit width** and a **1% grid**.
It fills the available width uniformly; tall figures scroll vertically rather
than becoming narrow thumbnails. Use **Fit whole figure** for an overview or
**125–200%** for closer editing. This view-only zoom preserves panel proportions,
layout coordinates, and exported PNG dimensions. Exports still crop to the
figure's content, not the visible editor workspace. Physical figure-size controls
remain separate and do affect rendering scale. Cube highlight
callouts identify iterations; red circles and blue diamonds match the symbols
in the corresponding SWV panels. Channel titles follow the cube top.

Layout editing: drag any of the eight edge/corner handles to resize. Enable
**Snap** and choose the grid spacing. Select a panel to enter X/Y/width/height
as percentages in the **Selected block** row; press Tab or click outside to
finish entering a value. Increasing width/height moves the block inward if
needed instead of silently reducing your requested size. Y starts at the bottom.
**Square** makes a square at the actual printed canvas proportions.
**Copy / Paste** (Ctrl/Cmd+C/V
while focused on the layout canvas) duplicates plot settings as well as geometry,
up to 12 panels. Copies switch a linked sweep template to a custom composition.
Use Ctrl/Cmd-click for multiple selection and right-click for alignment or equal
sizes. **Ctrl/Cmd+Z** undoes moves, resizing, square and alignment edits;
**Ctrl+Y** or **Ctrl/Cmd+Shift+Z** redoes them. This geometry history is local to
the open editor; adding/deleting panels or loading a different layout clears it.
It does not undo plot-content changes or panel deletion. Normal text-field
shortcuts remain native while typing. Selecting a block no longer triggers an
app rerun; use **Edit plot settings** explicitly to apply the layout and open
that panel's settings. **Reset edits** discards the draft and restores the latest
server layout. Click **Apply layout** before rendering. Type 1 SWV panels have no coloured
outer frames by default; reload the preset to replace older saved defaults.

**Type 2 uses the loaded BO optimization session**, including its iteration
history and paired buffer/target measurements. Choose the channel's group and
maximize (ON) or minimize (OFF). It is appropriate for kana, amp0 and vanco;
no parameter sweep is required. Type 2A retains only the cube, Q trend and
buffer/target trend. A sweep session belongs in Type 1 instead.

- **BO Session → Figure Composer** builds Types 1 and 2. Use **Edit panel** or click a layout rectangle to select a panel. The static rendered preview is not clickable. Content, appearance and layout are edited together in that panel. Type 1 links channel/iteration/slice choices to its cube markers, SWVs, planes and maps.
- **SWV → Paper Figures** builds Types 3 and 4 from a selected physical channel and two methods. It is a guided, reproducible template rather than a fully free-form panel editor: the row, methods, scan range, stack/overlay choice, font, width, and raster resolution are adjustable, but individual generated subpanels are not yet independently restyled. For fully individual titration panels, create/capture the underlying plots and use the Figure Composer.

| Template | Where to open it | Use it for |
|---|---|---|
| Type 1 – Parameter sweep | BO Session, **survey** session only | Standalone eight-panel setup-5 kanamycin landscape. The station-2 sweep can be supplementary, not a required main figure. |
| Type 1A / 1B – Sweep comparison | BO Session, **survey** session only | Mirrored layouts: cube-left (1A) and support-left (1B). Choose a shared channel, two planes, and the OFF example-selection rule. Large workspace; content-cropped export. |
| Type 2 – BO validation | BO Session, BO experiment session | setup-5 kana, amp0, vanco, and optionally station-2 kana. |
| Type 2A – BO validation (focused) | BO Session, BO experiment session | The Type 2 cube and the two essential quantitative trends only; omit the chronological stack and parallel coordinates. |
| Type 3 – SWV and titration response | SWV → Paper Figures | each selected physical channel: manual/reference vs one optimized method, traces + time course + Langmuir response. |
| Type 3B – Signal-on/off shared Langmuir | SWV → Paper Figures | One physical channel with fixed ON/manual and OFF/manual rows; one optimized-ON/OFF-only Langmuir panel spans the fourth column. |
| Type 4 – Concentration validation | SWV → Paper Figures | the same selected channel/method pair: concentration-by-measurement and predicted-vs-known. |

### Recommended first-pass figure set

1. **Kana setup 5 (main paper):** make all four types.
   - **Type 1:** open the setup-5 survey session below. Use shared channel **5** for the main figure; make a second version with channel **3** only if it adds useful SI evidence. Keep four well-spaced, actually measured step-size planes (the default 1/4/7/10 mV-style choice is appropriate when present). Pick two real sampled iterations—one low-quality and one high-quality—and verify the displayed traces match the highlighted markers.
   - **Type 2:** open the setup-5 BO session, select the group whose channel list contains **5**, and select **maximize**. Keep Q_run plus its 5-point dashed running mean and use the honest displayed metric label in the buffer/target panel. Do not enable display clipping for the first export.
   - **Types 3 and 4:** analyze `titration_only\kana_try`; use physical channel **5**. Select optimized-ON (500 Hz, 40 mV, 2 mV) against the 200 Hz/36 mV/2 mV manual method. Make a separate optimized-OFF-versus-manual version only if it strengthens the SI story.
2. **Amp0 (SI):** make Types 2–4, not Type 1. For Type 2 choose the group containing channel **10** and **minimize**. For Types 3/4 analyze `titration_only\amp0`, select physical channel **10**, then select the method labelled **minimize** and the 200 Hz manual/reference method. Preserve failed scans: the point is detectability rescue, not a falsely clean calibration.
3. **Vancomycin (SI):** make Types 2–4, not Type 1. For Type 2 choose the group containing channel **6** and **minimize** (channel 2 or 10 can be a documented alternative). For Types 3/4 analyze `titration_only\vanco`, use that same physical channel, the method labelled **minimize**, and its manual/reference method. Treat a lower-bound Kd or weak saturation as an uncertainty, not a fitted result to optimize visually.
4. **Kana station 2 (replicate/SI):** Type 2 is useful; Types 3/4 are optional after the main set. Choose the group containing channel **10**, **maximize**, and then channel 10 in `titration_only\kana_st2`. Do not reuse setup-5 crop or acceptance settings.

### Formatting defaults

- Original Type 1 uses a 7 × 7 in canvas. Type 1A/1B use a 9 × 10 in workspace; Type 2 uses 12 × 9 in (4:3), and Type 2A uses 12.8 × 7.2 in (16:9). The newer presets start with 10 pt text; use 9 pt for the checked sweep example. Canvas size is editable and export crops to content. Use **Edit panel** to change one panel; use **Expand all panel settings** only for an audit.
- For Type 1, use the shared controls, not per-panel overrides: choose one channel, two highlighted iterations, and up to four real step-size planes. The colour frames are intended to identify these links.
- For Type 2, retain the 5-point Q_run running mean; choose a channel-specific group and correct direction before rendering. If peak prominence has extreme values, inspect them rather than clipping them away.
- For Type 3, start with one comparison row, 10–12 in width, 8–10 pt font, 300 panel raster DPI, and no optional columns. Type 3B fixes two rows and a shared fourth column. Panel letters are optional; exports are transparent. Use stacked traces only when temporal progression matters; otherwise use overlays. Verify readability again at final manuscript placement size.
- For Type 4, use the same channel/method choices as Type 3, 10–12 in width, 8–10 pt font, and 300 panel raster DPI. The predicted-versus-known diagnostic uses data also used to fit the calibration, so it is descriptive rather than external validation.

1. **Main titration figure:** start with four panels: full-width channel-5 time course; three-method Langmuir response; optimized-ON SWV overlay; manual SWV overlay. Put OFF overlays, SNR, inverse-calibration diagnostics, and the all-channel table in SI. Keep the low-dose time course visible; omit equilibration only from the stated fit/statistics selection. A shared overlay colorbar requires the same limits and coordinate definition; otherwise keep clearly labeled separate bars.
2. **Composer workflow:** use the guided Type 3 template for a reproducible two-method comparison, or capture individual Metrics/Overlays plots and use a manual Composer layout when every titration subpanel needs independent styling. Verify all sources and render at final size. Save PNG/PDF plus the available metadata/portable export. A PNG or JSON config alone does not guarantee restoration of in-memory editable captures in a fresh browser session.
3. **Main BO figure:** open the setup-5 session; select Group 5 and maximize. Use Type 2 for Q_run with trailing mean, buffer/target peak prominence, a progression cube, chronological SWVs and parallel coordinates. Use Type 2A when the latter two panels are unnecessary. Set clipping off. Export minimize separately if showing signal-off optimization. The red dashed parallel-coordinate line identifies the selected observation, not an automatically verified optimum.
4. **Landscape:** open setup-5 sweep below. Explicitly choose channels 3 and 5 in each panel. Use paired Q if available and appropriate, and confirm colorbar labels. Set step slices explicitly to **0.001, 0.004, 0.007 V**, documenting slice tolerance/interpolation and showing sampled points. Use common color limits for directly comparable maps. Identify best/worst observations from the selected channel's history and verify the trace filename, iteration, phase, and waveform. Normalizing the same trace is not a best/worst comparison. Interpolated smooth maps are not additional measured data.
5. **Other datasets:** repeat section 1 with each exact snapshot, saving separate bundles. Use kana station 2 as a replicate with its different crop; amp0 and vanco as secondary results. Use minimize for the proposed amp0/vanco signal-off BO panels. Report unsuccessful channels and uncertain/nonsaturating fits alongside selected illustrations.
6. **Simulations last:** first define the benchmark: same objective/landscape, domain, observation noise, evaluation budget, failure handling, and multiple recorded seeds for BO and random search. Count initialization within the budget. Vary initialization while holding other settings fixed. Save per-seed histories and aggregate variability. A simulated benchmark must be labeled simulated; the existing 200-point survey is not by itself a repeated BO-versus-random benchmark. The old ten-panel "Hyperparameter Sweep" figure preset is a landscape layout, not a simulation experiment.

## 5. Paths and handoff

All paths below are relative to `C:\TEMP\BO`, except `paper/` paths which are in the software repository.

| Dataset | SWV folder | Exact BO session folder |
|---|---|---|
| Kana setup 5 | `titration_only\kana_try` | `500um_planar_BO_try_again_20260918_112055\planar_BO_kana_20260918_112056\bo_sessions\bo_113013_8dd581` |
| Kana station 2 | `titration_only\kana_st2` | `500um_planar_kana_20260917_170330\500um_planar_kana_20260917_170349\bo_sessions\bo_171120_dc5dbc` |
| Amp0 | `titration_only\amp0` | `100um_amp0_high_conc_20260915_130213\amp0_100um_20260915_130234\bo_sessions\bo_145633_9f5760` |
| Vanco | `titration_only\vanco` | `vanco_first_try_20260826_132944\vanco_try_1_20260826_132945\bo_sessions\bo_142117_84662a` |

Append `\bo_config_snapshot.json` to load SWV settings. Station 2 uses crop -0.45 to 0.10 V, minimum start -0.5 V, and different acceptance windows. Vanco disables prominent/required local minima. Load each snapshot rather than carrying setup-5 settings forward.

Setup-5 sweep: `500um_planar_BO_try_again_20260918_112055\parameter_sweep_20260921_110003\bo_sessions\bo_111904_a0d8ad`.

Station-2 sweep: `500um_planar_kana_20260917_170330\500um_planar_kana_param_sweep_20260921_111611\bo_sessions\bo_112206_86a834`.

Local organization: `paper/README.md` is the only active paper guide; `paper/reference/` holds historical ranking/configs; `paper/examples_previous/` holds existing draft figures; `paper/helpers_previous/` holds historical scripts; `paper/archive/` holds superseded notes. Historical copies are not final outputs. Raw CSVs and experiment folders are not migrated. The app's general README remains unchanged.

Cleanup completed: the original `AI_HANDOFF_PROMPT.md`, `STEP_BY_STEP.md`, `audit/bo_paper_data_audit.md`, and `audit/storage_audit.md` were removed only after each archived entry matched its original SHA256 hash. Restore any of them from `archive/superseded_notes_20261007.zip`. The original non-Markdown supporting files under `C:\TEMP\BO\analysis_plan` remain available; their useful copies are here. Existing application edits were preserved.

Still needed from the author, when writing Methods: actual manuscript/SI paths, target journal, notebook surface chemistry and chip identity, and the experimental rationale for treating the first dose as equilibration. These do not block the first verification/export.
