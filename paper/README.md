# BO paper: working guide

Updated 8 October 2026. This replaces the old handoff, STEP_BY_STEP, and storage/data-audit instructions. Raw data remain in `C:\TEMP\BO`. Start with the exact runbook below; the remaining sections explain the assessment and later work.

## 0. Start here: planar-kana sweep first

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
   - **Type 1A:** 3.3 × 7 in comparison half with the cube on the left.
   - **Type 1B:** mirrored Type 1A with support panels on the left.
4. For Type 1A/1B, use **Compact Type 1 linked controls**: choose one channel, then choose the real signal-on and signal-off observations and two measured step-size planes. The highlights, framed SWVs, planes, and maps stay linked.
5. Use Type 1A plus Type 1B only when you have two genuinely comparable sweep datasets (for example planar and nanoporous). Do not label arbitrary survey observations as signal-on/off without checking their files/settings.

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
5. Keep the dashed **5-point running mean** in Q_run vs iteration and leave display clipping off for the first export. Render, inspect, and download PNG/PDF.

| Dataset | Session path | Group / direction |
|---|---|---|
| Kana setup 5 | `C:\TEMP\BO\500um_planar_BO_try_again_20260918_112055\planar_BO_kana_20260918_112056\bo_sessions\bo_113013_8dd581` | Channel 5; **maximize** |
| Amp0 | `C:\TEMP\BO\100um_amp0_high_conc_20260915_130213\amp0_100um_20260915_130234\bo_sessions\bo_145633_9f5760` | Channel 10; **minimize** |
| Vanco | `C:\TEMP\BO\vanco_first_try_20260826_132944\vanco_try_1_20260826_132945\bo_sessions\bo_142117_84662a` | Channel 6; **minimize** |
| Kana station 2 | `C:\TEMP\BO\500um_planar_kana_20260917_170330\500um_planar_kana_20260917_170349\bo_sessions\bo_171120_dc5dbc` | Channel 10; **maximize** |

```text
Type 2                         Type 2A (focused)
┌─────────┬───────┐            ┌──────────────┬───────┐
│ A cube  │ B Q   │            │    A cube    │ B Q   │
│         ├───────┤            │    + path    ├───────┤
│         │ C phase│           │              │ C phase│
├─────────┼───────┤            └──────────────┴───────┘
│ D stack │ E ||| │
└─────────┴───────┘
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

- **BO Session → Figure Composer** builds Types 1 and 2. It has an **Edit panel** selector: choose A/B/C… to open that panel's plot type, data source, axes, colours, text, and layout settings. The layout canvas is for moving/resizing panels. It does not currently open a settings pane merely by clicking the rendered preview. Type 1 has intentionally linked channel/iteration/slice controls, so its highlighted cube points, SWV traces, planes, and maps cannot disagree.
- **SWV → Paper Figures** builds Types 3 and 4 from a selected physical channel and two methods. It is a guided, reproducible template rather than a fully free-form panel editor: the row, methods, scan range, stack/overlay choice, font, width, and raster resolution are adjustable, but individual generated subpanels are not yet independently restyled. For fully individual titration panels, create/capture the underlying plots and use the Figure Composer.

| Template | Where to open it | Use it for |
|---|---|---|
| Type 1 – Parameter sweep | BO Session, **survey** session only | Standalone eight-panel setup-5 kanamycin landscape. The station-2 sweep can be supplementary, not a required main figure. |
| Type 1A / 1B – Sweep comparison | BO Session, **survey** session only | Compact mirrored one-column halves for a planar/nanoporous side-by-side comparison: cube-left (1A) and support-left (1B). Select real signal-on/off observations and two measured step planes with the linked controls. |
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

- Types 1 and 2 load as 7 × 7 in ACS two-column square figures, Arial 7 pt, 9 pt panel letters, journal styling, and 600 DPI. Type 1A/1B load as 3.3 × 7.0 in ACS one-column tall halves. Leave those settings unchanged for the first render. Use **Edit panel** to change a single panel; use **Expand all panel settings** only for a final audit.
- For Type 1, use the shared controls, not per-panel overrides: choose one channel, two highlighted iterations, and up to four real step-size planes. The colour frames are intended to identify these links.
- For Type 2, retain the 5-point Q_run running mean; choose a channel-specific group and correct direction before rendering. If peak prominence has extreme values, inspect them rather than clipping them away.
- For Type 3, start with one comparison row, 7.0 in width, 7–8 pt font, 300 DPI, and no optional columns. Use stacked traces only when temporal progression matters; otherwise use overlays. Add SNR or predicted-vs-known only in the SI.
- For Type 4, use the same row/method choice as Type 3, 7.0 in width, 8 pt font, and 300 DPI. The predicted-versus-known diagnostic uses data also used to fit the calibration, so it is descriptive rather than external validation.

1. **Main titration figure:** start with four panels: full-width channel-5 time course; three-method Langmuir response; optimized-ON SWV overlay; manual SWV overlay. Put OFF overlays, SNR, inverse-calibration diagnostics, and the all-channel table in SI. Keep the low-dose time course visible; omit equilibration only from the stated fit/statistics selection. A shared overlay colorbar requires the same limits and coordinate definition; otherwise keep clearly labeled separate bars.
2. **Composer workflow:** use the guided Type 3 template for a reproducible two-method comparison, or capture individual Metrics/Overlays plots and use a manual Composer layout when every titration subpanel needs independent styling. Verify all sources and render at final size. Save PNG/PDF plus the available metadata/portable export. A PNG or JSON config alone does not guarantee restoration of in-memory editable captures in a fresh browser session.
3. **Main BO figure:** open setup-5 session below; select Group 5 and maximize. Start with Q_run/best-so-far, phase metric with its honest label, and selected buffer/target traces. Add either parameter parallel coordinates or a 3D view only if informative. Set clipping off. Export minimize separately if showing the signal-off optimization. Verify any claimed optimum within the exact group/direction; do not substitute another group's global maximum.
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
